import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import json
import numpy as np
from .utils import is_main_process

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("[Warning] DeepSpeed not available. Using regular PyTorch training.")


class UniGCRTrainer:
    def __init__(self, config, args, model, train_loader, val_loader=None):
        self.config = config
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader

        # label_smoothing=0.1 reduces overconfident predictions on the 4-layer code space
        self.gr_criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

        if config.enable_ctr:
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.ce_loss  = nn.CrossEntropyLoss()

        self.use_distributed = dist.is_available() and dist.is_initialized()

        use_deepspeed = (
            DEEPSPEED_AVAILABLE and
            (hasattr(args, 'deepspeed_config') and args.deepspeed_config or self.use_distributed)
        )

        if use_deepspeed:
            print("[Trainer] Using DeepSpeed")
            self.lr_scheduler = None
            ds_config = None
            if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
                try:
                    with open(args.deepspeed_config, 'r') as f:
                        ds_config = json.load(f)
                except Exception as e:
                    print(f"[Trainer] Error loading DS config: {e}")

            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                args=args, model=model, model_parameters=model.parameters(),
                config=ds_config, dist_init_required=self.use_distributed
            )
            self.model = self.model_engine.module if hasattr(self.model_engine, 'module') else self.model_engine
            self.device = self.model_engine.device
            self.use_deepspeed = True
        else:
            print("[Trainer] Using regular PyTorch (single-GPU)")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = model.to(self.device)

            lr = getattr(config, 'lr', getattr(args, 'learning_rate', 1e-4))
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-5)
            self.model_engine = None
            self.use_deepspeed = False
            print(f"[Trainer] Device: {self.device}, LR: {lr} → 1e-5 (cosine)")

        # Automatic Mixed Precision
        if not self.use_deepspeed and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                self.amp_dtype  = torch.bfloat16
                self.amp_scaler = None
                print("[Trainer] AMP: bfloat16")
            else:
                self.amp_dtype  = torch.float16
                self.amp_scaler = torch.amp.GradScaler('cuda')
                print("[Trainer] AMP: float16 + GradScaler")
            self.use_amp = True
        else:
            self.use_amp    = False
            self.amp_dtype  = torch.float32
            self.amp_scaler = None

    def calculate_ctr_loss(self, ctr_logits, ctr_labels):
        target_idx  = torch.argmax(ctr_labels, dim=1)
        loss_info   = self.ce_loss(ctr_logits / self.config.temp, target_idx)
        loss_bce    = self.bce_loss(ctr_logits, ctr_labels)
        return self.config.loss_alpha * loss_info + self.config.loss_beta * loss_bce

    def _layer_offsets(self):
        """Return per-layer token offsets consistent with GridMapper."""
        c = self.config.sem_id_codebook_size
        return [1 + i * c for i in range(self.config.sem_id_layers)]

    def train_epoch(self, epoch_idx):
        self.model.train()

        # Scheduled teacher forcing:
        #   Epochs  1-20 : TF = 1.00  (model learns the code space before any AR pressure)
        #   Epochs 21-80 : TF linearly 1.0 → 0.1  (gradual exposure bias reduction)
        #   Epochs 81+   : TF = 0.10  (floor prevents full cascade collapse)
        if epoch_idx <= 20:
            tf = 1.0
        elif epoch_idx <= 80:
            tf = max(0.1, 1.0 - (epoch_idx - 20) / 60)
        else:
            tf = 0.1

        self.model.teacher_forcing_ratio = tf
        if is_main_process():
            print(f"[Epoch {epoch_idx}] Teacher forcing: {tf:.2f}")

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} (TF={tf:.2f})") \
            if is_main_process() else self.train_loader

        total_loss = gr_loss_sum = ctr_loss_sum = 0.0
        step = 0
        grid_mapper = self.train_loader.dataset.grid_mapper
        layer_offsets = self._layer_offsets()

        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            self.optimizer.zero_grad()

            # Build target_codes_seq (raw codes for teacher forcing, no offsets)
            if self.config.use_semantic_seq and 'sem_target' in batch:
                sem_target = batch['sem_target']
                B = sem_target.size(0)
                if sem_target.dim() == 2 and sem_target.size(1) % self.config.sem_id_layers == 0:
                    sem_target = sem_target.view(B, -1, self.config.sem_id_layers)
                offsets = torch.tensor(layer_offsets, device=sem_target.device).view(1, 1, -1)
                target_codes_seq = sem_target - offsets
                target_codes_seq = torch.where(
                    target_codes_seq < 0,
                    torch.tensor(-100, device=target_codes_seq.device),
                    target_codes_seq
                )
                batch['target_codes_seq'] = target_codes_seq

            loss = 0.0
            loss_gr = torch.tensor(0.0, device=self.device)
            loss_ctr_val = 0.0

            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                u, gr_logits, _ = self.model(batch)

                if self.config.use_semantic_seq:
                    sem_target = batch['sem_target']
                    B = sem_target.size(0)
                    if sem_target.dim() == 2 and sem_target.size(1) % self.config.sem_id_layers == 0:
                        sem_target = sem_target.view(B, -1, self.config.sem_id_layers)

                    lengths_items = batch['lengths'] // self.config.sem_id_layers
                    num_items_dim = sem_target.size(1)
                    pos_idx    = torch.arange(num_items_dim, device=self.device).unsqueeze(0)
                    valid_mask = pos_idx < lengths_items.unsqueeze(1)

                    # One-time diagnostic at first step
                    if epoch_idx == 1 and step == 0 and is_main_process():
                        print(f"[DIAG] sem_target shape: {sem_target.shape}, "
                              f"lengths range: [{lengths_items.min()}, {lengths_items.max()}]")
                        for li in range(self.config.sem_id_layers):
                            raw = sem_target[:, :, li] - layer_offsets[li]
                            vld = valid_mask
                            if vld.any():
                                print(f"[DIAG] Layer {li}: raw range "
                                      f"[{raw[vld].min().item()}, {raw[vld].max().item()}]")

                    loss_gr = 0.0
                    for li, logits_layer in enumerate(gr_logits):
                        targets = sem_target[:, :, li] - layer_offsets[li]
                        vocab   = logits_layer.size(-1)
                        valid   = valid_mask & (targets >= 0) & (targets < vocab)
                        targets = torch.where(valid, targets, torch.full_like(targets, -100))
                        loss_gr += self.gr_criterion(
                            logits_layer.reshape(-1, vocab), targets.reshape(-1))
                    loss_gr = loss_gr / self.config.sem_id_layers
                    loss += loss_gr

                if self.config.enable_ctr:
                    pos_codes = batch['ctr_pos_codes']
                    ctr_logits, ctr_labels = self.model.predict_ctr(
                        u, batch, pos_codes, self.device, grid_mapper)
                    loss_ctr    = self.calculate_ctr_loss(ctr_logits, ctr_labels)
                    loss        += loss_ctr
                    loss_ctr_val = loss_ctr.item()
                    ctr_loss_sum += loss_ctr_val

            # Debug: log prediction accuracy at last position every 200 steps
            if is_main_process() and self.config.use_semantic_seq and step % 200 == 0:
                with torch.no_grad():
                    b = 0
                    n_hist  = (batch['lengths'][b] // self.config.sem_id_layers).item()
                    last    = n_hist - 1
                    gt_raw  = [sem_target[b, last, li].item() - layer_offsets[li]
                               for li in range(self.config.sem_id_layers)]
                    pr_raw  = [gr_logits[li][b, last].argmax().item()
                               for li in range(self.config.sem_id_layers)]
                    match   = (pr_raw == gt_raw)
                    print(f"[Step {step}] GT={gt_raw}  Pred={pr_raw}  Match={'Y' if match else 'N'}"
                          f"  GR_loss={loss_gr.item():.4f}")
            step += 1

            if self.use_deepspeed:
                self.model_engine.backward(loss)
                self.model_engine.step()
            elif self.amp_scaler is not None:
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            if self.config.use_semantic_seq:
                gr_loss_sum += loss_gr.item()

            if is_main_process():
                logs = {'Loss': f"{loss.item():.4f}",
                        'GR': f"{loss_gr.item() if self.config.use_semantic_seq else 0:.4f}"}
                if self.config.enable_ctr:
                    logs['CTR'] = f"{loss_ctr_val:.4f}"
                pbar.set_postfix(logs)

        n = len(self.train_loader)
        return {'loss': total_loss / n, 'gr_loss': gr_loss_sum / n, 'ctr_loss': ctr_loss_sum / n}

    @torch.no_grad()
    def evaluate(self, topk=10):
        if not self.val_loader:
            return {}

        from .utils import compute_gr_metrics

        # Force TF=0 during eval so val loss reflects true inference conditions
        train_tf = getattr(self.model, 'teacher_forcing_ratio', 1.0)
        self.model.teacher_forcing_ratio = 0.0
        self.model.eval()

        grid_mapper   = self.val_loader.dataset.grid_mapper
        layer_offsets = self._layer_offsets()

        val_gr_loss_sum = val_ctr_loss_sum = 0.0
        all_ctr_logits, all_ctr_labels = [], []
        all_hit_sums = all_ndcg_sums = 0.0
        all_gr_count = 0

        iterator = tqdm(self.val_loader, desc="Eval") if is_main_process() else self.val_loader

        for batch in iterator:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            u, gr_logits, _ = self.model(batch)

            if self.config.use_semantic_seq:
                sem_target = batch['sem_target']
                B = sem_target.size(0)
                if sem_target.dim() == 2 and sem_target.size(1) % self.config.sem_id_layers == 0:
                    sem_target = sem_target.view(B, -1, self.config.sem_id_layers)

                lengths_items = batch['lengths'] // self.config.sem_id_layers
                num_items_dim = sem_target.size(1)
                pos_idx    = torch.arange(num_items_dim, device=self.device).unsqueeze(0)
                valid_mask = pos_idx < lengths_items.unsqueeze(1)

                loss_gr = 0.0
                for li, logits_layer in enumerate(gr_logits):
                    targets = sem_target[:, :, li] - layer_offsets[li]
                    vocab   = logits_layer.size(-1)
                    valid   = valid_mask & (targets >= 0) & (targets < vocab)
                    targets = torch.where(valid, targets, torch.full_like(targets, -100))
                    loss_gr += self.gr_criterion(
                        logits_layer.reshape(-1, vocab), targets.reshape(-1))
                val_gr_loss_sum += (loss_gr / self.config.sem_id_layers).item()

            if self.config.enable_ctr:
                ctr_logits, ctr_labels = self.model.predict_ctr(
                    u, batch, batch['ctr_pos_codes'], self.device, grid_mapper)
                val_ctr_loss_sum += self.calculate_ctr_loss(ctr_logits, ctr_labels).item()
                all_ctr_logits.append(ctr_logits.view(-1))
                all_ctr_labels.append(ctr_labels.view(-1))

            candidates = self.model.generate_gr_candidates(batch, k=topk, grid_mapper=grid_mapper)
            target_ids = batch.get('sem_target_eval')
            if target_ids is not None:
                bh, bn = compute_gr_metrics(candidates, target_ids, k=topk)
                all_hit_sums  += bh * target_ids.size(0)
                all_ndcg_sums += bn * target_ids.size(0)
                all_gr_count  += target_ids.size(0)

        n = len(self.val_loader)
        results = {
            'val_gr_loss': val_gr_loss_sum / n,
            'Hit@10':  all_hit_sums  / all_gr_count if all_gr_count > 0 else 0.0,
            'NDCG@10': all_ndcg_sums / all_gr_count if all_gr_count > 0 else 0.0,
        }

        if self.config.enable_ctr and all_ctr_logits:
            local_logits = torch.cat(all_ctr_logits)
            local_labels = torch.cat(all_ctr_labels)
            if self.use_distributed:
                from .utils import gather_tensors
                local_logits = gather_tensors(local_logits)
                local_labels = gather_tensors(local_labels)
            results['val_ctr_loss'] = val_ctr_loss_sum / n
            if is_main_process():
                from .utils import compute_ctr_metrics
                results['AUC'], results['LogLoss'] = compute_ctr_metrics(local_logits, local_labels)

        self.model.teacher_forcing_ratio = train_tf
        return results

    def save(self, tag):
        import os
        os.makedirs("checkpoints", exist_ok=True)
        if self.use_deepspeed:
            self.model_engine.save_checkpoint(save_dir="checkpoints", tag=tag)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
            }, f"checkpoints/{tag}.pt")
            if is_main_process():
                print(f"[Trainer] Saved checkpoints/{tag}.pt")

    def train(self):
        monitor_metric = 'LogLoss' if self.config.enable_ctr else 'Hit@10'
        mode           = 'min'     if self.config.enable_ctr else 'max'
        best_val       = float('inf') if mode == 'min' else 0.0
        patience_counter = 0

        if is_main_process():
            print(f"Training. Monitor: {monitor_metric} ({mode})")

        for epoch in range(1, self.config.epochs + 1):
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                if is_main_process():
                    train_metrics['lr'] = self.optimizer.param_groups[0]['lr']

            eval_metrics = self.evaluate(topk=10)

            if is_main_process():
                lr_str  = f"LR={train_metrics['lr']:.2e} | " if 'lr' in train_metrics else ""
                log_str = (f"Ep {epoch} | Tr_Loss: GR={train_metrics['gr_loss']:.4f} | "
                           f"{lr_str}Eval: Hit@10={eval_metrics['Hit@10']:.4f} "
                           f"NDCG@10={eval_metrics['NDCG@10']:.4f} "
                           f"GR_Loss={eval_metrics['val_gr_loss']:.4f}")
                if self.config.enable_ctr:
                    log_str += (f" AUC={eval_metrics.get('AUC',0):.4f} "
                                f"LogLoss={eval_metrics.get('LogLoss',0):.4f}")
                print(log_str)

                current_val = eval_metrics.get(monitor_metric, float('inf'))
                improved    = (current_val < best_val) if mode == 'min' else (current_val > best_val)

                if improved:
                    best_val = current_val
                    patience_counter = 0
                    print(f" >> New best {monitor_metric}={best_val:.4f}. Saving...")
                    self.save("best_model")
                else:
                    patience_counter += 1
                    print(f" >> No improvement. Patience {patience_counter}/{self.config.patience}")

            if patience_counter >= self.config.patience:
                if is_main_process():
                    print("Early stopping.")
                break
