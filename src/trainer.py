import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import json
import numpy as np
from .utils import is_main_process

# Make DeepSpeed optional (not needed for single-GPU training)
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
        
        # --- Loss Definitions ---
        # GR 任务: 预测下一个 Semantic Code (Cross Entropy)
        self.gr_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # CTR 任务: 联合 Loss
        if config.enable_ctr:
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.ce_loss = nn.CrossEntropyLoss()
            
        # Check if we're in distributed mode
        self.use_distributed = dist.is_available() and dist.is_initialized()

        # Decide whether to use DeepSpeed
        # Use DeepSpeed if: (1) available, (2) explicitly requested, or (3) multi-GPU
        use_deepspeed = (
            DEEPSPEED_AVAILABLE and
            (hasattr(args, 'deepspeed_config') and args.deepspeed_config or self.use_distributed)
        )

        if use_deepspeed:
            # --- DeepSpeed Path (Multi-GPU or explicitly requested) ---
            print("[Trainer] Using DeepSpeed training")
            self.lr_scheduler = None  # DeepSpeed manages LR internally
            ds_config = None
            if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
                try:
                    with open(args.deepspeed_config, 'r') as f:
                        ds_config = json.load(f)
                    if is_main_process():
                        print(f"[Trainer] Loaded DeepSpeed config from {args.deepspeed_config}")
                except Exception as e:
                    print(f"[Trainer] Error loading DS config: {e}")

            # Initialize DeepSpeed Engine
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                args=args,
                model=model,
                model_parameters=model.parameters(),
                config=ds_config,
                dist_init_required=self.use_distributed
            )
            self.model = self.model_engine.module if hasattr(self.model_engine, 'module') else self.model_engine
            self.device = self.model_engine.device
            self.use_deepspeed = True
        else:
            # --- Regular PyTorch Path (Single-GPU, simpler) ---
            print("[Trainer] Using regular PyTorch training (single-GPU)")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = model.to(self.device)

            # Create optimizer
            lr = getattr(config, 'lr', getattr(args, 'learning_rate', 1e-4))
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            # Cosine LR decay: warm lr → 1e-5 over all epochs.
            # This prevents training from stalling at a flat plateau.
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=1e-5,
            )

            # No model_engine in regular mode
            self.model_engine = None
            self.use_deepspeed = False

            print(f"[Trainer] Device: {self.device}, Learning rate: {lr} → 1e-5 (cosine)")

        # ── Automatic Mixed Precision (AMP) ──────────────────────────────────
        # DeepSpeed handles its own AMP (fp16/bf16 via ds_config).
        # For regular PyTorch path we set up autocast + GradScaler here.
        if not self.use_deepspeed and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                # BF16: no loss scaling needed (no underflow risk), ~1.5-2× faster
                self.amp_dtype = torch.bfloat16
                self.amp_scaler = None
                print("[Trainer] Mixed precision: bfloat16 (no scaler needed)")
            else:
                # FP16: requires GradScaler to avoid underflow
                self.amp_dtype = torch.float16
                self.amp_scaler = torch.amp.GradScaler('cuda')
                print("[Trainer] Mixed precision: float16 + GradScaler")
            self.use_amp = True
        else:
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.amp_scaler = None
            if self.use_deepspeed:
                print("[Trainer] AMP: managed by DeepSpeed config")
            else:
                print("[Trainer] AMP: disabled (no CUDA)")

    def calculate_ctr_loss(self, ctr_logits, ctr_labels):
        """
        计算 CTR 任务的混合 Loss (InfoNCE + BCE)
        """
        # 1. InfoNCE (Ranking): 找出正样本所在的 Index
        target_idx = torch.argmax(ctr_labels, dim=1)
        loss_info = self.ce_loss(ctr_logits / self.config.temp, target_idx)
        
        # 2. BCE (Calibration): 逐个判断是点击(1)还是未点击(0)
        loss_bce = self.bce_loss(ctr_logits, ctr_labels)
        
        # 加权求和
        return (self.config.loss_alpha * loss_info) + (self.config.loss_beta * loss_bce)

    def train_epoch(self, epoch_idx):
        """
        完整的训练 Epoch 逻辑
        """
        self.model.train()

        # Scheduled sampling:
        #   Epochs  1-10 : TF = 1.0  (full teacher forcing — model learns correct code space)
        #   Epochs 11-40 : TF linearly 1.0 → 0.1  (reduce exposure bias while keeping a small
        #                  ground-truth signal so L1/L2 don't cascade-collapse on wrong L0)
        #   Epochs 41+   : TF = 0.1  (90% self-prediction; 10% ground-truth prevents L1 bias)
        #
        # WHY 0.1 floor instead of 0.0:
        #   At TF=0.0, L0 is almost always wrong early → L1 always sees OOD L0 → L1 collapses
        #   to its own modal codes (cascading bias).  Keeping 10% true L0 conditioning lets L1
        #   occasionally reinforce the correct P(L1|true L0, history) distribution.
        if epoch_idx <= 10:
            teacher_forcing_ratio = 1.0
        elif epoch_idx <= 40:
            teacher_forcing_ratio = max(0.1, 1.0 - (epoch_idx - 10) / 30)  # 1.0 → 0.1 over 30 epochs
        else:
            teacher_forcing_ratio = 0.1

        self.model.teacher_forcing_ratio = teacher_forcing_ratio

        if is_main_process():
            print(f"[Epoch {epoch_idx}] Teacher forcing ratio: {teacher_forcing_ratio:.2f}")

        # 仅主进程显示进度条
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} (TF={teacher_forcing_ratio:.2f})")
        else:
            pbar = self.train_loader

        # 累计 Loss 用于日志
        total_loss = 0.0
        gr_loss_sum = 0.0
        ctr_loss_sum = 0.0
        step = 0  # batch counter for debug printing

        # 获取 GridMapper (用于 Beam Search 时的 Masking)
        # 注意: DataLoader 可能经过 DistributedSampler 封装，需要通过 dataset 访问
        grid_mapper = self.train_loader.dataset.grid_mapper
        if grid_mapper is None and self.config.use_semantic_seq:
            raise ValueError("GridMapper is required for Semantic ID training but not found.")

        for batch in pbar:
            # 1. 将 Batch 数据移动到 GPU
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            self.optimizer.zero_grad()

            # 2. Prepare target codes for autoregressive teacher forcing
            if self.config.use_semantic_seq and 'sem_target' in batch:
                sem_target = batch['sem_target']
                B = sem_target.size(0)

                # Reshape if flattened
                if sem_target.dim() == 2 and sem_target.size(1) % self.config.sem_id_layers == 0:
                    num_items = sem_target.size(1) // self.config.sem_id_layers
                    sem_target = sem_target.view(B, num_items, self.config.sem_id_layers)

                # Remove offsets to get RAW codes for teacher forcing
                # sem_target: (B, num_items, 4) with offsets [1-256, 257-512, 513-768, 769-787]
                # target_codes_seq: (B, num_items, 4) WITHOUT offsets [0-255, 0-255, 0-255, 0-18]
                layer_offsets = torch.tensor([1, 257, 513, 769], device=sem_target.device)
                target_codes_seq = sem_target - layer_offsets.view(1, 1, -1)  # Broadcast and subtract

                # Handle padding: mask out padding tokens (0 becomes -1 after offset removal)
                # Replace negative values with -100 (ignore_index for cross_entropy)
                target_codes_seq = torch.where(
                    target_codes_seq < 0,
                    torch.tensor(-100, device=target_codes_seq.device),
                    target_codes_seq
                )

                # Add to batch for model access
                batch['target_codes_seq'] = target_codes_seq

            # 3. Forward + Loss
            # torch.autocast wraps the entire forward+loss section so matrix multiplications
            # and activations run in fp16/bf16 while accumulators stay in fp32.
            # When use_amp=False (CPU or DeepSpeed) the context is a no-op.
            loss = 0.0
            loss_gr = torch.tensor(0.0, device=self.device)   # sentinel (in case GR disabled)
            loss_ctr_val = 0.0

            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):

                # ── Forward Pass ────────────────────────────────────────────
                # u: User State (B, D)
                # gr_logits: list of 4 tensors [L0, L1, L2, D] with different vocab sizes
                u, gr_logits, _ = self.model(batch)

                # ── Task A: Generative Retrieval (GR) ───────────────────────
                if self.config.use_semantic_seq:
                    # gr_logits shapes:
                    #   L0/L1/L2 : (B, num_items, 256)
                    #   Dedup    : (B, num_items, 19)
                    # sem_target : (B, num_items*4) or (B, num_items, 4) — WITH offsets

                    # Reshape targets to (B, num_items, 4) if flattened
                    sem_target = batch['sem_target']
                    B = sem_target.size(0)
                    if sem_target.dim() == 2 and sem_target.size(1) % self.config.sem_id_layers == 0:
                        num_items = sem_target.size(1) // self.config.sem_id_layers
                        sem_target = sem_target.view(B, num_items, self.config.sem_id_layers)

                    # Layer offsets:  L0:1, L1:257, L2:513, Dedup:769
                    layer_offsets = [
                        1,
                        1 + self.config.sem_id_codebook_size,
                        1 + 2 * self.config.sem_id_codebook_size,
                        1 + 3 * self.config.sem_id_codebook_size,
                    ]

                    # Position mask: ignore PAD positions beyond actual history length.
                    # lengths[b] = number of history TOKENS (e.g. 8 = 2 items × 4 layers).
                    lengths_items = batch['lengths'] // self.config.sem_id_layers  # (B,)
                    num_items_dim  = sem_target.size(1)                             # 38
                    pos_idx = torch.arange(num_items_dim, device=self.device).unsqueeze(0)  # (1, 38)
                    valid_mask = pos_idx < lengths_items.unsqueeze(1)               # (B, 38) bool

                    # ── One-time diagnostic (epoch 1, step 0) ──────────────
                    if epoch_idx == 1 and step == 0 and is_main_process():
                        print(f"\n[DIAG] sem_target shape: {sem_target.shape}")
                        print(f"[DIAG] lengths_items range: [{lengths_items.min().item()}, {lengths_items.max().item()}]")
                        for _li in range(self.config.sem_id_layers):
                            _raw = sem_target[:, :, _li] - layer_offsets[_li]
                            _vld = valid_mask
                            print(f"[DIAG] Layer {_li}: offset={layer_offsets[_li]}, "
                                  f"raw valid range=[{_raw[_vld].min().item()}, {_raw[_vld].max().item()}]"
                                  if _vld.any() else f"[DIAG] Layer {_li}: no valid positions!")
                    # ── End diagnostic ─────────────────────────────────────

                    # Compute cross-entropy loss for each layer separately
                    loss_gr = 0.0
                    for layer_idx, logits_layer in enumerate(gr_logits):
                        targets_layer_offset = sem_target[:, :, layer_idx]      # (B, num_items)
                        # Remove offset → raw codes (0-255 or 0-18)
                        targets_layer = targets_layer_offset - layer_offsets[layer_idx]

                        # Three conditions make a position invalid → set to -100 (ignored):
                        #   1. Beyond actual history length  (valid_mask=False)
                        #   2. Raw code is negative          (PAD token 0 after offset removal)
                        #   3. Raw code >= vocab_size        (config/data mismatch)
                        vocab_size = logits_layer.size(-1)
                        valid_targets = valid_mask & (targets_layer >= 0) & (targets_layer < vocab_size)
                        targets_layer = torch.where(valid_targets, targets_layer,
                                                    torch.full_like(targets_layer, -100))

                        logits_flat  = logits_layer.reshape(-1, logits_layer.size(-1))
                        targets_flat = targets_layer.reshape(-1)
                        loss_gr += self.gr_criterion(logits_flat, targets_flat)

                    loss_gr = loss_gr / self.config.sem_id_layers   # average over 4 layers
                    loss += loss_gr

                # ── Task B: CTR Prediction (Optional) ───────────────────────
                if self.config.enable_ctr:
                    pos_codes = batch['ctr_pos_codes']
                    ctr_logits, ctr_labels = self.model.predict_ctr(
                        u, batch, pos_codes, self.device, grid_mapper
                    )
                    loss_ctr = self.calculate_ctr_loss(ctr_logits, ctr_labels)
                    loss += loss_ctr
                    loss_ctr_val = loss_ctr.item()
                    ctr_loss_sum += loss_ctr_val

            # autocast context closed — tensors already computed, safe to read outside
            if self.config.use_semantic_seq:
                gr_loss_sum += loss_gr.item()

            # ── DEBUG: print sample 0 every N batches ──────────────────────
            debug_every = 200
            if is_main_process() and self.config.use_semantic_seq and step % debug_every == 0:
                with torch.no_grad():
                    b = 0  # inspect sample 0 in the batch
                    n_layers  = self.config.sem_id_layers
                    lengths_b = batch['lengths'][b].item()
                    n_hist    = lengths_b // n_layers   # number of history items
                    last_pos  = n_hist - 1              # position where eval target is predicted

                    # Layer offsets from config (hardcoded)
                    layer_offsets_list = [
                        1,
                        1 + self.config.sem_id_codebook_size,
                        1 + 2 * self.config.sem_id_codebook_size,
                        1 + 3 * self.config.sem_id_codebook_size,
                    ]
                    layer_names = ['L0', 'L1', 'L2', 'D']

                    # Check if GridMapper offsets match the hardcoded config offsets.
                    # Mismatch = Colab data has different layer order from what model expects.
                    gm_ranges = [grid_mapper.layer_ranges[i][0] for i in range(n_layers)]
                    offset_mismatch = (gm_ranges != layer_offsets_list)

                    # --- History items (sem_history reshaped) ---
                    hist_items = batch['sem_history'][b].view(-1, n_layers)  # (38, 4)

                    # --- Ground-truth target at eval position (with offsets) ---
                    tgt_items = batch['sem_target'][b].view(-1, n_layers)    # (38, 4)
                    gt_codes_offset = tgt_items[last_pos].tolist()
                    gt_raw = [gt_codes_offset[i] - layer_offsets_list[i] for i in range(n_layers)]

                    # --- Model prediction at eval position (argmax of logits) ---
                    pred_raw = [
                        gr_logits[li][b, last_pos].argmax().item()
                        for li in range(n_layers)
                    ]
                    pred_offset = [pred_raw[i] + layer_offsets_list[i] for i in range(n_layers)]
                    match = (pred_offset == gt_codes_offset)

                    # --- Check layer ordering of first input item (sanity check) ---
                    first_item = hist_items[0].tolist()
                    expected_ranges = [(layer_offsets_list[i], layer_offsets_list[i] + (
                        self.config.sem_id_codebook_size if i < n_layers-1 else self.config.sem_id_dedup_size
                    )) for i in range(n_layers)]
                    layer_ok = [expected_ranges[i][0] <= first_item[i] < expected_ranges[i][1]
                                for i in range(n_layers)]

                    print(f"\n{'─'*60}")
                    print(f"[DEBUG] Epoch {epoch_idx}  Step {step}  (sample 0, TF={teacher_forcing_ratio:.2f})")
                    if offset_mismatch:
                        print(f"  ⚠ OFFSET MISMATCH: GridMapper={gm_ranges} vs config={layer_offsets_list}")
                        print(f"    → Data layer order may differ from model expectation!")
                    if not all(layer_ok):
                        print(f"  ⚠ LAYER ORDER CHECK FAILED on first item {first_item}")
                        print(f"    Expected: {[f'{layer_names[i]}∈[{expected_ranges[i][0]},{expected_ranges[i][1]})' for i in range(n_layers)]}")
                    print(f"  History items : {n_hist}  (last predict pos = {last_pos})")
                    print(f"  Last input    : {hist_items[last_pos].tolist()}  "
                          f"({', '.join(f'{layer_names[i]}={hist_items[last_pos][i].item()}' for i in range(n_layers))})")
                    print(f"  GT target     : {gt_codes_offset}  (raw: {gt_raw})")
                    print(f"  Predicted     : {pred_offset}  (raw: {pred_raw})")
                    print(f"  Per-layer     : " + "  ".join(
                        f"{layer_names[i]}: pred={pred_raw[i]} gt={gt_raw[i]} {'✓' if pred_raw[i]==gt_raw[i] else '✗'}"
                        for i in range(n_layers)))
                    print(f"  Match         : {'✓ YES' if match else '✗ NO'}")
                    print(f"  GR loss       : {loss_gr.item():.4f}")
                    print(f"{'─'*60}\n")
            step += 1
            # ── END DEBUG ───────────────────────────────────────────────────

            # 4. Backward & Optimizer Step
            if self.use_deepspeed:
                self.model_engine.backward(loss)
                self.model_engine.step()
            elif self.amp_scaler is not None:
                # FP16 path: scale loss to prevent gradient underflow
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                # BF16 or no-AMP path: standard backward
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            if is_main_process():
                logs = {
                    'Loss': f"{loss.item():.4f}", 
                    'GR': f"{loss_gr.item() if self.config.use_semantic_seq else 0:.4f}"
                }
                if self.config.enable_ctr:
                    logs['CTR'] = f"{loss_ctr_val:.4f}"
                pbar.set_postfix(logs)
            
        # 计算平均 Loss
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'gr_loss': gr_loss_sum / num_batches,
            'ctr_loss': ctr_loss_sum / num_batches
        }

    @torch.no_grad()
    def evaluate(self, topk=10):
        """
        完整的评估逻辑：Beam Search 生成 -> Item 还原 -> 指标计算
        """
        """
        评估逻辑：
        1. 计算 Validation Loss (GR & CTR) -> 用于 Early Stop
        2. 计算 Metrics (Hit/NDCG/AUC/LogLoss) -> 用于展示效果
        """
        if not self.val_loader: return {}

        # Import metrics functions
        from .utils import compute_gr_metrics

        # Force TF=0.0 during evaluation so val_gr_loss and Hit@10 both reflect
        # true inference conditions (no ground-truth peeking).
        # Restore the training TF ratio afterward.
        train_tf = getattr(self.model, 'teacher_forcing_ratio', 1.0)
        self.model.teacher_forcing_ratio = 0.0

        self.model.eval()
        grid_mapper = self.val_loader.dataset.grid_mapper

        # 统计变量 (用于 Loss 计算)
        val_gr_loss_sum = 0.0
        val_ctr_loss_sum = 0.0

        # CTR (收集 Logits 和 Labels 计算全局 AUC)
        all_ctr_logits = []
        all_ctr_labels = []

        # GR Ranking Metrics (Hit/NDCG)
        all_hit_sums = 0.0
        all_ndcg_sums = 0.0
        all_gr_count = 0

        iterator = tqdm(self.val_loader, desc="Eval") if is_main_process() else self.val_loader

        for batch in iterator:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # --- A. 计算 Validation Loss ---
            # 1. Forward
            u, gr_logits, _ = self.model(batch)

            # 2. GR Val Loss
            if self.config.use_semantic_seq:
                # gr_logits: List of 4 tensors with different vocab sizes
                # sem_target: (B, num_items, 4) or (B, num_items*4) flattened - WITH offsets

                # Reshape targets to (B, num_items, 4) if flattened
                sem_target = batch['sem_target']
                B = sem_target.size(0)

                if sem_target.dim() == 2 and sem_target.size(1) % self.config.sem_id_layers == 0:
                    num_items = sem_target.size(1) // self.config.sem_id_layers
                    sem_target = sem_target.view(B, num_items, self.config.sem_id_layers)

                # Layer offsets (same as in train_epoch)
                layer_offsets = [
                    1,
                    1 + self.config.sem_id_codebook_size,
                    1 + 2 * self.config.sem_id_codebook_size,
                    1 + 3 * self.config.sem_id_codebook_size,
                ]

                # Build position mask (same logic as training)
                lengths_items = batch['lengths'] // self.config.sem_id_layers  # (B,)
                num_items_dim  = sem_target.size(1)
                pos_idx = torch.arange(num_items_dim, device=self.device).unsqueeze(0)
                valid_mask = pos_idx < lengths_items.unsqueeze(1)               # (B, num_items)

                # Compute loss for each layer separately
                loss_gr = 0.0
                for layer_idx, logits_layer in enumerate(gr_logits):
                    targets_layer_offset = sem_target[:, :, layer_idx]

                    # Remove offset to get raw codes
                    targets_layer = targets_layer_offset - layer_offsets[layer_idx]

                    # Mask PAD/invalid positions with -100 (ignored by criterion).
                    # Same three-way guard as training: position, sign, and vocab bounds.
                    vocab_size = logits_layer.size(-1)
                    valid_targets = valid_mask & (targets_layer >= 0) & (targets_layer < vocab_size)
                    targets_layer = torch.where(valid_targets, targets_layer,
                                                torch.full_like(targets_layer, -100))

                    logits_flat  = logits_layer.reshape(-1, logits_layer.size(-1))
                    targets_flat = targets_layer.reshape(-1)
                    loss_gr += self.gr_criterion(logits_flat, targets_flat)

                # Average over 4 layers
                loss_gr = loss_gr / self.config.sem_id_layers
                val_gr_loss_sum += loss_gr.item()

            # 3. CTR Val Loss & Logits Collection
            if self.config.enable_ctr:
                # 在 Eval 阶段，predict_ctr 内部依然会做 Beam Search 生成负样本
                # 这保证了 Loss 的计算方式与训练一致
                ctr_logits, ctr_labels = self.model.predict_ctr(
                    u, batch, batch['ctr_pos_codes'], self.device, grid_mapper
                )

                loss_ctr, _, _ = self.calculate_ctr_loss(ctr_logits, ctr_labels)
                val_ctr_loss_sum += loss_ctr.item()

                # 收集用于计算 AUC/LogLoss 指标
                all_ctr_logits.append(ctr_logits.view(-1))
                all_ctr_labels.append(ctr_labels.view(-1))

            # --- B. 计算 GR Ranking Metrics (Hit/NDCG) ---
            # Use rank-matched beam search for parallel prediction
            # Generate top-k candidates using the new beam search implementation
            candidates = self.model.generate_gr_candidates(
                batch, k=topk, grid_mapper=grid_mapper
            )

            # DEBUG: Check first batch only (disabled for cleaner output)
            # if is_main_process() and all_gr_count == 0:
            #     print(f"\n[DEBUG] Candidates: {torch.unique(candidates).size(0)} unique items")
            #     print(f"  First 3 users: {[candidates[i, :5].tolist() for i in range(3)]}")

            # Compute Hit@k and NDCG@k
            # sem_target_eval: (B,) - ground truth item indices
            # candidates: (B, k) - predicted item IDs
            target_item_ids = batch.get('sem_target_eval', None)

            if target_item_ids is not None:
                batch_hit, batch_ndcg = compute_gr_metrics(
                    candidates, target_item_ids, k=topk
                )

                all_hit_sums += batch_hit * target_item_ids.size(0)
                all_ndcg_sums += batch_ndcg * target_item_ids.size(0)
                all_gr_count += target_item_ids.size(0)

        # --- 汇总结果 ---
        num_batches = len(self.val_loader)

        # 1. Loss 汇总 (Mean across batches)
        # 简单平均即可，不需要 gather (因为 DP 每个卡数据量差不多)
        avg_gr_loss = val_gr_loss_sum / num_batches
        avg_ctr_loss = val_ctr_loss_sum / num_batches

        # 2. GR Metrics 汇总
        # Compute average Hit@k and NDCG@k across all validation samples
        if all_gr_count > 0:
            avg_hit = all_hit_sums / all_gr_count
            avg_ndcg = all_ndcg_sums / all_gr_count
        else:
            avg_hit = 0.0
            avg_ndcg = 0.0

        results = {
            'val_gr_loss': avg_gr_loss,
            'Hit@10': avg_hit,
            'NDCG@10': avg_ndcg,
        }

        # 3. CTR Metrics 汇总 (Gather & Sklearn)
        if self.config.enable_ctr and len(all_ctr_logits) > 0:
            local_logits = torch.cat(all_ctr_logits)
            local_labels = torch.cat(all_ctr_labels)

            if self.use_distributed:
                # Gather 全局数据算 AUC 才准确 (only for multi-GPU)
                from .utils import gather_tensors
                global_logits = gather_tensors(local_logits)
                global_labels = gather_tensors(local_labels)
            else:
                # Single-GPU: no need to gather
                global_logits = local_logits
                global_labels = local_labels

            results['val_ctr_loss'] = avg_ctr_loss

            if is_main_process():
                from .utils import compute_ctr_metrics
                auc, logloss = compute_ctr_metrics(global_logits, global_labels)
                results['AUC'] = auc
                results['LogLoss'] = logloss

        # Restore training TF ratio before returning
        self.model.teacher_forcing_ratio = train_tf

        return results

    def save(self, tag):
        """保存 Checkpoint"""
        import os
        os.makedirs("checkpoints", exist_ok=True)

        if self.use_deepspeed:
            # DeepSpeed has built-in checkpoint saving
            self.model_engine.save_checkpoint(save_dir="checkpoints", tag=tag)
        else:
            # Regular PyTorch checkpoint saving
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, f"checkpoints/{tag}.pt")
            if is_main_process():
                print(f"[Trainer] Saved checkpoint to checkpoints/{tag}.pt")

    def train(self):
        # Early Stopping 策略设置
        # 如果开启 CTR: 监控 CTR LogLoss (min)
        # 如果仅 GR: 监控 GR Loss (min)
        if self.config.enable_ctr:
            monitor_metric = 'LogLoss' # 实际对应 val_ctr_loss 或 metrics 里的 LogLoss
            mode = 'min'
            best_val = float('inf')
        else:
            monitor_metric = 'val_gr_loss'
            mode = 'min'
            best_val = float('inf')
            
        patience_counter = 0
        
        if is_main_process():
            print(f"Start Training. Monitor: {monitor_metric} (Best: {mode})")

        for epoch in range(1, self.config.epochs + 1):
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # 1. Train
            train_metrics = self.train_epoch(epoch)

            # Step LR scheduler after each epoch (cosine decay)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                if is_main_process():
                    current_lr = self.optimizer.param_groups[0]['lr']
                    train_metrics['lr'] = current_lr

            # 2. Eval
            eval_metrics = self.evaluate(topk=10)
            
            # 3. Logging & Early Stop Logic
            if is_main_process():
                # 打印日志
                log_str = f"Ep {epoch} | "
                log_str += f"Tr_Loss: GR={train_metrics['gr_loss']:.4f} "
                if self.config.enable_ctr:
                    log_str += f"CTR={train_metrics['ctr_loss']:.4f} | "
                else:
                    log_str += "| "
                
                if 'lr' in train_metrics:
                    log_str += f"LR={train_metrics['lr']:.2e} | "
                log_str += f"Eval: "
                log_str += f"Hit@10={eval_metrics['Hit@10']:.4f} NDCG@10={eval_metrics['NDCG@10']:.4f} GR_Loss={eval_metrics['val_gr_loss']:.4f} "
                
                if self.config.enable_ctr:
                    log_str += f"AUC={eval_metrics.get('AUC',0):.4f} LogLoss={eval_metrics.get('LogLoss',0):.4f}"
                
                print(log_str)
                
                # 获取当前监控指标
                current_val = eval_metrics.get(monitor_metric, float('inf'))
                
                # 判断更优
                improved = False
                if mode == 'min':
                    if current_val < best_val: improved = True
                else:
                    if current_val > best_val: improved = True
                
                if improved:
                    best_val = current_val
                    patience_counter = 0
                    print(f" >> New Best {monitor_metric}! Saving...")
                    self.save("best_model")
                else:
                    patience_counter += 1
                    print(f" >> No improve. Patience {patience_counter}/{self.config.patience}")
                
            # 同步 Early Stop 状态 (可选，这里依赖主进程 break 也可以，或者广播)
            # 简单起见，如果达到耐心值，主进程抛出异常或结束，这里我们不做多进程同步退出
            if patience_counter >= self.config.patience:
                if is_main_process(): print("Early Stopping.")
                break
