import torch
import torch.nn as nn
from tqdm import tqdm
import deepspeed
import json
import numpy as np
from .utils import is_main_process

class UniGCRTrainer:
    def __init__(self, config, args, model, train_loader, val_loader=None):
        self.config = config
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # --- Loss Definitions ---
        # GR 任务: 预测下一个 Semantic Code (Cross Entropy)
        self.gr_criterion = nn.CrossEntropyLoss()
        
        # CTR 任务: 联合 Loss
        if config.enable_ctr:
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.ce_loss = nn.CrossEntropyLoss()
            
        # --- DeepSpeed Initialization ---
        ds_config = None
        if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
            try:
                with open(args.deepspeed_config, 'r') as f:
                    ds_config = json.load(f)
                if is_main_process():
                    print(f"[Trainer] Loaded DeepSpeed config from {args.deepspeed_config}")
            except Exception as e:
                print(f"[Trainer] Error loading DS config: {e}")
                
        # 初始化 DeepSpeed Engine
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=args, 
            model=model, 
            model_parameters=model.parameters(), 
            config=ds_config, 
            dist_init_required=True
        )

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
        self.model_engine.train()
        
        # 仅主进程显示进度条
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}")
        else:
            pbar = self.train_loader
            
        # 累计 Loss 用于日志
        total_loss = 0.0
        gr_loss_sum = 0.0
        ctr_loss_sum = 0.0
        
        # 获取 GridMapper (用于 Beam Search 时的 Masking)
        # 注意: DataLoader 可能经过 DistributedSampler 封装，需要通过 dataset 访问
        grid_mapper = self.train_loader.dataset.grid_mapper
        if grid_mapper is None and self.config.use_semantic_seq:
            raise ValueError("GridMapper is required for Semantic ID training but not found.")
        
        for batch in pbar:
            # 1. 将 Batch 数据移动到 GPU
            batch = {
                k: v.to(self.model_engine.device) 
                for k, v in batch.items() 
                if isinstance(v, torch.Tensor)
            }
            
            self.model_engine.zero_grad()
            
            # 2. Forward Pass (Shared Backbone & GR Head)
            # u: User State (B, D)
            # gr_logits: (B, Vocab) - Next Token Prediction Logits
            u, gr_logits = self.model_engine(batch)
            
            loss = 0.0
            
            # --- Task A: Generative Retrieval (GR) ---
            if self.config.use_semantic_seq:
                # Target: Flattened Semantic Sequence
                # shape: (B * Layers)
                target_flat = batch['sem_target'].view(-1)
                loss_gr = self.gr_criterion(gr_logits, target_flat)
                
                loss += loss_gr
                gr_loss_sum += loss_gr.item()
            
            # --- Task B: CTR Prediction (Optional) ---
            loss_ctr_val = 0.0
            if self.config.enable_ctr:
                # 获取原始模型 (DeepSpeed wrap 了 module)
                real_model = self.model_engine.module if hasattr(self.model_engine, 'module') else self.model_engine
                
                # 正样本: 用户实际点击的 Item 的 Semantic Codes
                # shape: (B, Layers)
                pos_codes = batch['ctr_pos_codes']
                
                # 调用 predict_ctr
                # 注意: 内部包含 Beam Search 逻辑，需要传入 batch (获取 History) 和 mapper
                ctr_logits, ctr_labels = real_model.predict_ctr(
                    u, batch, pos_codes, self.model_engine.device, grid_mapper
                )
                
                # 计算 Loss
                loss_ctr = self.calculate_ctr_loss(ctr_logits, ctr_labels)
                
                loss += loss_ctr
                loss_ctr_val = loss_ctr.item()
                ctr_loss_sum += loss_ctr_val

            # 3. Backward & Step (DeepSpeed)
            self.model_engine.backward(loss)
            self.model_engine.step()
            
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
        
        self.model_engine.eval()
        device = self.model_engine.device
        grid_mapper = self.val_loader.dataset.grid_mapper
        
        # 统计变量 (用于 Loss 计算)
        val_gr_loss_sum = 0.0
        val_ctr_loss_sum = 0.0
        
        # 统计变量 (用于 Metrics 计算)
        # GR
        all_hit_sums = 0.0
        all_ndcg_sums = 0.0
        all_gr_count = 0
        
        # CTR (收集 Logits 和 Labels 计算全局 AUC)
        all_ctr_logits = []
        all_ctr_labels = []

        iterator = tqdm(self.val_loader, desc="Eval") if is_main_process() else self.val_loader
        real_model = self.model_engine.module if hasattr(self.model_engine, 'module') else self.model_engine

        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # --- A. 计算 Validation Loss ---
            # 1. Forward
            u, gr_logits = self.model_engine(batch)
            
            # 2. GR Val Loss
            if self.config.use_semantic_seq:
                loss_gr = self.gr_criterion(gr_logits, batch['sem_target'].view(-1))
                val_gr_loss_sum += loss_gr.item()
                
            # 3. CTR Val Loss & Logits Collection
            if self.config.enable_ctr:
                # 在 Eval 阶段，predict_ctr 内部依然会做 Beam Search 生成负样本
                # 这保证了 Loss 的计算方式与训练一致
                ctr_logits, ctr_labels = real_model.predict_ctr(
                    u, batch, batch['ctr_pos_codes'], device, grid_mapper
                )
                
                loss_ctr, _, _ = self.calculate_ctr_loss(ctr_logits, ctr_labels)
                val_ctr_loss_sum += loss_ctr.item()
                
                # 收集用于计算 AUC/LogLoss 指标
                all_ctr_logits.append(ctr_logits.view(-1))
                all_ctr_labels.append(ctr_labels.view(-1))

            # --- B. 计算 GR Ranking Metrics (Hit/NDCG) ---
            # 这部分需要 Beam Search 生成，比较耗时
            # 如果只为了 Early Stop (Loss based)，可以跳过这步，但为了监控指标还是加上
            candidates = real_model.generate_gr_candidates(
                batch, k=topk, grid_mapper=grid_mapper
            )
            
            # 计算当前 batch 的平均 Hit/NDCG
            # 注意: target_eval 是原始 Item Codes，不是 flattened
            batch_hit, batch_ndcg = compute_gr_metrics(
                candidates, batch['sem_target_eval'], grid_mapper, k=topk
            )
            
            all_hit_sums += batch_hit * batch['sem_target_eval'].size(0)
            all_ndcg_sums += batch_ndcg * batch['sem_target_eval'].size(0)
            all_gr_count += batch['sem_target_eval'].size(0)

        # --- 汇总结果 ---
        num_batches = len(self.val_loader)
        
        # 1. Loss 汇总 (Mean across batches)
        # 简单平均即可，不需要 gather (因为 DP 每个卡数据量差不多)
        avg_gr_loss = val_gr_loss_sum / num_batches
        avg_ctr_loss = val_ctr_loss_sum / num_batches
        
        # 2. GR Metrics 汇总 (AllReduce)
        # 转 Tensor
        gr_stats = torch.tensor([all_hit_sums, all_ndcg_sums, all_gr_count], device=device)
        torch.distributed.all_reduce(gr_stats, op=torch.distributed.ReduceOp.SUM)
        
        final_hit = gr_stats[0] / gr_stats[2] if gr_stats[2] > 0 else 0.0
        final_ndcg = gr_stats[1] / gr_stats[2] if gr_stats[2] > 0 else 0.0
        
        results = {
            'val_gr_loss': avg_gr_loss,
            'Hit@10': final_hit.item(),
            'NDCG@10': final_ndcg.item()
        }
        
        # 3. CTR Metrics 汇总 (Gather & Sklearn)
        if self.config.enable_ctr and len(all_ctr_logits) > 0:
            local_logits = torch.cat(all_ctr_logits)
            local_labels = torch.cat(all_ctr_labels)
            
            # Gather 全局数据算 AUC 才准确
            global_logits = gather_tensors(local_logits)
            global_labels = gather_tensors(local_labels)
            
            results['val_ctr_loss'] = avg_ctr_loss
            
            if is_main_process():
                auc, logloss = compute_ctr_metrics(global_logits, global_labels)
                results['AUC'] = auc
                results['LogLoss'] = logloss
        
        return results

    def save(self, tag):
        """保存 Checkpoint"""
        self.model_engine.save_checkpoint(save_dir="checkpoints", tag=tag)

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
