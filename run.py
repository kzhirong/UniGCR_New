import sys
import os
import argparse
import torch

# CRITICAL: Import fbgemm_gpu FIRST to register custom ops
# This must happen before any torch.ops.fbgemm calls
try:
    import fbgemm_gpu
    # Test that ops are available
    torch.ops.fbgemm.asynchronous_complete_cumsum
    print("[✓] fbgemm_gpu loaded successfully")
except (ImportError, AttributeError) as e:
    print(f"[WARNING] fbgemm_gpu not available: {e}")
    print("Install with: pip install fbgemm-gpu==1.1.0 --index-url https://download.pytorch.org/whl/cu124")

# Import DeepSpeed if available (optional)
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("[Note] DeepSpeed not available - will use regular PyTorch training")

from src.config import UniGCRConfig
from src.data_amazon import get_dataloaders
from src.model import UniGCRModel
from src.trainer import UniGCRTrainer
from src.utils import set_seed, setup_distributed, is_main_process

def parse_args():
    parser = argparse.ArgumentParser(description="Uni-GCR Training Launch")

    # 允许命令行覆盖部分关键参数
    parser.add_argument('--data_path', type=str, default='data/train_sequences.json')
    parser.add_argument('--grid_mapping', type=str, default='data/semantic_id_kmean.pt',
                        help='Path to semantic ID mapping (RQ-VAE + Dedup)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only (no training)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to checkpoint for evaluation')

    # 注册 DeepSpeed 参数 (optional - only if DeepSpeed is available)
    if DEEPSPEED_AVAILABLE:
        parser = deepspeed.add_config_arguments(parser)
    else:
        # Add placeholder for deepspeed_config when DeepSpeed not available
        parser.add_argument('--deepspeed_config', type=str, default=None,
                          help='DeepSpeed config (ignored if DeepSpeed not installed)')

    args = parser.parse_args()
    return args

def main():
    # 1. 解析参数
    args = parse_args()
    
    # 2. 初始化分布式环境 (NCCL)
    setup_distributed()
    
    # 3. 初始化配置 (根据你的需求定制)
    conf = UniGCRConfig()
    
    # --- [关键配置区域] ---
    
    # A. 核心任务: GR 预测 Semantic ID
    conf.use_semantic_seq = True
    conf.grid_mapping_path = args.grid_mapping
    conf.sem_id_layers = 4          # 4 layers: [L0, L1, L2, Dedup]
    conf.sem_id_codebook_size = 256  # Vocab size for L0, L1, L2
    conf.sem_id_dedup_size = 19      # Vocab size for Dedup layer (0-18)
    
    # B. 辅助特征: Atomic ID (仅作 Input Context)
    conf.use_atomic_seq = False
    conf.num_atomic_items = 12101   # [注意] 请替换为你数据集真实的 Item 数量
    conf.max_atomic_len = 50
    
    # C. 辅助特征: User Profiles
    conf.use_cat_profile = False
    conf.use_num_profile = False
    # [注意] 请根据实际数据修改维度
    conf.cat_feature_vocab_sizes = [10000, 50, 100] # 示例: UserID, Region, Device
    conf.num_feature_size = 3                       # 示例: Age, Rating, Clicks
    
    # D. 联合训练开关
    conf.enable_ctr = False          # 开启 CTR 任务
    conf.ctr_use_self_attn = True   # 开启 Candidate 自注意力
    conf.ctr_use_cross_attn = True  # 开启 User-Item 交叉注意力
    
    # --------------------
    
    conf.data_path = args.data_path
    set_seed(conf.seed)
    
    # 4. 准备数据
    # DataLoader 内部会根据 conf 自动处理 Semantic/Atomic/Profile 的读取和对齐
    if is_main_process():
        print(f"Loading Data from {conf.data_path}...")
        print(f"Using GRID Mapping: {conf.grid_mapping_path}")
        
    train_dl, val_dl = get_dataloaders(conf)
    
    if is_main_process():
        print(f"Data Loaded. Semantic Vocab Size: {conf.sem_total_vocab}")
    
    # 5. 初始化模型
    model = UniGCRModel(conf)
    
    # 6. 初始化 Trainer (含 DeepSpeed)
    trainer = UniGCRTrainer(
        config=conf,
        args=args,
        model=model,
        train_loader=train_dl,
        val_loader=val_dl
    )
    
    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model & Trainer Initialized.")
        print(f"  embed_dim   : {conf.embed_dim}")
        print(f"  hstu_layers : {conf.hstu_layers}")
        print(f"  hstu_heads  : {conf.hstu_heads}")
        print(f"  max_seq_len : {conf.max_seq_len} ({(conf.max_seq_len-1)//conf.sem_id_layers} items)")
        print(f"  batch_size  : {conf.batch_size}")
        print(f"  parameters  : {num_params:,}")

    # 7. Check if eval-only mode
    if args.eval_only:
        if is_main_process():
            print(f"\n[Eval-Only Mode] Loading checkpoint from {args.checkpoint}...")

        # Load checkpoint
        if os.path.exists(args.checkpoint):
            # weights_only=False is safe here since we trust our own checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])

            if is_main_process():
                print(f"✓ Loaded checkpoint from {args.checkpoint}")
                print(f"\nRunning evaluation on validation set...")

            # Run evaluation with rank-matched beam search (fast!)
            # Rank-matched generates only k=10 combinations (not k^4=10,000)
            eval_results = trainer.evaluate(topk=10)

            # Print results
            if is_main_process():
                print("\n" + "=" * 60)
                print("EVALUATION RESULTS")
                print("=" * 60)
                print(f"GR Loss:     {eval_results['val_gr_loss']:.4f}")
                print(f"Hit@10:      {eval_results['Hit@10']:.4f}")
                print(f"NDCG@10:     {eval_results['NDCG@10']:.4f}")

                if conf.enable_ctr:
                    print(f"CTR Loss:    {eval_results.get('val_ctr_loss', 0):.4f}")
                    print(f"AUC:         {eval_results.get('AUC', 0):.4f}")
                    print(f"LogLoss:     {eval_results.get('LogLoss', 0):.4f}")

                print("=" * 60)
        else:
            if is_main_process():
                print(f"✗ Checkpoint not found at {args.checkpoint}")
                print("Please ensure the checkpoint file exists or train the model first.")
    else:
        # Normal training mode
        if is_main_process():
            print("Starting Training...")
        trainer.train()

if __name__ == "__main__":
    main()
