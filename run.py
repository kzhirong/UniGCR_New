import sys
import os
import argparse
import deepspeed
from src.config import UniGCRConfig
from src.data import get_dataloaders
from src.model import UniGCRModel
from src.trainer import UniGCRTrainer
from src.utils import set_seed, setup_distributed, is_main_process

def parse_args():
    parser = argparse.ArgumentParser(description="Uni-GCR Training Launch")
    
    # DeepSpeed 分布式必备参数
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank passed from distributed launcher')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json',
                        help='Path to DeepSpeed config file')
    
    # 允许命令行覆盖部分关键参数
    parser.add_argument('--data_path', type=str, default='data/beauty/reviews_Beauty_5.json.gz')
    parser.add_argument('--grid_mapping', type=str, default='data/beauty/semantic_ids.json',
                        help='Path to GRID generated mapping json')
    
    # 注册 DeepSpeed 参数
    parser = deepspeed.add_config_arguments(parser)
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
    conf.sem_id_layers = 3          # 假设 GRID 是 3 层 (3x8)
    conf.sem_id_codebook_size = 256
    
    # B. 辅助特征: Atomic ID (仅作 Input Context)
    conf.use_atomic_seq = True
    conf.num_atomic_items = 50000   # [注意] 请替换为你数据集真实的 Item 数量
    conf.max_atomic_len = 50
    
    # C. 辅助特征: User Profiles
    conf.use_cat_profile = True
    conf.use_num_profile = True
    # [注意] 请根据实际数据修改维度
    conf.cat_feature_vocab_sizes = [10000, 50, 100] # 示例: UserID, Region, Device
    conf.num_feature_size = 3                       # 示例: Age, Rating, Clicks
    
    # D. 联合训练开关
    conf.enable_ctr = True          # 开启 CTR 任务
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
        print("Model & Trainer Initialized. Starting Training...")
    
    # 7. 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
