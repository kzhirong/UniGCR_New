import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from .grid_utils import GridMapper

class UniversalDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.data = []
        
        # Grid Mapper (必须有，因为 GR 依赖 Semantic ID)
        self.grid_mapper = None
        if config.use_semantic_seq:
            self.grid_mapper = GridMapper(
                config.grid_mapping_path,
                config.sem_id_layers,
                config.sem_id_codebook_size
            )
            self.config.sem_total_vocab = self.grid_mapper.total_vocab_size
            
        self._load_data(config.data_path)

    def _load_data(self, path):
        # 模拟数据加载
        print(f"Loading data (Sim Mode)...")
        for _ in range(1000):
            sample = {}
            # 必须包含 Target Item
            target_item = np.random.randint(1, 1000)
            sample['target_item'] = target_item
            
            # 1. Semantic History (用户点的 Item 对应的 Semantic IDs)
            if self.config.use_semantic_seq:
                sample['sem_seq'] = np.random.randint(1, 1000, 20).tolist()
            
            # 2. Atomic History (作为特征)
            if self.config.use_atomic_seq:
                sample['atom_seq'] = np.random.randint(1, self.config.num_atomic_items+1, 20).tolist()
            
            # 3. Profiles
            if self.config.use_cat_profile:
                sample['cat_feats'] = [np.random.randint(0, v) for v in self.config.cat_feature_vocab_sizes]
            if self.config.use_num_profile:
                sample['num_feats'] = np.random.rand(self.config.num_feature_size).tolist()
                
            self.data.append(sample)

    def __getitem__(self, idx):
        raw_item = self.data[idx]
        output = {}
        
        # 目标 Item 的 Semantic Codes (用于 GR Target 和 CTR Positive)
        # shape: [3] (假设3层)
        tgt_codes = self.grid_mapper.get_codes(raw_item['target_item'])
        
        # ===========================
        # 1. Semantic Sequence & Target
        # ===========================
        if self.config.use_semantic_seq:
            # History Flatten: [Item1_C1, Item1_C2, ..., ItemN_C3]
            seq_codes = self.grid_mapper.flatten_sequence(raw_item['sem_seq'])
            
            # 构造训练序列: History + Target
            # GR 任务是 Shifted Prediction
            full_seq = seq_codes + tgt_codes
            
            max_len = self.config.max_seq_len
            if len(full_seq) > max_len:
                full_seq = full_seq[-max_len:]
            else:
                full_seq = [0] * (max_len - len(full_seq)) + full_seq
            
            # Input to Model
            output['sem_history'] = torch.tensor(full_seq[:-1], dtype=torch.long)
            # Label for GR Loss (Next Token)
            output['sem_target'] = torch.tensor(full_seq[1:], dtype=torch.long)
            
            # Label for CTR Positive (完整 Item 表示)
            output['ctr_pos_codes'] = torch.tensor(tgt_codes, dtype=torch.long)

        # ===========================
        # 2. Atomic Sequence (仅特征)
        # ===========================
        if self.config.use_atomic_seq:
            seq = raw_item['atom_seq']
            max_len = self.config.max_atomic_len
            if len(seq) > max_len:
                seq = seq[-max_len:]
            else:
                seq = [0] * (max_len - len(seq)) + seq
            # 只返回 History，不返回 Target
            output['atom_history'] = torch.tensor(seq, dtype=torch.long)
            
        # ===========================
        # 3. User Profiles
        # ===========================
        if self.config.use_cat_profile:
            output['cat_feats'] = torch.tensor(raw_item['cat_feats'], dtype=torch.long)
        if self.config.use_num_profile:
            output['num_feats'] = torch.tensor(raw_item['num_feats'], dtype=torch.float)
            
        return output

    def __len__(self):
        return len(self.data)

def get_dataloaders(config):
    train_ds = UniversalDataset(config, mode='train')
    val_ds = UniversalDataset(config, mode='val')
    # Sync dynamic config
    if config.use_semantic_seq:
        val_ds.config.sem_total_vocab = train_ds.config.sem_total_vocab
        
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, sampler=val_sampler, num_workers=2)
    return train_dl, val_dl
