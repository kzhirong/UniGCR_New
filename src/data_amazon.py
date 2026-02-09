"""
Modified data loader for Amazon Beauty dataset
This replaces the simulated data in data.py with real Amazon sequential data
"""
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import json
from .grid_utils import GridMapper

class AmazonBeautyDataset(Dataset):
    """
    Dataset for Amazon Beauty with Semantic IDs
    Expects preprocessed data from prepare_amazon_data.py
    """
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

        # 1. Load Grid Mapper (Semantic ID mapping)
        self.grid_mapper = None
        if config.use_semantic_seq:
            self.grid_mapper = GridMapper(
                config.grid_mapping_path,
                auto_detect=True  # Auto-detect num_layers and codebook sizes
            )
            # Update config with auto-detected values
            self.config.sem_total_vocab = self.grid_mapper.total_vocab_size
            self.config.sem_id_layers = self.grid_mapper.num_layers
            print(f"[Dataset] Auto-detected {self.grid_mapper.num_layers} layers")
            print(f"[Dataset] Codebook sizes: {self.grid_mapper.codebook_sizes}")
            print(f"[Dataset] Total semantic vocab size: {self.config.sem_total_vocab}")

        # 2. Load item2idx mapping (maps raw ASIN to integer IDs)
        self.item2idx = self._load_item2idx(config.data_path)

        # Update num_atomic_items based on actual data
        if self.item2idx:
            self.config.num_atomic_items = len(self.item2idx)
            print(f"[Dataset] Found {self.config.num_atomic_items} unique items")

        # 3. Load sequential data
        self.data = self._load_data(config.data_path, mode)
        print(f"[Dataset] Loaded {len(self.data)} {mode} samples")

    def _load_item2idx(self, data_path):
        """Load item to index mapping"""
        import os
        base_dir = os.path.dirname(data_path)
        item2idx_path = os.path.join(base_dir, 'item2idx.json')

        if not os.path.exists(item2idx_path):
            print(f"[Warning] item2idx.json not found at {item2idx_path}")
            print(f"Please run: python scripts/prepare_amazon_data.py <review_file>")
            return {}

        with open(item2idx_path, 'r') as f:
            return json.load(f)

    def _load_data(self, data_path, mode):
        """
        Load preprocessed sequential data
        Expected format: [{'user_id': ..., 'history': [...], 'target': ...}, ...]
        """
        import os
        base_dir = os.path.dirname(data_path)

        if mode == 'train':
            seq_path = os.path.join(base_dir, 'train_sequences.json')
        else:
            seq_path = os.path.join(base_dir, 'test_sequences.json')

        if not os.path.exists(seq_path):
            print(f"[Error] Sequence file not found: {seq_path}")
            print(f"Please run: python scripts/prepare_amazon_data.py {data_path}")
            # Return empty list to avoid crash - will error later if used
            return []

        with open(seq_path, 'r') as f:
            data = json.load(f)

        return data

    def _asin_to_idx(self, asin):
        """Convert ASIN (Amazon item ID) to integer index"""
        return self.item2idx.get(asin, 0)  # 0 for unknown items

    def __getitem__(self, idx):
        """
        Returns a training sample with semantic IDs
        """
        raw_sample = self.data[idx]
        output = {}

        # Convert ASINs to integer indices
        target_asin = raw_sample['target']
        target_idx = self._asin_to_idx(target_asin)

        history_asins = raw_sample['history']
        history_indices = [self._asin_to_idx(asin) for asin in history_asins]

        # ===========================
        # 1. Semantic Sequence (for GR task)
        # ===========================
        if self.config.use_semantic_seq and self.grid_mapper:
            # Get target item's semantic codes
            tgt_codes = self.grid_mapper.get_codes(target_idx)

            # Flatten history into semantic token sequence
            # [Item1_C1, Item1_C2, Item1_C3, Item2_C1, ...]
            seq_codes = self.grid_mapper.flatten_sequence(history_indices)

            # Construct full sequence: History + Target
            full_seq = seq_codes + tgt_codes

            # Track actual length before padding (for variable-length attention)
            actual_length = len(full_seq) - 1  # Subtract 1 because we'll take [:-1] for input

            # Truncate/pad to max_seq_len
            max_len = self.config.max_seq_len
            if len(full_seq) > max_len:
                full_seq = full_seq[-max_len:]
                actual_length = max_len - 1  # All tokens are valid after truncation
            else:
                full_seq = [0] * (max_len - len(full_seq)) + full_seq

            # Input: all tokens except last
            # Label: all tokens except first (shifted by 1)
            output['sem_history'] = torch.tensor(full_seq[:-1], dtype=torch.long)
            output['sem_target'] = torch.tensor(full_seq[1:], dtype=torch.long)

            # Lengths: actual (non-padded) sequence length in tokens
            # Model uses this for variable-length masking in HSTU
            output['lengths'] = torch.tensor(actual_length, dtype=torch.long)

            # For CTR task (if enabled): target item codes
            output['ctr_pos_codes'] = torch.tensor(tgt_codes, dtype=torch.long)

            # For evaluation metrics: target item index (not semantic codes)
            # This is used by trainer to compute Hit@K and NDCG@K
            output['sem_target_eval'] = torch.tensor(target_idx, dtype=torch.long)

        # ===========================
        # 2. Atomic Sequence (optional auxiliary feature)
        # ===========================
        if self.config.use_atomic_seq:
            seq = history_indices
            max_len = self.config.max_atomic_len

            if len(seq) > max_len:
                seq = seq[-max_len:]
            else:
                seq = [0] * (max_len - len(seq)) + seq

            output['atom_history'] = torch.tensor(seq, dtype=torch.long)

        # ===========================
        # 3. User Profiles (optional - set to dummy if not available)
        # ===========================
        if self.config.use_cat_profile:
            # Dummy categorical features - replace with real user features if available
            output['cat_feats'] = torch.zeros(len(self.config.cat_feature_vocab_sizes), dtype=torch.long)

        if self.config.use_num_profile:
            # Dummy numerical features
            output['num_feats'] = torch.zeros(self.config.num_feature_size, dtype=torch.float)

        return output

    def __len__(self):
        return len(self.data)


def get_dataloaders(config):
    """
    Create train and validation dataloaders for Amazon Beauty dataset
    """
    train_ds = AmazonBeautyDataset(config, mode='train')
    val_ds = AmazonBeautyDataset(config, mode='test')

    # Sync dynamic config from train to val
    if config.use_semantic_seq:
        val_ds.config.sem_total_vocab = train_ds.config.sem_total_vocab

    # Use distributed samplers only if distributed training is initialized
    use_distributed = dist.is_available() and dist.is_initialized()

    if use_distributed:
        print("[DataLoader] Using DistributedSampler for distributed training")
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_dl = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True
        )
    else:
        print("[DataLoader] Using regular DataLoader (non-distributed)")
        train_dl = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    return train_dl, val_dl
