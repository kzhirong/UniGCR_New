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
            # CRITICAL FIX: Use len(seq_codes) (history only) to ensure clean item boundaries
            # sem_history will contain full_seq[:-1], which includes partial target tokens
            # But for item-level extraction, we need to extract from last HISTORY position
            actual_length = len(seq_codes)

            # Truncate/pad to max_seq_len
            max_len = self.config.max_seq_len
            if len(full_seq) > max_len:
                # Truncate from the left (keep most recent history + target).
                # IMPORTANT: always remove whole items (round UP to nearest num_layers)
                # so the sequence stays item-aligned.  A non-aligned cut leaves a partial
                # item's D-code at position 0, causing view(-1, n_layers) to misalign and
                # the debug / training to see D-range codes in the L0 column.
                n_l = self.config.sem_id_layers
                tokens_to_remove = len(full_seq) - max_len
                items_to_remove  = (tokens_to_remove + n_l - 1) // n_l  # ceil division
                aligned_remove   = items_to_remove * n_l
                full_seq = full_seq[aligned_remove:]
                actual_length = max(0, len(seq_codes) - aligned_remove)
                # aligned_remove may be slightly more than tokens_to_remove,
                # so full_seq may now be 1-3 tokens shorter than max_len → pad end
                if len(full_seq) < max_len:
                    full_seq = full_seq + [0] * (max_len - len(full_seq))
            else:
                # CRITICAL FIX: Use RIGHT-PADDING instead of LEFT-PADDING
                # HSTU expects valid items at the start, not the end!
                full_seq = full_seq + [0] * (max_len - len(full_seq))

            # Input: all tokens except last (38 items × 4 tokens = 152 tokens)
            # Label: shift by 1 ITEM (4 tokens), not 1 token
            #   full_seq[:-1]              = 152 tokens (positions 0..151)
            #   full_seq[num_layers:-1]    = 148 tokens (positions 4..151, drops first item)
            #   + [0] * num_layers         = pads 4 zeros at end → 152 tokens total
            #
            # Result after reshape to (38, 4):
            #   sem_history[i] = Item i
            #   sem_target[i]  = Item i+1  (next item, which HSTU causally cannot see)
            num_layers = self.config.sem_id_layers
            output['sem_history'] = torch.tensor(full_seq[:-1], dtype=torch.long)

            target_tokens = full_seq[num_layers:-1] + [0] * num_layers  # 152 tokens, with offsets
            output['sem_target'] = torch.tensor(target_tokens, dtype=torch.long)

            # target_codes_seq: raw codes (NO offsets) for teacher forcing inside
            # predict_codes_autoregressive.  Shape: (num_items, 4).
            # Layer starts are the offsets applied by GridMapper (e.g. [1, 257, 513, 769]).
            layer_starts = torch.tensor(
                [self.grid_mapper.layer_ranges[i][0] for i in range(num_layers)],
                dtype=torch.long
            )  # (4,)
            target_raw = torch.tensor(target_tokens, dtype=torch.long).view(-1, num_layers)  # (38, 4)
            target_raw = torch.clamp(target_raw - layer_starts.unsqueeze(0), min=0)          # (38, 4)
            output['target_codes_seq'] = target_raw   # model reads this for teacher forcing

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
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    return train_dl, val_dl
