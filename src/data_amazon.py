import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .grid_utils import GridMapper


class AmazonBeautyDataset(Dataset):
    """
    Dataset for Amazon Beauty sequential recommendation with GRID semantic IDs.
    Expects preprocessed data from scripts/prepare_amazon_data.py.
    """

    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

        # Load GridMapper and update config with auto-detected values
        self.grid_mapper = None
        if config.use_semantic_seq:
            self.grid_mapper = GridMapper(config.grid_mapping_path, auto_detect=True)
            self.config.sem_total_vocab = self.grid_mapper.total_vocab_size
            self.config.sem_id_layers = self.grid_mapper.num_layers

        # Load item2idx and sequence data
        self.item2idx = self._load_item2idx(config.data_path)
        if self.item2idx:
            self.config.num_atomic_items = len(self.item2idx)
            print(f"[Dataset] {self.config.num_atomic_items} unique items")

        self.data = self._load_data(config.data_path, mode)
        print(f"[Dataset] {len(self.data)} {mode} samples")

    def _load_item2idx(self, data_path):
        # asin_to_idx.json is 0-indexed (matches part-00000.pkl item IDs exactly)
        item2idx_path = os.path.join(os.path.dirname(data_path), 'asin_to_idx.json')
        if not os.path.exists(item2idx_path):
            print(f"[Warning] asin_to_idx.json not found at {item2idx_path}")
            return {}
        with open(item2idx_path, 'r') as f:
            return json.load(f)

    def _load_data(self, data_path, mode):
        fname = 'train_sequences.json' if mode == 'train' else 'test_sequences.json'
        seq_path = os.path.join(os.path.dirname(data_path), fname)
        if not os.path.exists(seq_path):
            print(f"[Error] Sequence file not found: {seq_path}")
            return []
        with open(seq_path, 'r') as f:
            return json.load(f)

    def _asin_to_idx(self, asin):
        return self.item2idx.get(asin, 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw = self.data[idx]
        output = {}

        target_idx = self._asin_to_idx(raw['target'])
        history_indices = [self._asin_to_idx(a) for a in raw['history']]

        if self.config.use_semantic_seq and self.grid_mapper:
            tgt_codes = self.grid_mapper.get_codes(target_idx)
            seq_codes = self.grid_mapper.flatten_sequence(history_indices)
            full_seq = seq_codes + tgt_codes
            actual_length = len(seq_codes)  # token count of history (no padding)

            max_len = self.config.max_seq_len
            n_l = self.config.sem_id_layers

            if len(full_seq) > max_len:
                # Remove whole items from the left so the sequence stays layer-aligned.
                # Misaligned truncation would corrupt the (N, 4) reshape.
                tokens_to_remove = len(full_seq) - max_len
                items_to_remove = (tokens_to_remove + n_l - 1) // n_l  # ceil
                aligned_remove = items_to_remove * n_l
                full_seq = full_seq[aligned_remove:]
                actual_length = max(0, len(seq_codes) - aligned_remove)
                if len(full_seq) < max_len:
                    full_seq = full_seq + [0] * (max_len - len(full_seq))
            else:
                # Right-pad: HSTU expects valid tokens at the start of the sequence
                full_seq = full_seq + [0] * (max_len - len(full_seq))

            # sem_history: input tokens (first max_len-1 positions)
            # sem_target: next-item labels shifted by one item (n_l tokens)
            output['sem_history'] = torch.tensor(full_seq[:-1], dtype=torch.long)
            target_tokens = full_seq[n_l:-1] + [0] * n_l
            output['sem_target'] = torch.tensor(target_tokens, dtype=torch.long)

            # Raw (no offset) codes for teacher forcing inside predict_codes_autoregressive
            layer_starts = torch.tensor(
                [self.grid_mapper.layer_ranges[i][0] for i in range(n_l)], dtype=torch.long
            )
            target_raw = torch.tensor(target_tokens, dtype=torch.long).view(-1, n_l)
            target_raw = torch.clamp(target_raw - layer_starts.unsqueeze(0), min=0)
            output['target_codes_seq'] = target_raw  # (num_items, 3)

            output['lengths'] = torch.tensor(actual_length, dtype=torch.long)
            output['ctr_pos_codes'] = torch.tensor(tgt_codes, dtype=torch.long)
            output['sem_target_eval'] = torch.tensor(target_idx, dtype=torch.long)

        if self.config.use_atomic_seq:
            seq = history_indices
            max_len = self.config.max_atomic_len
            seq = seq[-max_len:] if len(seq) > max_len else [0] * (max_len - len(seq)) + seq
            output['atom_history'] = torch.tensor(seq, dtype=torch.long)

        if self.config.use_cat_profile:
            output['cat_feats'] = torch.zeros(len(self.config.cat_feature_vocab_sizes), dtype=torch.long)

        if self.config.use_num_profile:
            output['num_feats'] = torch.zeros(self.config.num_feature_size, dtype=torch.float)

        return output


def get_dataloaders(config):
    """Create train and test dataloaders for Amazon Beauty."""
    train_ds = AmazonBeautyDataset(config, mode='train')
    val_ds = AmazonBeautyDataset(config, mode='test')

    if config.use_semantic_seq:
        val_ds.config.sem_total_vocab = train_ds.config.sem_total_vocab

    use_distributed = dist.is_available() and dist.is_initialized()

    if use_distributed:
        train_dl = DataLoader(train_ds, batch_size=config.batch_size,
                              sampler=DistributedSampler(train_ds, shuffle=True),
                              num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=config.batch_size,
                            sampler=DistributedSampler(val_ds, shuffle=False),
                            num_workers=2, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, persistent_workers=True)
        val_dl = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    return train_dl, val_dl
