import json
import os
import pickle
import numpy as np
import torch


class GridMapper:
    """
    Handles semantic ID mapping for GRID-encoded items.

    Loads a pre-computed mapping (item_id -> [L0, L1, L2] raw codes),
    builds offset-adjusted token ranges for each layer, and constructs a
    prefix trie used by constrained beam search at evaluation time.
    """

    def __init__(self, mapping_path, num_layers=None, codebook_size=None, auto_detect=True):
        self.mapping = self._load_mapping(mapping_path)

        if auto_detect and self.mapping:
            self.num_layers, self.codebook_sizes = self._auto_detect_structure()
            print(f"[GridMapper] {self.num_layers} layers, codebook sizes: {self.codebook_sizes}")
        else:
            if num_layers is None or codebook_size is None:
                raise ValueError("num_layers and codebook_size must be specified if auto_detect=False")
            self.num_layers = num_layers
            self.codebook_sizes = [codebook_size] * num_layers

        # Layer token ranges (token 0 reserved for padding):
        #   Layer 0: [1, 1+size0)
        #   Layer 1: [1+size0, 1+size0+size1)  ...
        self.layer_ranges = []
        start = 1
        for layer_size in self.codebook_sizes:
            end = start + layer_size
            self.layer_ranges.append((start, end))
            start = end
        self.total_vocab_size = start

        # Reverse mapping: offset code tuple -> item_id (for exact lookup)
        self.reverse_mapping = {}
        for item_id, codes in self.mapping.items():
            offset_codes = tuple(self._apply_offset(codes))
            self.reverse_mapping[offset_codes] = item_id

        # Prefix trie for constrained beam search
        self.trie = self._build_trie()
        print(f"[GridMapper] {len(self.reverse_mapping)} items, "
              f"{len(self.trie)} L0 roots, vocab size {self.total_vocab_size}")

    def _auto_detect_structure(self):
        """Infer num_layers and per-layer codebook sizes from actual data."""
        first_item = next(iter(self.mapping.values()))
        num_layers = len(first_item)
        max_values = [0] * num_layers

        for codes in self.mapping.values():
            if len(codes) != num_layers:
                raise ValueError(f"Inconsistent layer count: expected {num_layers}, got {len(codes)}")
            for layer_idx, code in enumerate(codes):
                max_values[layer_idx] = max(max_values[layer_idx], code)

        return num_layers, [v + 1 for v in max_values]

    def _build_trie(self):
        """
        Build a prefix trie over raw item codes for constrained beam search.

        Structure: {L0: {L1: {L2: {D: item_id}}}}
        All keys are raw codes (no offsets). Leaf values are integer item IDs.
        At each beam search layer, only codes that exist as trie children of
        the current prefix are considered, guaranteeing every completed beam
        maps to a real catalog item.
        """
        trie = {}
        for item_id, codes in self.mapping.items():
            node = trie
            for code in codes[:-1]:
                if code not in node:
                    node[code] = {}
                node = node[code]
            node[codes[-1]] = item_id
        return trie

    def _load_mapping(self, path):
        if not os.path.exists(path):
            print(f"[Warning] GRID mapping not found at {path}")
            return {}

        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {path}, got {type(data)}")
            mapping = {}
            for item in data:
                item_id, semantic_ids = item
                if isinstance(semantic_ids, np.ndarray):
                    semantic_ids = semantic_ids.tolist()
                elif isinstance(semantic_ids, torch.Tensor):
                    semantic_ids = semantic_ids.tolist()
                mapping[int(item_id)] = [int(c) for c in semantic_ids]
            print(f"[GridMapper] Loaded {len(mapping)} items from PKL "
                  f"({len(next(iter(mapping.values())))} layers).")
            return mapping

        if path.endswith('.pt'):
            tensor = torch.load(path)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected tensor in {path}, got {type(tensor)}")
            num_layers, num_items = tensor.shape
            print(f"[GridMapper] Loaded: {num_layers} layers × {num_items} items")
            return {item_id: tensor[:, item_id].tolist() for item_id in range(num_items)}

        with open(path, 'r') as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    def _apply_offset(self, codes):
        return [self.layer_ranges[i][0] + c for i, c in enumerate(codes)]

    def get_codes(self, item_id):
        """Return offset-adjusted codes for item_id (padding codes if unknown)."""
        codes = self.mapping.get(item_id, [0] * self.num_layers)
        return self._apply_offset(codes)

    def flatten_sequence(self, item_seq):
        """Flatten a list of item IDs into a token sequence with layer offsets."""
        flat = []
        for item_id in item_seq:
            flat.extend(self.get_codes(item_id))
        return flat

    def codes_to_item(self, code_tuple):
        """Exact lookup: offset code tuple -> item_id, or None if not found."""
        return self.reverse_mapping.get(tuple(code_tuple), None)

    def get_layer_range(self, layer_idx):
        return self.layer_ranges[layer_idx]
