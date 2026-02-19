import json
import os

class GridMapper:
    def __init__(self, mapping_path, num_layers, codebook_size):
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.mapping = self._load_mapping(mapping_path)
        
        # 定义每一层的 ID 范围 (0 是 padding)
        # Layer 0: [1, 256]
        # Layer 1: [257, 512] ...
        self.layer_ranges = []
        start = 1
        for _ in range(num_layers):
            end = start + codebook_size
            self.layer_ranges.append((start, end))
            start = end
        self.total_vocab_size = start

        self.reverse_mapping = {}
        if self.mapping:
            for item_id, codes in self.mapping.items():
                offset_codes = tuple(self._apply_offset(codes))
                self.reverse_mapping[offset_codes] = item_id

    def _load_mapping(self, path):
        if not os.path.exists(path):
            print(f"[Warning] GRID mapping not found at {path}")
            return {}
        with open(path, 'r') as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    def _apply_offset(self, codes):
        out = []
        for i, c in enumerate(codes):
            out.append(self.layer_ranges[i][0] + c)
        return out

    def get_codes(self, item_id):
        codes = self.mapping.get(item_id, [0]*self.num_layers)
        return self._apply_offset(codes)

    def flatten_sequence(self, item_seq):
        flat = []
        for item_id in item_seq:
            flat.extend(self.get_codes(item_id))
        return flat

    def codes_to_item(self, code_tuple):
        return self.reverse_mapping.get(tuple(code_tuple), None)
        
    def get_layer_range(self, layer_idx):
        return self.layer_ranges[layer_idx]
