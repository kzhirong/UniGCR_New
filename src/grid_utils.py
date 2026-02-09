import json
import os
import torch

class GridMapper:
    def __init__(self, mapping_path, num_layers=None, codebook_size=None, auto_detect=True):
        """
        GridMapper for handling semantic IDs with variable codebook sizes per layer

        Args:
            mapping_path: Path to semantic_ids.json (item_id -> [code1, code2, ...])
            num_layers: Number of semantic layers (auto-detected if not specified)
            codebook_size: Size of codebook (only used if auto_detect=False)
            auto_detect: If True, automatically detect codebook sizes from data
        """
        self.mapping = self._load_mapping(mapping_path)

        # Auto-detect structure from actual data
        if auto_detect and self.mapping:
            print(f"[GridMapper] Auto-detecting codebook sizes from {mapping_path}...")
            self.num_layers, self.codebook_sizes = self._auto_detect_structure()
            print(f"[GridMapper] Detected {self.num_layers} layers with sizes: {self.codebook_sizes}")
        else:
            # Fallback to manual specification (backward compatibility)
            if num_layers is None or codebook_size is None:
                raise ValueError("num_layers and codebook_size must be specified if auto_detect=False")
            self.num_layers = num_layers
            self.codebook_sizes = [codebook_size] * num_layers
            print(f"[GridMapper] Manual mode: {self.num_layers} layers, uniform size {codebook_size}")

        # Build layer ranges with variable sizes
        # Layer 0: [1, 1+size0]
        # Layer 1: [1+size0, 1+size0+size1] ...
        # Token 0 is reserved for padding
        self.layer_ranges = []
        start = 1
        for layer_size in self.codebook_sizes:
            end = start + layer_size
            self.layer_ranges.append((start, end))
            start = end
        self.total_vocab_size = start

        print(f"[GridMapper] Layer ranges: {self.layer_ranges}")
        print(f"[GridMapper] Total vocab size: {self.total_vocab_size}")

        # Build reverse mapping (semantic_codes -> item_id)
        self.reverse_mapping = {}
        if self.mapping:
            for item_id, codes in self.mapping.items():
                offset_codes = tuple(self._apply_offset(codes))
                self.reverse_mapping[offset_codes] = item_id

    def _auto_detect_structure(self):
        """
        Auto-detect number of layers and codebook size per layer from actual data

        Returns:
            num_layers: Number of semantic layers
            codebook_sizes: List of codebook sizes (max_value + 1) for each layer
        """
        if not self.mapping:
            raise ValueError("Cannot auto-detect: mapping is empty")

        # Get first item to determine number of layers
        first_item = next(iter(self.mapping.values()))
        num_layers = len(first_item)

        # Find max value in each layer across all items
        max_values = [0] * num_layers

        for codes in self.mapping.values():
            if len(codes) != num_layers:
                raise ValueError(f"Inconsistent layer count: expected {num_layers}, got {len(codes)}")

            for layer_idx, code in enumerate(codes):
                max_values[layer_idx] = max(max_values[layer_idx], code)

        # Codebook size = max_value + 1 (since codes start from 0)
        codebook_sizes = [max_val + 1 for max_val in max_values]

        return num_layers, codebook_sizes

    def _load_mapping(self, path):
        if not os.path.exists(path):
            print(f"[Warning] GRID mapping not found at {path}")
            return {}

        # Support both .pt (PyTorch tensor) and .json formats
        if path.endswith('.pt'):
            # Load PyTorch tensor: shape [num_layers, num_items]
            # Each column represents an item's semantic codes across layers
            tensor = torch.load(path)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected tensor in {path}, got {type(tensor)}")

            num_layers, num_items = tensor.shape
            print(f"[GridMapper] Loaded tensor mapping: {num_layers} layers × {num_items} items")

            # Convert to dict: {item_id: [code_L0, code_L1, code_L2, code_Dedup]}
            mapping = {}
            for item_id in range(num_items):
                codes = tensor[:, item_id].tolist()  # Extract column for this item
                mapping[item_id] = codes
            return mapping
        else:
            # JSON format: {item_id: [code1, code2, ...]}
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
