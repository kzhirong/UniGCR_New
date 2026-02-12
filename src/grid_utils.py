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

        # Build vectorized lookup table for fast nearest neighbor search
        # Shape: (num_items, num_layers)
        if self.reverse_mapping:
            self._build_vectorized_lookup()

        # Build prefix trie for constrained beam search.
        # Trie structure: {L0_raw: {L1_raw: {L2_raw: {D_raw: item_id}}}}
        # All keys are raw codes (NO offsets). Leaves are integer item IDs.
        # This lets beam search restrict expansion at each layer to only codes
        # that appear in at least one real item — every final beam is a valid item.
        self.trie = self._build_trie()
        print(f"[GridMapper] Built prefix trie: {len(self.trie)} L0 roots, "
              f"{len(self.reverse_mapping)} total items")

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

    def _build_trie(self):
        """
        Build a prefix trie from raw item codes for constrained beam search.

        Structure: {L0_raw: {L1_raw: {L2_raw: {D_raw: item_id, ...}, ...}, ...}, ...}

        Keys at every level are raw integer codes (no layer offsets applied).
        Leaf values are integer item IDs from self.mapping.

        During constrained beam search, at each layer we look up the current beam's
        partial code path in the trie and only allow expansion into codes that exist
        as children of that node.  This guarantees every completed beam maps to a
        real catalog item — no nearest-neighbour fallback required.
        """
        trie = {}
        for item_id, codes in self.mapping.items():
            node = trie
            for code in codes[:-1]:          # descend through all layers except the last
                if code not in node:
                    node[code] = {}
                node = node[code]
            node[codes[-1]] = item_id        # leaf: map final code to item_id
        return trie

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

    def codes_to_item_nearest(self, code_tuple):
        """
        Find item ID for code_tuple, with fallback to nearest neighbor if exact match fails.

        Args:
            code_tuple: Tuple or list of offset codes [L0+offset, L1+offset, L2+offset, Dedup+offset]

        Returns:
            item_id: Item ID (int), or None if mapping is empty
        """
        # Try exact match first
        code_tuple = tuple(code_tuple)
        exact_match = self.reverse_mapping.get(code_tuple, None)
        if exact_match is not None:
            return exact_match

        # Fallback: Find nearest neighbor using L1 distance
        if not self.reverse_mapping:
            return None

        min_dist = float('inf')
        best_item = None

        # Convert query to tensor for faster computation
        query = torch.tensor(code_tuple, dtype=torch.long)

        # Search for nearest valid code (L1 distance)
        for valid_codes, item_id in self.reverse_mapping.items():
            valid_tensor = torch.tensor(valid_codes, dtype=torch.long)
            dist = torch.abs(query - valid_tensor).sum().item()

            if dist < min_dist:
                min_dist = dist
                best_item = item_id

        return best_item

    def _build_vectorized_lookup(self):
        """
        Build vectorized tensors for fast batch nearest neighbor search.
        Precomputes all valid code combinations and their item IDs.
        """
        # Extract all valid codes and item IDs
        valid_codes_list = []
        item_ids_list = []

        for codes, item_id in self.reverse_mapping.items():
            valid_codes_list.append(list(codes))
            item_ids_list.append(item_id)

        # Convert to tensors: (num_items, num_layers)
        self.valid_codes_tensor = torch.tensor(valid_codes_list, dtype=torch.long)  # (N, 4)
        self.item_ids_tensor = torch.tensor(item_ids_list, dtype=torch.long)  # (N,)

        print(f"[GridMapper] Built vectorized lookup: {self.valid_codes_tensor.shape[0]} items")

    def codes_to_item_nearest_batch(self, codes_batch, use_weighted_hamming=True):
        """
        Vectorized batch nearest neighbor lookup using Weighted Hamming Distance.

        Weighted Hamming Distance is appropriate for semantic codes because:
        - Semantic codes are discrete cluster labels (not continuous values)
        - Earlier layers (L0, L1) are more important (coarse categories)
        - Later layers (L2, Dedup) encode finer details

        Args:
            codes_batch: Tensor of shape (batch_size, num_layers) with offset codes
            use_weighted_hamming: If True, use weighted Hamming; if False, use L1 (legacy)

        Returns:
            item_ids: Tensor of shape (batch_size,) with item IDs
        """
        if not hasattr(self, 'valid_codes_tensor'):
            # Fallback to single lookup if vectorized lookup not built
            return torch.tensor([self.codes_to_item_nearest(codes) for codes in codes_batch])

        # Move to same device as input
        device = codes_batch.device
        valid_codes = self.valid_codes_tensor.to(device)  # (N, 4)
        item_ids = self.item_ids_tensor.to(device)  # (N,)

        if use_weighted_hamming:
            # ============================================================
            # Weighted Hamming Distance (RECOMMENDED)
            # ============================================================
            # Treats codes as discrete labels, weights hierarchical importance

            # Step 1: Compare codes element-wise (exact match check)
            # codes_batch: (B, 4) -> (B, 1, 4)
            # valid_codes: (N, 4) -> (1, N, 4)
            # matches: (B, N, 4) - True where codes match
            matches = (codes_batch.unsqueeze(1) == valid_codes.unsqueeze(0))

            # Step 2: Convert to mismatches (1 if different, 0 if same)
            mismatches = (~matches).float()  # (B, N, 4)

            # Step 3: Apply hierarchical weights
            # L0 (coarse category) = 8, L1 (subcategory) = 4,
            # L2 (details) = 2, Dedup (collision ID) = 1
            weights = torch.tensor([8.0, 4.0, 2.0, 1.0], device=device)

            # Step 4: Compute weighted distance
            # (B, N, 4) * (4,) -> (B, N, 4) -> sum -> (B, N)
            distances = (mismatches * weights).sum(dim=2)

            # Perfect match = 0, all layers wrong = 15 (8+4+2+1)

        else:
            # ============================================================
            # L1 Distance (LEGACY - NOT RECOMMENDED)
            # ============================================================
            # Treats codes as continuous numbers (incorrect assumption!)
            distances = torch.abs(codes_batch.unsqueeze(1) - valid_codes.unsqueeze(0)).sum(dim=2)

        # Find nearest neighbor for each query
        nearest_indices = torch.argmin(distances, dim=1)  # (B,)

        # Return corresponding item IDs
        return item_ids[nearest_indices]

    def get_layer_range(self, layer_idx):
        return self.layer_ranges[layer_idx]
