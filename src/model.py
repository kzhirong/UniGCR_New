import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import UniGCRConfig
from .hstu_builder import build_research_hstu

class UnifiedInputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        
        # 1. Semantic Embedding (RQ-VAE + Deduplication: 4 layers)
        if config.use_semantic_seq:
            # Create separate embedding tables for each semantic layer
            # Layers: [L0, L1, L2, Dedup]
            # - L0, L1, L2: RQ-VAE layers (vocab size 256)
            # - Dedup: Handles collisions (vocab size 19: values 0-18)
            self.sem_emb_layers = nn.ModuleList([
                nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),  # L0: 256 vocab
                nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),  # L1: 256 vocab
                nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),  # L2: 256 vocab
                nn.Embedding(config.sem_id_dedup_size, dim, padding_idx=0),     # Dedup: 19 vocab
            ])
            self.num_semantic_layers = config.sem_id_layers  # 4
            
        # 2. Atomic Embedding
        if config.use_atomic_seq:
            self.atom_emb = nn.Embedding(config.num_atomic_items + 1, dim, padding_idx=0)
            
        # 3. Profile Embeddings
        if config.use_cat_profile:
            self.cat_embs = nn.ModuleList([nn.Embedding(v, dim) for v in config.cat_feature_vocab_sizes])
        if config.use_num_profile:
            self.num_projs = nn.ModuleList([nn.Linear(1, dim) for _ in range(config.num_feature_size)])
            
        if config.use_cat_profile or config.use_num_profile:
            self.feat_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, input_dict, is_generation=False):
        """
        is_generation: 如果为 True，则只处理 Semantic History 部分 (用于 Beam Search 扩展)
        """
        tokens = []
        
        # A. Profile (Prefix)
        prefix = []
        if self.config.use_cat_profile and 'cat_feats' in input_dict:
            for i, emb in enumerate(self.cat_embs):
                prefix.append(emb(input_dict['cat_feats'][:, i]).unsqueeze(1))
        if self.config.use_num_profile and 'num_feats' in input_dict:
            for i, proj in enumerate(self.num_projs):
                prefix.append(proj(input_dict['num_feats'][:, i].unsqueeze(-1)).unsqueeze(1))
        if prefix:
            tokens.append(self.feat_mlp(torch.cat(prefix, dim=1)))
            
        # B. Atomic Sequence
        if self.config.use_atomic_seq and 'atom_history' in input_dict:
            tokens.append(self.atom_emb(input_dict['atom_history']))
            
        # C. Semantic Sequence (RQ-VAE + Deduplication: 4 tokens per item)
        # sem_history is flattened with offsets applied by GridMapper:
        # [tok0_L0, tok0_L1, tok0_L2, tok0_Dedup, tok1_L0, tok1_L1, tok1_L2, tok1_Dedup, ...]
        # We need to:
        # 1. Reshape to (B, num_items, num_layers)
        # 2. Remove layer offsets to get raw codes (0-255 for L0/L1/L2, 0-18 for Dedup)
        # 3. Embed each layer separately
        if self.config.use_semantic_seq and 'sem_history' in input_dict:
            sem_history = input_dict['sem_history']  # (B, N) - flattened tokens with offsets
            batch_size, total_tokens = sem_history.shape

            # Reshape: (B, N) → (B, num_items, num_layers)
            # e.g., (4, 16) → (4, 4, 4) for 4 items with 4 tokens each [L0, L1, L2, Dedup]
            num_items = total_tokens // self.num_semantic_layers
            sem_tokens = sem_history.view(batch_size, num_items, self.num_semantic_layers)

            # Layer offset ranges (computed by GridMapper):
            # L0: [1, 257)    → raw codes 0-255 (subtract 1)
            # L1: [257, 513)  → raw codes 0-255 (subtract 257)
            # L2: [513, 769)  → raw codes 0-255 (subtract 513)
            # Dedup: [769, 788) → raw codes 0-18 (subtract 769)
            layer_offsets = [
                1,    # L0 offset
                1 + self.config.sem_id_codebook_size,  # L1 offset (1 + 256 = 257)
                1 + 2 * self.config.sem_id_codebook_size,  # L2 offset (1 + 512 = 513)
                1 + 3 * self.config.sem_id_codebook_size,  # Dedup offset (1 + 768 = 769)
            ]

            # Embed each layer separately and sum them
            layer_embeddings = []
            for layer_idx in range(self.num_semantic_layers):
                layer_tokens_offset = sem_tokens[:, :, layer_idx]  # (B, num_items) - with offset

                # Remove offset to get raw codes for embedding lookup
                layer_tokens = layer_tokens_offset - layer_offsets[layer_idx]  # (B, num_items) - raw codes

                # Get vocab size for this layer from the embedding layer
                vocab_size = self.sem_emb_layers[layer_idx].num_embeddings

                # Clamp to valid range [0, vocab_size-1]
                # - Padding token (0) becomes negative after offset removal → clamp to 0
                # - Any out-of-range tokens → clamp to vocab_size-1
                layer_tokens = torch.clamp(layer_tokens, min=0, max=vocab_size - 1)

                layer_emb = self.sem_emb_layers[layer_idx](layer_tokens)  # (B, num_items, D)
                layer_embeddings.append(layer_emb)

            # Combine layer embeddings (sum across layers)
            item_embeddings = torch.stack(layer_embeddings, dim=0).sum(dim=0)  # (B, num_items, D)
            tokens.append(item_embeddings)
            
        if not tokens: raise ValueError("Empty inputs")
        return torch.cat(tokens, dim=1)

class UniGCRModel(nn.Module):
    def __init__(self, config: UniGCRConfig):
        super().__init__()
        self.config = config

        # 1. Input Layer (handles 4-token-per-item semantic IDs: L0, L1, L2, Dedup)
        self.input_layer = UnifiedInputLayer(config)

        # 2. Research HSTU Backbone (receives pre-computed item embeddings)
        self.backbone = build_research_hstu(config)

        # 3. GR Heads - predict all 4 semantic tokens for complete item prediction
        # Four heads: L0, L1, L2 (RQ-VAE layers), Dedup (collision resolution)
        self.gr_head_L0 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)      # vocab=256
        self.gr_head_L1 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)      # vocab=256
        self.gr_head_L2 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)      # vocab=256
        self.gr_head_Dedup = nn.Linear(config.embed_dim, config.sem_id_dedup_size)      # vocab=19

        # 4. CTR Components (will be used in Phase 2)
        if config.enable_ctr:
            self.cand_proj = nn.Sequential(
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, config.embed_dim)
            )

            scorer_dim = config.embed_dim * 2
            if config.ctr_use_self_attn:
                self.cand_self_attn = nn.MultiheadAttention(config.embed_dim, 2, batch_first=True)
                scorer_dim += config.embed_dim
            if config.ctr_use_cross_attn:
                self.user_cross_attn = nn.MultiheadAttention(config.embed_dim, 2, batch_first=True)
                scorer_dim += config.embed_dim

            self.scorer = nn.Sequential(nn.Linear(scorer_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch_dict):
        """
        Forward pass for GR-only training (Phase 1).

        Args:
            batch_dict: {
                'sem_history': (B, N_tokens) - flattened semantic tokens (num_items * 4)
                'lengths': (B,) - actual sequence length in TOKENS (optional)
                ... other fields (for atomic, profile - not used in Phase 1)
            }

        Returns:
            For GR training:
                u: (B, D) - user representation from last item position
                logits_seq: List of 4 tensors (different vocab sizes):
                    - logits_L0: (B, num_items, 256) - RQ-VAE layer 0
                    - logits_L1: (B, num_items, 256) - RQ-VAE layer 1
                    - logits_L2: (B, num_items, 256) - RQ-VAE layer 2
                    - logits_Dedup: (B, num_items, 19) - Deduplication layer
                candidate_embeddings: None - not used in Research HSTU
        """
        # 1. Get item-level embeddings from UnifiedInputLayer
        # InputLayer converts: (B, N_tokens) → (B, num_items, D)
        # by reshaping to (B, num_items, 4) and embedding each layer separately
        embeddings = self.input_layer(batch_dict)  # (B, num_items, D)
        batch_size, num_items, embed_dim = embeddings.shape

        # 2. Extract sequence lengths (convert from token-level to item-level)
        lengths_tokens = batch_dict.get('lengths')
        if lengths_tokens is not None:
            # Convert token-level lengths to item-level lengths
            # e.g., 20 tokens = 5 items (20 // 4)
            lengths = lengths_tokens // self.config.sem_id_layers
        else:
            # Fallback: assume no padding
            lengths = torch.full((batch_size,), num_items, dtype=torch.long, device=embeddings.device)

        # 3. Input validation
        assert (lengths > 0).all(), "All sequence lengths must be positive"
        assert (lengths <= num_items).all(), f"Lengths {lengths.max()} exceed num_items {num_items}"

        # 4. Prepare inputs for Research HSTU
        # Create dummy item IDs (HSTU API requirement, but uses past_embeddings instead)
        past_ids = torch.arange(num_items, device=embeddings.device).unsqueeze(0).expand(batch_size, -1)
        past_embeddings = embeddings  # Pre-computed item embeddings from InputLayer
        past_payloads = {}  # Can add timestamps here if needed

        # 5. Forward through Research HSTU
        # Returns: (B, max_output_len, D) - HSTU may pad output to max_output_len
        full_embeddings_all = self.backbone(
            past_lengths=lengths,
            past_ids=past_ids,           # Dummy IDs (API requirement)
            past_embeddings=past_embeddings,  # Actual item embeddings (pre-computed)
            past_payloads=past_payloads,
        )  # (B, max_output_len, D) - may be larger than num_items

        # Slice to only the actual number of items (HSTU pads to max_output_len internally)
        full_embeddings = full_embeddings_all[:, :num_items, :]  # (B, num_items, D)

        # 6. Prediction heads - predict all 4 semantic tokens for complete items
        # Each position predicts the next item's [L0, L1, L2, Dedup] tokens
        logits_L0 = self.gr_head_L0(full_embeddings)      # (B, num_items, 256)
        logits_L1 = self.gr_head_L1(full_embeddings)      # (B, num_items, 256)
        logits_L2 = self.gr_head_L2(full_embeddings)      # (B, num_items, 256)
        logits_Dedup = self.gr_head_Dedup(full_embeddings)  # (B, num_items, 19)

        # Return as list since vocab sizes differ (256, 256, 256, 19)
        # List of 4 tensors: [L0_logits, L1_logits, L2_logits, Dedup_logits]
        logits_seq = [logits_L0, logits_L1, logits_L2, logits_Dedup]

        # 7. User representation from last valid item position
        batch_indices = torch.arange(batch_size, device=embeddings.device)
        last_positions = lengths - 1
        u = full_embeddings[batch_indices, last_positions, :]  # (B, D)

        return u, logits_seq, None  # candidate_embeddings not used in Research HSTU

    def _get_item_vector(self, codes):
        """
        将 Semantic Codes (B, Layers) 转化为 Embedding。
        根据你的要求：使用"SemanticID最后一个hidden states"
        实现：将 Codes 视为一个短序列，通过 Embedding 层 (或者 InputLayer)，
        这里简单起见，取所有 Code Embedding 的 Sum 或 Last。
        """
        # (B, Layers, D)
        embs = self.input_layer.sem_emb(codes)
        
        # 方式 A: Sum (常用，信息无损)
        # return torch.sum(embs, dim=1)
        
        # 方式 B: Last Hidden State (你的要求)
        # return embs[:, -1, :]
        
        # 方式 C: 既然是 Hidden State，可能需要过一层 MLP 融合
        # 这里取 Sum 作为最稳健的表示
        return torch.sum(embs, dim=1)

    @torch.no_grad()
    def _beam_search_hard_negatives(self, batch_dict, u_current, beam_width=5, grid_mapper=None):
        """
        Rank-Matched Top-K beam search for parallel 4-layer prediction.

        For each semantic layer, independently select top-k predictions,
        then combine them by matching ranks (rank-1 with rank-1, rank-2 with rank-2, etc).

        This is MUCH faster than full cartesian product (k combinations vs k^4).
        Assumes layers are somewhat correlated (top codes tend to appear together).

        Args:
            batch_dict: Input batch dict (not used in this version, kept for API compatibility)
            u_current: User state from forward pass (B, D)
            beam_width: Number of candidates to return (k)
            grid_mapper: GridMapper instance (not used in this version, kept for API compatibility)

        Returns:
            beam_results: (B, k, num_layers) tensor of semantic codes (RAW codes, not offset)
        """
        # 1. Get predictions for next item from all 4 GR heads
        # These are the logits for predicting the NEXT item
        logits_L0 = self.gr_head_L0(u_current)        # (B, 256)
        logits_L1 = self.gr_head_L1(u_current)        # (B, 256)
        logits_L2 = self.gr_head_L2(u_current)        # (B, 256)
        logits_Dedup = self.gr_head_Dedup(u_current)  # (B, 19)

        all_logits = [logits_L0, logits_L1, logits_L2, logits_Dedup]

        # 2. For each layer, get top-k predictions (codes only, no log probs needed)
        layer_topk_codes = []  # List of (B, k) tensors

        for layer_idx, logits in enumerate(all_logits):
            # Get top-k codes for this layer
            _, topk_codes = torch.topk(logits, beam_width, dim=-1)
            # topk_codes: (B, k) - RAW codes (0-255 for L0/L1/L2, 0-18 for Dedup)
            layer_topk_codes.append(topk_codes)

        # 3. Rank-Matched Stacking
        # Combine rank-1 from all layers, rank-2 from all layers, etc.
        # Stack along last dimension to get (B, k, 4)
        # pred_L0[:, 0] with pred_L1[:, 0] with pred_L2[:, 0] with pred_Dedup[:, 0] -> rank-1 combo
        # pred_L0[:, 1] with pred_L1[:, 1] with pred_L2[:, 1] with pred_Dedup[:, 1] -> rank-2 combo
        # ...
        beam_results = torch.stack(layer_topk_codes, dim=-1)  # (B, k, 4)

        return beam_results

    def predict_ctr(self, u, batch_dict, pos_codes, device, grid_mapper):
        """
        Memory Bank 构造：
        1. GT Injection (pos_codes)
        2. Hard Negatives (Beam Search from GR) - 剔除 GT
        3. Random Negatives
        """
        B = u.size(0)
        N = self.config.memory_bank_size
        K = self.config.num_hard_negatives
        
        # --- 1. GT Embedding ---
        # pos_codes: (B, Layers)
        pos_emb = self._get_item_vector(pos_codes).unsqueeze(1) # (B, 1, D)
        
        # --- 2. Hard Negatives (Beam Search) ---
        # 这是一个耗时操作，训练时只生成 K 个
        # candidates: (B, Beam_Width, Layers)
        # 为了效率，我们让 Beam_Width = K
        candidates = self._beam_search_hard_negatives(
            batch_dict, u, beam_width=K, grid_mapper=grid_mapper
        )
        
        # 剔除 GT: 比较 candidates 和 pos_codes
        # 如果 candidate == pos_codes, 它是 False Negative, 需要替换
        # 简单处理：如果撞了，就随机替换成一个 random code，或者保留但 label 设为 0 (模型会困惑)
        # 这里采用严谨做法：Mask 掉
        
        # candidates: (B, K, Layers)
        # pos_codes: (B, Layers) -> (B, 1, Layers)
        pos_exp = pos_codes.unsqueeze(1)
        # exact match check: (B, K)
        is_hit = torch.all(candidates == pos_exp, dim=-1)
        
        # 如果命中 GT，替换为随机负样本
        # 构造一个随机掩码
        rand_backup = torch.randint(1, self.config.sem_total_vocab, candidates.shape).to(device)
        candidates = torch.where(is_hit.unsqueeze(-1), rand_backup, candidates)
        
        # 获取 Hard Negative Embeddings
        # candidates: (B, K, Layers) -> flat -> (B*K, Layers) -> emb -> reshape
        hard_embs = self._get_item_vector(candidates.view(-1, self.config.sem_id_layers))
        hard_embs = hard_embs.view(B, K, -1) # (B, K, D)
        
        # --- 3. Random Negatives ---
        num_easy = N - 1 - K
        rand_codes = torch.randint(1, self.config.sem_total_vocab, (B, num_easy, self.config.sem_id_layers)).to(device)
        easy_embs = self.input_layer.sem_emb(rand_codes).sum(dim=2) # Simple Sum for random
        
        # --- 4. Construct Bank ---
        # bank: (B, 1+K+Easy, D)
        bank = torch.cat([pos_emb, hard_embs, easy_embs], dim=1)
        
        # --- 5. Labels & Shuffle ---
        labels = torch.zeros((B, N), device=device)
        labels[:, 0] = 1.0 # GT is at 0
        
        perm = torch.randperm(N).to(device)
        bank = bank[:, perm, :]
        labels = labels[:, perm]
        
        # --- 6. Scoring ---
        bank_proj = self.cand_proj(bank)
        u_exp = u.unsqueeze(1).repeat(1, N, 1)
        feats = [bank_proj, u_exp]
        
        if self.config.ctr_use_self_attn:
            self_out, _ = self.cand_self_attn(bank_proj, bank_proj, bank_proj)
            feats.append(self_out)
        if self.config.ctr_use_cross_attn:
            u_q = u.unsqueeze(1)
            cross_out, _ = self.user_cross_attn(u_q, bank_proj, bank_proj)
            feats.append(cross_out.repeat(1, N, 1))
            
        final_feats = torch.cat(feats, dim=-1)
        ctr_logits = self.scorer(final_feats).squeeze(-1)

        return ctr_logits, labels

    @torch.no_grad()
    def generate_gr_candidates(self, batch_dict, k=10, grid_mapper=None):
        """
        Generate top-k candidate items using beam search for evaluation

        Args:
            batch_dict: Input batch containing user history (sem_history, etc.)
            k: Number of candidates to generate (beam width)
            grid_mapper: GridMapper instance for converting semantic codes to item IDs

        Returns:
            candidates: (B, k) tensor of predicted item indices
        """
        # 1. Forward pass to get user state
        u, _, _ = self.forward(batch_dict)  # forward returns (u, logits_seq, None)

        # 2. Use beam search to generate top-k semantic code candidates
        # beam_results shape: (B, k, num_layers)
        beam_results = self._beam_search_hard_negatives(
            batch_dict,
            u,
            beam_width=k,
            grid_mapper=grid_mapper
        )

        # 3. Convert semantic codes to item indices
        B, beam_width, num_layers = beam_results.shape

        # Flatten to (B*k, num_layers) for batch processing
        codes_flat = beam_results.view(-1, num_layers)

        # Convert each semantic code sequence to item ID
        item_indices = []
        debug_printed = False
        for i in range(codes_flat.size(0)):
            codes_raw = codes_flat[i].cpu().tolist()  # RAW codes: [0-255, 0-255, 0-255, 0-18]

            # Use grid_mapper to reverse lookup: semantic_codes -> item_id
            if grid_mapper:
                # IMPORTANT: grid_mapper.codes_to_item() expects OFFSET codes!
                # Apply offsets: [L0+1, L1+257, L2+513, Dedup+769]
                codes_offset = grid_mapper._apply_offset(codes_raw)

                # Use nearest neighbor fallback for invalid code combinations
                # (Model predicts layers independently, so not all combinations are valid)
                item_id = grid_mapper.codes_to_item_nearest(codes_offset)

                # DEBUG: Print first 3 lookups to diagnose the issue
                if not debug_printed and i < 3:
                    print(f"\n[DEBUG] Lookup #{i}:")
                    print(f"  codes_raw: {codes_raw}")
                    print(f"  codes_offset: {codes_offset}")
                    print(f"  item_id result: {item_id} (nearest neighbor)")
                    if i == 0:
                        # Print sample of reverse_mapping keys
                        sample_keys = list(grid_mapper.reverse_mapping.keys())[:5]
                        print(f"  Sample reverse_mapping keys: {sample_keys}")
                        print(f"  Total keys in reverse_mapping: {len(grid_mapper.reverse_mapping)}")
                        print(f"  [Note] Model predicts layers independently → most predictions need NN mapping")
                    if i == 2:
                        debug_printed = True

                # If code doesn't map to any item (shouldn't happen), use 0
                item_indices.append(item_id if item_id is not None else 0)
            else:
                # Fallback if no grid_mapper provided
                item_indices.append(0)

        # 4. Reshape back to (B, k)
        candidates = torch.tensor(item_indices, dtype=torch.long, device=u.device)
        candidates = candidates.view(B, beam_width)

        return candidates
