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

        # 3b. Autoregressive Combiners - project concatenated embeddings back to embed_dim
        # These allow each layer to condition on previous layer predictions
        self.autoregressive_combiner_L1 = nn.Linear(config.embed_dim * 2, config.embed_dim)     # u + emb_L0
        self.autoregressive_combiner_L2 = nn.Linear(config.embed_dim * 3, config.embed_dim)     # u + emb_L0 + emb_L1
        self.autoregressive_combiner_Dedup = nn.Linear(config.embed_dim * 4, config.embed_dim)  # u + emb_L0 + emb_L1 + emb_L2

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

        # 6. Autoregressive Prediction - each layer conditioned on previous layers
        # Each position predicts the next item's [L0, L1, L2, Dedup] tokens autoregressively

        # Prepare target codes for teacher forcing (if available during training)
        target_codes_seq = batch_dict.get('target_codes_seq', None)  # (B, num_items, 4) if provided

        # Initialize lists to collect logits for each layer across all positions
        all_logits = [[], [], [], []]  # List for each layer: [L0_list, L1_list, L2_list, Dedup_list]

        # Loop through each position and predict autoregressively
        for pos in range(num_items):
            u_pos = full_embeddings[:, pos, :]  # (B, D) - user representation at this position

            # Get target codes for this position (for teacher forcing during training)
            if target_codes_seq is not None:
                target_codes_pos = target_codes_seq[:, pos, :]  # (B, 4)
            else:
                target_codes_pos = None

            # Autoregressive prediction for this position
            # Returns: [(B, 256), (B, 256), (B, 256), (B, 19)]
            # Get teacher forcing ratio (default 1.0 for full teacher forcing)
            teacher_forcing_ratio = getattr(self, 'teacher_forcing_ratio', 1.0)

            logits_list, _ = self.predict_codes_autoregressive(
                u_pos,
                target_codes=target_codes_pos,
                training=self.training,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            # Collect logits for each layer
            for layer_idx, logits in enumerate(logits_list):
                all_logits[layer_idx].append(logits.unsqueeze(1))  # (B, 1, vocab)

        # Stack logits across positions: List of [(B, 1, vocab)] -> (B, num_items, vocab)
        logits_seq = [torch.cat(layer_logits, dim=1) for layer_logits in all_logits]
        # Result: [(B, num_items, 256), (B, num_items, 256), (B, num_items, 256), (B, num_items, 19)]

        # 7. User representation from last valid item position
        batch_indices = torch.arange(batch_size, device=embeddings.device)
        last_positions = lengths - 1
        u = full_embeddings[batch_indices, last_positions, :]  # (B, D)

        return u, logits_seq, None  # candidate_embeddings not used in Research HSTU

    def predict_codes_autoregressive(self, u, target_codes=None, training=True, teacher_forcing_ratio=1.0):
        """
        Autoregressive prediction of 4-layer semantic codes.
        Each layer is predicted conditioned on previous layers.

        Args:
            u: (B, D) user representation from HSTU
            target_codes: (B, 4) ground truth RAW codes (for teacher forcing during training)
                         Codes are WITHOUT offsets: [0-255, 0-255, 0-255, 0-18]
            training: If True, use teacher forcing; else use greedy sampling
            teacher_forcing_ratio: Probability of using ground truth vs model prediction (0.0-1.0)
                                   1.0 = always use ground truth (full teacher forcing)
                                   0.0 = always use model predictions (no teacher forcing)

        Returns:
            logits_list: List of 4 tensors [(B, 256), (B, 256), (B, 256), (B, 19)]
            sampled_codes: List of 4 tensors [(B,), (B,), (B,), (B,)] - codes used for conditioning
        """
        logits_list = []
        sampled_codes = []

        # ── Per-sample scheduled sampling mask ───────────────────────────────────
        # FIX: The old code called torch.rand(1) once per layer, producing a single
        # scalar applied to the whole batch of B samples — so an entire batch was
        # either fully teacher-forced or fully autoregressive, not a proper per-sample
        # mix.  Two bugs fixed here:
        #
        #   Bug 1 (per-batch → per-sample): torch.rand(B) gives each sample its own
        #   independent coin flip, so ~TF% of SAMPLES use GT context every batch.
        #   This gives stable, low-variance gradient signal at every step.
        #
        #   Bug 2 (cascade consistency): the old code flipped independently for each
        #   of the 4 layers, so a sample could get L0=GT, L1=pred(given GT_L0),
        #   L2=GT — an incoherent mixed chain.  Now a single mask is shared across
        #   all 4 layers: if sample b is TF, ALL its layers use GT context; if AR,
        #   ALL use the model's own predictions, maintaining a valid autoregressive chain.
        if training and target_codes is not None:
            use_tf = torch.rand(u.size(0), device=u.device) < teacher_forcing_ratio  # (B,) bool
        else:
            use_tf = None  # Inference: always use model predictions (greedy / beam)

        # ============================================================
        # Layer 0: Predict L0 (unconditional, only depends on user state)
        # ============================================================
        context = u  # Start with user embedding
        logits_L0 = self.gr_head_L0(context)  # (B, 256)
        logits_list.append(logits_L0)

        # Per-sample scheduled sampling: ~TF% of samples use GT, rest use prediction
        if use_tf is not None:
            code_L0 = torch.where(use_tf, target_codes[:, 0], torch.argmax(logits_L0, dim=1))
        else:
            code_L0 = torch.argmax(logits_L0, dim=1)  # Greedy at inference
        sampled_codes.append(code_L0)

        # ============================================================
        # Layer 1: Predict L1 conditioned on L0
        # ============================================================
        # IMPORTANT: Clamp code to valid range before embedding (handle -100 padding)
        code_L0_for_embed = torch.clamp(code_L0, min=0, max=255)
        emb_L0 = self.input_layer.sem_emb_layers[0](code_L0_for_embed)  # (B, D)
        context = torch.cat([u, emb_L0], dim=1)  # (B, 2D)
        context = self.autoregressive_combiner_L1(context)  # (B, D) - project back

        logits_L1 = self.gr_head_L1(context)  # (B, 256)
        logits_list.append(logits_L1)

        # Same use_tf mask → cascade consistency maintained
        if use_tf is not None:
            code_L1 = torch.where(use_tf, target_codes[:, 1], torch.argmax(logits_L1, dim=1))
        else:
            code_L1 = torch.argmax(logits_L1, dim=1)
        sampled_codes.append(code_L1)

        # ============================================================
        # Layer 2: Predict L2 conditioned on L0, L1
        # ============================================================
        # IMPORTANT: Clamp code to valid range before embedding (handle -100 padding)
        code_L1_for_embed = torch.clamp(code_L1, min=0, max=255)
        emb_L1 = self.input_layer.sem_emb_layers[1](code_L1_for_embed)  # (B, D)
        context = torch.cat([u, emb_L0, emb_L1], dim=1)  # (B, 3D)
        context = self.autoregressive_combiner_L2(context)  # (B, D)

        logits_L2 = self.gr_head_L2(context)  # (B, 256)
        logits_list.append(logits_L2)

        # Same use_tf mask → cascade consistency maintained
        if use_tf is not None:
            code_L2 = torch.where(use_tf, target_codes[:, 2], torch.argmax(logits_L2, dim=1))
        else:
            code_L2 = torch.argmax(logits_L2, dim=1)
        sampled_codes.append(code_L2)

        # ============================================================
        # Layer 3: Predict Dedup conditioned on L0, L1, L2
        # ============================================================
        # IMPORTANT: Clamp code to valid range before embedding (handle -100 padding)
        code_L2_for_embed = torch.clamp(code_L2, min=0, max=255)
        emb_L2 = self.input_layer.sem_emb_layers[2](code_L2_for_embed)  # (B, D)
        context = torch.cat([u, emb_L0, emb_L1, emb_L2], dim=1)  # (B, 4D)
        context = self.autoregressive_combiner_Dedup(context)  # (B, D)

        logits_Dedup = self.gr_head_Dedup(context)  # (B, 19)
        logits_list.append(logits_Dedup)

        # Same use_tf mask → cascade consistency maintained
        if use_tf is not None:
            code_Dedup = torch.where(use_tf, target_codes[:, 3], torch.argmax(logits_Dedup, dim=1))
        else:
            code_Dedup = torch.argmax(logits_Dedup, dim=1)
        sampled_codes.append(code_Dedup)

        return logits_list, sampled_codes

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
        Autoregressive beam search for 4-layer semantic code prediction.

        Maintains top-k beams at each layer, where each beam is a partial code sequence.
        For each layer, we:
        1. Predict next layer conditioned on previous layers for all beams
        2. Expand beams by considering all possible next codes
        3. Keep top-k beams by cumulative log probability

        This ensures generated combinations are valid and high-probability under the model.

        Args:
            batch_dict: Input batch dict (not used, kept for API compatibility)
            u_current: User state from forward pass (B, D)
            beam_width: Number of candidates to return (k)
            grid_mapper: GridMapper instance (not used, kept for API compatibility)

        Returns:
            beam_results: (B, k, 4) tensor of semantic codes (RAW codes, not offset)
        """
        B = u_current.size(0)
        device = u_current.device

        # ============================================================
        # Layer 0: Initialize beams with top-k L0 predictions
        # ============================================================
        logits_L0 = self.gr_head_L0(u_current)  # (B, 256)
        log_probs_L0 = torch.log_softmax(logits_L0, dim=-1)  # (B, 256)

        # Get top-k L0 codes and their log probs
        beam_log_probs, beam_codes_L0 = torch.topk(log_probs_L0, beam_width, dim=-1)
        # beam_log_probs: (B, k)
        # beam_codes_L0: (B, k)

        # Initialize beam state
        # We'll accumulate codes as we go: start with L0
        beam_codes = beam_codes_L0.unsqueeze(-1)  # (B, k, 1)

        # ============================================================
        # Layer 1: Predict L1 conditioned on L0 for each beam
        # ============================================================
        # Expand u_current to match beam dimension: (B, k, D)
        u_expanded = u_current.unsqueeze(1).expand(B, beam_width, -1)  # (B, k, D)

        # Get L0 embeddings for all beams
        emb_L0 = self.input_layer.sem_emb_layers[0](beam_codes_L0)  # (B, k, D)

        # Combine with user state and project
        context = torch.cat([u_expanded, emb_L0], dim=-1)  # (B, k, 2D)
        context = context.view(B * beam_width, -1)  # (B*k, 2D)
        context = self.autoregressive_combiner_L1(context)  # (B*k, D)

        # Predict L1
        logits_L1 = self.gr_head_L1(context)  # (B*k, 256)
        logits_L1 = logits_L1.view(B, beam_width, 256)  # (B, k, 256)
        log_probs_L1 = torch.log_softmax(logits_L1, dim=-1)  # (B, k, 256)

        # Expand beams: each of k beams can branch into 256 L1 codes
        # New beam scores: old_score + new_log_prob
        expanded_log_probs = beam_log_probs.unsqueeze(-1) + log_probs_L1  # (B, k, 256)
        expanded_log_probs = expanded_log_probs.view(B, -1)  # (B, k*256)

        # Keep top-k
        beam_log_probs, topk_indices = torch.topk(expanded_log_probs, beam_width, dim=-1)
        # beam_log_probs: (B, k)
        # topk_indices: (B, k) - indices in flattened (k*256) space

        # Decode which beam and which L1 code
        beam_indices = topk_indices // 256  # Which of the k original beams
        codes_L1 = topk_indices % 256  # Which L1 code

        # Update beam_codes: gather L0 codes from selected beams, append L1
        # Use advanced indexing to select beams
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, beam_width)
        beam_codes_L0_selected = beam_codes_L0[batch_indices, beam_indices]  # (B, k)
        beam_codes = torch.stack([beam_codes_L0_selected, codes_L1], dim=-1)  # (B, k, 2)

        # ============================================================
        # Layer 2: Predict L2 conditioned on L0, L1 for each beam
        # ============================================================
        # Get embeddings for current beams
        emb_L0 = self.input_layer.sem_emb_layers[0](beam_codes[:, :, 0])  # (B, k, D)
        emb_L1 = self.input_layer.sem_emb_layers[1](beam_codes[:, :, 1])  # (B, k, D)

        # Combine and predict
        context = torch.cat([u_expanded, emb_L0, emb_L1], dim=-1)  # (B, k, 3D)
        context = context.view(B * beam_width, -1)  # (B*k, 3D)
        context = self.autoregressive_combiner_L2(context)  # (B*k, D)

        logits_L2 = self.gr_head_L2(context)  # (B*k, 256)
        logits_L2 = logits_L2.view(B, beam_width, 256)
        log_probs_L2 = torch.log_softmax(logits_L2, dim=-1)

        # Expand and keep top-k
        expanded_log_probs = beam_log_probs.unsqueeze(-1) + log_probs_L2
        expanded_log_probs = expanded_log_probs.view(B, -1)
        beam_log_probs, topk_indices = torch.topk(expanded_log_probs, beam_width, dim=-1)

        beam_indices = topk_indices // 256
        codes_L2 = topk_indices % 256

        # Update beam_codes
        beam_codes_prev = beam_codes[batch_indices, beam_indices]  # (B, k, 2)
        beam_codes = torch.cat([beam_codes_prev, codes_L2.unsqueeze(-1)], dim=-1)  # (B, k, 3)

        # ============================================================
        # Layer 3: Predict Dedup conditioned on L0, L1, L2 for each beam
        # ============================================================
        emb_L0 = self.input_layer.sem_emb_layers[0](beam_codes[:, :, 0])
        emb_L1 = self.input_layer.sem_emb_layers[1](beam_codes[:, :, 1])
        emb_L2 = self.input_layer.sem_emb_layers[2](beam_codes[:, :, 2])

        context = torch.cat([u_expanded, emb_L0, emb_L1, emb_L2], dim=-1)  # (B, k, 4D)
        context = context.view(B * beam_width, -1)
        context = self.autoregressive_combiner_Dedup(context)

        logits_Dedup = self.gr_head_Dedup(context)  # (B*k, 19)
        logits_Dedup = logits_Dedup.view(B, beam_width, 19)
        log_probs_Dedup = torch.log_softmax(logits_Dedup, dim=-1)

        # Expand and keep top-k
        expanded_log_probs = beam_log_probs.unsqueeze(-1) + log_probs_Dedup
        expanded_log_probs = expanded_log_probs.view(B, -1)  # (B, k*19)
        beam_log_probs, topk_indices = torch.topk(expanded_log_probs, beam_width, dim=-1)

        beam_indices = topk_indices // 19
        codes_Dedup = topk_indices % 19

        # Final beam_codes
        beam_codes_prev = beam_codes[batch_indices, beam_indices]  # (B, k, 3)
        beam_codes = torch.cat([beam_codes_prev, codes_Dedup.unsqueeze(-1)], dim=-1)  # (B, k, 4)

        return beam_codes

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
    def _constrained_beam_search(self, u_current, grid_mapper, beam_width=10):
        """
        Trie-constrained beam search for 4-layer semantic code prediction.

        At each layer, only codes that appear in at least one real catalog item
        (given the current beam prefix) are considered.  Every completed beam is
        therefore guaranteed to correspond to a valid item — no nearest-neighbour
        fallback needed.

        Contrast with _beam_search_hard_negatives (unconstrained): that method
        generates arbitrary code tuples and falls back to Weighted-Hamming
        nearest-neighbour lookup, which (a) wastes beam slots on non-existent
        items and (b) causes multiple beams to collapse to the same nearest item.

        Args:
            u_current:   (B, D) user representation from the HSTU backbone
            grid_mapper: GridMapper instance — must have .trie built at init
            beam_width:  k — number of candidates to return per user

        Returns:
            candidates: (B, k) tensor of item IDs (integer indices into catalog)
        """
        B      = u_current.size(0)
        device = u_current.device
        trie   = grid_mapper.trie           # {l0: {l1: {l2: {d: item_id}}}}

        # ── Layer 0: restrict to L0 codes that exist in the trie ─────────────
        logits_L0 = self.gr_head_L0(u_current)                    # (B, 256)
        vocab_l0  = logits_L0.size(-1)

        mask_l0 = torch.full((vocab_l0,), float('-inf'), device=device)
        mask_l0[list(trie.keys())] = 0.0                           # valid L0 codes → 0

        log_probs_L0    = torch.log_softmax(logits_L0 + mask_l0, dim=-1)
        beam_log_probs, beam_codes_L0 = torch.topk(log_probs_L0, beam_width, dim=-1)
        # beam_log_probs: (B, k)   beam_codes_L0: (B, k) — all guaranteed in trie

        # ── Layer 1: restrict valid L1 codes per beam based on its L0 code ───
        u_exp  = u_current.unsqueeze(1).expand(B, beam_width, -1) # (B, k, D)
        emb_l0 = self.input_layer.sem_emb_layers[0](beam_codes_L0) # (B, k, D)
        ctx_l1 = self.autoregressive_combiner_L1(
            torch.cat([u_exp, emb_l0], dim=-1).view(B * beam_width, -1)
        )                                                           # (B*k, D)
        logits_L1 = self.gr_head_L1(ctx_l1)                        # (B*k, 256)
        vocab_l1  = logits_L1.size(-1)

        # Build per-beam validity mask from trie: different L0 → different valid L1 set
        l0_flat  = beam_codes_L0.view(-1).cpu().tolist()           # (B*k,)
        mask_l1  = torch.full((B * beam_width, vocab_l1), float('-inf'), device=device)
        for i, l0 in enumerate(l0_flat):
            sub = trie.get(l0)
            if sub:
                mask_l1[i, list(sub.keys())] = 0.0

        log_probs_L1 = torch.log_softmax(
            logits_L1 + mask_l1, dim=-1
        ).view(B, beam_width, vocab_l1)                            # (B, k, 256)

        expanded       = (beam_log_probs.unsqueeze(-1) + log_probs_L1).view(B, -1)
        beam_log_probs, topk_idx = torch.topk(expanded, beam_width, dim=-1)
        parent_l1  = topk_idx // vocab_l1                         # which L0 beam
        codes_L1   = topk_idx %  vocab_l1                         # L1 code chosen

        batch_idx     = torch.arange(B, device=device).unsqueeze(1).expand(B, beam_width)
        beam_codes_L0 = beam_codes_L0[batch_idx, parent_l1]       # (B, k) re-aligned
        beam_codes    = torch.stack([beam_codes_L0, codes_L1], dim=-1)  # (B, k, 2)

        # ── Layer 2: restrict valid L2 codes given (L0, L1) ──────────────────
        emb_l0 = self.input_layer.sem_emb_layers[0](beam_codes[:, :, 0])  # (B, k, D)
        emb_l1 = self.input_layer.sem_emb_layers[1](beam_codes[:, :, 1])  # (B, k, D)
        ctx_l2 = self.autoregressive_combiner_L2(
            torch.cat([u_exp, emb_l0, emb_l1], dim=-1).view(B * beam_width, -1)
        )                                                           # (B*k, D)
        logits_L2 = self.gr_head_L2(ctx_l2)                        # (B*k, 256)
        vocab_l2  = logits_L2.size(-1)

        pairs_flat = beam_codes.view(-1, 2).cpu().tolist()         # (B*k, 2)
        mask_l2    = torch.full((B * beam_width, vocab_l2), float('-inf'), device=device)
        for i, (l0, l1) in enumerate(pairs_flat):
            sub = trie.get(l0, {}).get(l1)
            if sub:
                mask_l2[i, list(sub.keys())] = 0.0

        log_probs_L2 = torch.log_softmax(
            logits_L2 + mask_l2, dim=-1
        ).view(B, beam_width, vocab_l2)                            # (B, k, 256)

        expanded       = (beam_log_probs.unsqueeze(-1) + log_probs_L2).view(B, -1)
        beam_log_probs, topk_idx = torch.topk(expanded, beam_width, dim=-1)
        parent_l2  = topk_idx // vocab_l2
        codes_L2   = topk_idx %  vocab_l2

        beam_codes = beam_codes[batch_idx, parent_l2]              # (B, k, 2)
        beam_codes = torch.cat([beam_codes, codes_L2.unsqueeze(-1)], dim=-1)  # (B, k, 3)

        # ── Dedup layer: restrict valid D codes given (L0, L1, L2) ───────────
        emb_l0 = self.input_layer.sem_emb_layers[0](beam_codes[:, :, 0])
        emb_l1 = self.input_layer.sem_emb_layers[1](beam_codes[:, :, 1])
        emb_l2 = self.input_layer.sem_emb_layers[2](beam_codes[:, :, 2])
        ctx_d  = self.autoregressive_combiner_Dedup(
            torch.cat([u_exp, emb_l0, emb_l1, emb_l2], dim=-1).view(B * beam_width, -1)
        )                                                           # (B*k, D)
        logits_D = self.gr_head_Dedup(ctx_d)                       # (B*k, 19)
        vocab_d  = logits_D.size(-1)

        triples_flat = beam_codes.view(-1, 3).cpu().tolist()       # (B*k, 3)
        mask_d       = torch.full((B * beam_width, vocab_d), float('-inf'), device=device)
        for i, (l0, l1, l2) in enumerate(triples_flat):
            sub = trie.get(l0, {}).get(l1, {}).get(l2)
            if sub:
                # sub keys are item_ids at this point — we need only D codes
                # sub is {d_code: item_id}, so sub.keys() = valid D codes
                mask_d[i, list(sub.keys())] = 0.0

        log_probs_D = torch.log_softmax(
            logits_D + mask_d, dim=-1
        ).view(B, beam_width, vocab_d)                             # (B, k, 19)

        expanded       = (beam_log_probs.unsqueeze(-1) + log_probs_D).view(B, -1)
        beam_log_probs, topk_idx = torch.topk(expanded, beam_width, dim=-1)
        parent_d = topk_idx // vocab_d
        codes_D  = topk_idx %  vocab_d

        beam_codes = beam_codes[batch_idx, parent_d]               # (B, k, 3)
        beam_codes = torch.cat([beam_codes, codes_D.unsqueeze(-1)], dim=-1)  # (B, k, 4)

        # ── Exact item ID lookup from trie ────────────────────────────────────
        # All beams are valid items (guaranteed by trie traversal above).
        # Look up item_id directly — no nearest-neighbour needed.
        codes_list = beam_codes.view(-1, 4).cpu().tolist()         # (B*k, 4)
        item_ids   = []
        for l0, l1, l2, d in codes_list:
            item_id = trie.get(l0, {}).get(l1, {}).get(l2, {}).get(d, -1)
            item_ids.append(item_id)

        candidates = torch.tensor(item_ids, dtype=torch.long, device=device)
        return candidates.view(B, beam_width)                      # (B, k)

    @torch.no_grad()
    def generate_gr_candidates(self, batch_dict, k=10, grid_mapper=None):
        """
        Generate top-k candidate items using beam search for evaluation.

        Uses trie-constrained beam search when grid_mapper.trie is available
        (the default after GridMapper.__init__).  Every candidate is then a
        real catalog item and is looked up via exact trie traversal — no
        nearest-neighbour fallback needed.

        Falls back to the legacy unconstrained beam search + Weighted-Hamming
        nearest-neighbour lookup when no grid_mapper is provided.

        Args:
            batch_dict:  Input batch containing user history
            k:           Number of candidates to return (beam width)
            grid_mapper: GridMapper instance (must have .trie attribute)

        Returns:
            candidates: (B, k) tensor of predicted item IDs
        """
        # 1. Forward pass to get user representation
        u, _, _ = self.forward(batch_dict)

        # 2. Constrained beam search — grid_mapper is always required
        if grid_mapper is None:
            raise ValueError("grid_mapper is required for generate_gr_candidates")

        return self._constrained_beam_search(u, grid_mapper, beam_width=k)
