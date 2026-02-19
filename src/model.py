import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import UniGCRConfig
from .hstu_builder import build_research_hstu


class UnifiedInputLayer(nn.Module):
    """Converts raw token sequences into item-level embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim

        if config.use_semantic_seq:
            # One embedding table per layer: L0, L1, L2 (vocab=256 each)
            self.sem_emb_layers = nn.ModuleList([
                nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),
                nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),
                nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),
            ])
            self.num_semantic_layers = config.sem_id_layers

        if config.use_atomic_seq:
            self.atom_emb = nn.Embedding(config.num_atomic_items + 1, dim, padding_idx=0)

        if config.use_cat_profile:
            self.cat_embs = nn.ModuleList([nn.Embedding(v, dim) for v in config.cat_feature_vocab_sizes])
        if config.use_num_profile:
            self.num_projs = nn.ModuleList([nn.Linear(1, dim) for _ in range(config.num_feature_size)])
        if config.use_cat_profile or config.use_num_profile:
            self.feat_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, input_dict):
        tokens = []

        # Profile prefix
        prefix = []
        if self.config.use_cat_profile and 'cat_feats' in input_dict:
            for i, emb in enumerate(self.cat_embs):
                prefix.append(emb(input_dict['cat_feats'][:, i]).unsqueeze(1))
        if self.config.use_num_profile and 'num_feats' in input_dict:
            for i, proj in enumerate(self.num_projs):
                prefix.append(proj(input_dict['num_feats'][:, i].unsqueeze(-1)).unsqueeze(1))
        if prefix:
            tokens.append(self.feat_mlp(torch.cat(prefix, dim=1)))

        # Atomic sequence
        if self.config.use_atomic_seq and 'atom_history' in input_dict:
            tokens.append(self.atom_emb(input_dict['atom_history']))

        # Semantic sequence: (B, N_tokens) → reshape → embed each layer → sum
        # Tokens are offset-coded: L0 in [1,257), L1 in [257,513), etc.
        if self.config.use_semantic_seq and 'sem_history' in input_dict:
            sem_history = input_dict['sem_history']           # (B, N_tokens)
            B, total_tokens = sem_history.shape
            num_items = total_tokens // self.num_semantic_layers
            sem_tokens = sem_history.view(B, num_items, self.num_semantic_layers)

            layer_offsets = [
                1,
                1 + self.config.sem_id_codebook_size,
                1 + 2 * self.config.sem_id_codebook_size,
                1 + 3 * self.config.sem_id_codebook_size,
            ]

            layer_embs = []
            for li in range(self.num_semantic_layers):
                raw = sem_tokens[:, :, li] - layer_offsets[li]
                raw = torch.clamp(raw, 0, self.sem_emb_layers[li].num_embeddings - 1)
                layer_embs.append(self.sem_emb_layers[li](raw))

            tokens.append(torch.stack(layer_embs, dim=0).sum(dim=0))  # (B, num_items, D)

        if not tokens:
            raise ValueError("No input tokens found in batch")
        return torch.cat(tokens, dim=1)


class UniGCRModel(nn.Module):
    def __init__(self, config: UniGCRConfig):
        super().__init__()
        self.config = config

        self.input_layer = UnifiedInputLayer(config)
        self.backbone = build_research_hstu(config)

        # GR heads: predict all 3 semantic tokens autoregressively
        self.gr_head_L0 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)
        self.gr_head_L1 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)
        self.gr_head_L2 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)

        # Autoregressive combiners: project [u, prev_layer_embs] back to embed_dim
        self.autoregressive_combiner_L1 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.autoregressive_combiner_L2 = nn.Linear(config.embed_dim * 3, config.embed_dim)

        # CTR components (optional)
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
        Args:
            batch_dict: {
                'sem_history': (B, N_tokens) flattened semantic tokens with layer offsets
                'lengths':     (B,) actual sequence length in tokens
            }
        Returns:
            u:          (B, D) user representation from last valid item position
            logits_seq: list of 4 tensors [(B, N, 256), (B, N, 256), (B, N, 256), (B, N, 19)]
            None:       placeholder (no external candidate embeddings)
        """
        embeddings = self.input_layer(batch_dict)              # (B, num_items, D)
        B, num_items, _ = embeddings.shape

        lengths_tokens = batch_dict.get('lengths')
        if lengths_tokens is not None:
            lengths = lengths_tokens // self.config.sem_id_layers
        else:
            lengths = torch.full((B,), num_items, dtype=torch.long, device=embeddings.device)

        assert (lengths > 0).all()
        assert (lengths <= num_items).all()

        past_ids = torch.arange(num_items, device=embeddings.device).unsqueeze(0).expand(B, -1)
        full_output = self.backbone(
            past_lengths=lengths,
            past_ids=past_ids,
            past_embeddings=embeddings,
            past_payloads={},
        )[:, :num_items, :]                                    # (B, num_items, D)

        target_codes_seq = batch_dict.get('target_codes_seq')  # (B, num_items, 3) or None
        tf_ratio = getattr(self, 'teacher_forcing_ratio', 1.0)

        all_logits = [[], [], []]
        for pos in range(num_items):
            logits_list, _ = self.predict_codes_autoregressive(
                full_output[:, pos, :],
                target_codes=target_codes_seq[:, pos, :] if target_codes_seq is not None else None,
                training=self.training,
                teacher_forcing_ratio=tf_ratio,
            )
            for li, logits in enumerate(logits_list):
                all_logits[li].append(logits.unsqueeze(1))

        logits_seq = [torch.cat(layer_logits, dim=1) for layer_logits in all_logits]

        batch_idx = torch.arange(B, device=embeddings.device)
        u = full_output[batch_idx, lengths - 1, :]             # (B, D)

        return u, logits_seq, None

    def predict_codes_autoregressive(self, u, target_codes=None, training=True, teacher_forcing_ratio=1.0):
        """
        Autoregressively predict 3-layer semantic codes for one sequence position.

        A single per-sample scheduled-sampling mask (use_tf) is shared across all 3
        layers so that conditioning is consistent within each sample: a sample either
        uses ground-truth context for all layers or model-predicted context for all.

        Args:
            u:                    (B, D) user state from HSTU
            target_codes:         (B, 3) raw GT codes without offsets, or None
            teacher_forcing_ratio: fraction of samples that use GT context
        Returns:
            logits_list:   [(B, 256), (B, 256), (B, 256)]
            sampled_codes: [(B,), (B,), (B,)]
        """
        logits_list, sampled_codes = [], []

        # One coin flip per sample, shared across all 3 layers (cascade consistency)
        if training and target_codes is not None:
            use_tf = torch.rand(u.size(0), device=u.device) < teacher_forcing_ratio
        else:
            use_tf = None

        def pick(logits, gt_col):
            pred = torch.argmax(logits, dim=1)
            if use_tf is not None:
                return torch.where(use_tf, gt_col, pred)
            return pred

        # L0
        logits_L0 = self.gr_head_L0(u)
        logits_list.append(logits_L0)
        code_L0 = pick(logits_L0, target_codes[:, 0] if target_codes is not None else None)
        sampled_codes.append(code_L0)

        # L1 conditioned on L0
        emb_L0 = self.input_layer.sem_emb_layers[0](torch.clamp(code_L0, 0, 255))
        ctx = self.autoregressive_combiner_L1(torch.cat([u, emb_L0], dim=1))
        logits_L1 = self.gr_head_L1(ctx)
        logits_list.append(logits_L1)
        code_L1 = pick(logits_L1, target_codes[:, 1] if target_codes is not None else None)
        sampled_codes.append(code_L1)

        # L2 conditioned on L0, L1
        emb_L1 = self.input_layer.sem_emb_layers[1](torch.clamp(code_L1, 0, 255))
        ctx = self.autoregressive_combiner_L2(torch.cat([u, emb_L0, emb_L1], dim=1))
        logits_L2 = self.gr_head_L2(ctx)
        logits_list.append(logits_L2)
        code_L2 = pick(logits_L2, target_codes[:, 2] if target_codes is not None else None)
        sampled_codes.append(code_L2)

        return logits_list, sampled_codes

    @torch.no_grad()
    def _constrained_beam_search(self, u_current, grid_mapper, beam_width=10):
        """
        Trie-constrained beam search for 3-layer semantic code prediction.

        At each layer, only codes that appear in at least one real catalog item
        (given the current beam prefix) are allowed. Every completed beam is
        therefore guaranteed to be a valid item — no nearest-neighbour fallback.

        Args:
            u_current:   (B, D) user representation from HSTU
            grid_mapper: GridMapper with .trie built at init
            beam_width:  k candidates to return per user
        Returns:
            candidates: (B, k) tensor of item IDs
        """
        B, device = u_current.size(0), u_current.device
        trie = grid_mapper.trie  # {l0: {l1: {l2: item_id}}}

        # L0
        logits_L0 = self.gr_head_L0(u_current)
        vocab_l0  = logits_L0.size(-1)
        mask_l0   = torch.full((vocab_l0,), float('-inf'), device=device)
        mask_l0[list(trie.keys())] = 0.0
        beam_log_probs, beam_codes_L0 = torch.topk(
            torch.log_softmax(logits_L0 + mask_l0, dim=-1), beam_width, dim=-1
        )  # (B, k)

        u_exp = u_current.unsqueeze(1).expand(B, beam_width, -1)  # (B, k, D)

        # L1
        emb_l0   = self.input_layer.sem_emb_layers[0](beam_codes_L0)
        ctx_l1   = self.autoregressive_combiner_L1(
            torch.cat([u_exp, emb_l0], dim=-1).view(B * beam_width, -1))
        logits_L1 = self.gr_head_L1(ctx_l1)
        vocab_l1  = logits_L1.size(-1)
        mask_l1   = torch.full((B * beam_width, vocab_l1), float('-inf'), device=device)
        for i, l0 in enumerate(beam_codes_L0.view(-1).cpu().tolist()):
            sub = trie.get(l0)
            if sub:
                mask_l1[i, list(sub.keys())] = 0.0
        log_p_L1 = torch.log_softmax(logits_L1 + mask_l1, dim=-1).view(B, beam_width, vocab_l1)
        expanded = (beam_log_probs.unsqueeze(-1) + log_p_L1).view(B, -1)
        beam_log_probs, topk = torch.topk(expanded, beam_width, dim=-1)
        batch_idx     = torch.arange(B, device=device).unsqueeze(1).expand(B, beam_width)
        beam_codes_L0 = beam_codes_L0[batch_idx, topk // vocab_l1]
        beam_codes    = torch.stack([beam_codes_L0, topk % vocab_l1], dim=-1)  # (B, k, 2)

        # L2
        emb_l0   = self.input_layer.sem_emb_layers[0](beam_codes[:, :, 0])
        emb_l1   = self.input_layer.sem_emb_layers[1](beam_codes[:, :, 1])
        ctx_l2   = self.autoregressive_combiner_L2(
            torch.cat([u_exp, emb_l0, emb_l1], dim=-1).view(B * beam_width, -1))
        logits_L2 = self.gr_head_L2(ctx_l2)
        vocab_l2  = logits_L2.size(-1)
        mask_l2   = torch.full((B * beam_width, vocab_l2), float('-inf'), device=device)
        for i, (l0, l1) in enumerate(beam_codes.view(-1, 2).cpu().tolist()):
            sub = trie.get(l0, {}).get(l1)
            if sub:
                mask_l2[i, list(sub.keys())] = 0.0
        log_p_L2 = torch.log_softmax(logits_L2 + mask_l2, dim=-1).view(B, beam_width, vocab_l2)
        expanded = (beam_log_probs.unsqueeze(-1) + log_p_L2).view(B, -1)
        _, topk  = torch.topk(expanded, beam_width, dim=-1)
        beam_codes = beam_codes[batch_idx, topk // vocab_l2]
        beam_codes = torch.cat([beam_codes, (topk % vocab_l2).unsqueeze(-1)], dim=-1)  # (B, k, 3)

        # Exact item ID lookup via trie (all beams guaranteed valid)
        # 3-layer trie leaf is item_id directly: {l0: {l1: {l2: item_id}}}
        item_ids = [
            trie.get(l0, {}).get(l1, {}).get(l2, -1)
            for l0, l1, l2 in beam_codes.view(-1, 3).cpu().tolist()
        ]
        return torch.tensor(item_ids, dtype=torch.long, device=device).view(B, beam_width)

    @torch.no_grad()
    def generate_gr_candidates(self, batch_dict, k=10, grid_mapper=None):
        """
        Generate top-k candidate items using trie-constrained beam search.

        Args:
            batch_dict:  input batch
            k:           number of candidates per user
            grid_mapper: GridMapper instance (required)
        Returns:
            candidates: (B, k) tensor of item IDs
        """
        if grid_mapper is None:
            raise ValueError("grid_mapper is required for generate_gr_candidates")
        u, _, _ = self.forward(batch_dict)
        return self._constrained_beam_search(u, grid_mapper, beam_width=k)
