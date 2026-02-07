# Data Flow Verification: 4-Layer Semantic IDs

This document traces the complete data flow through the UniGCR model with 4-layer semantic IDs.

---

## Complete Data Flow

### Stage 1: Input (Flattened Tokens)
**Location:** `model.py:65-66` (InputLayer.forward)

```python
sem_history = input_dict['sem_history']  # (B, N) - flattened tokens
batch_size, total_tokens = sem_history.shape
```

**Data Format:**
- Input shape: `(B, N)` where N = num_items × 4
- Example: 5 items × 4 tokens = 20 total tokens
- Flattened structure: `[item0_L0, item0_L1, item0_L2, item0_Dedup, item1_L0, item1_L1, item1_L2, item1_Dedup, ...]`

**Verification:**
- ✅ Comment at line 62 correctly describes 4 tokens per item
- ✅ Comment at line 62 correctly shows flattened format with Dedup

---

### Stage 2: Reshape to Items
**Location:** `model.py:68-71` (InputLayer.forward)

```python
# Reshape: (B, N) → (B, num_items, num_layers)
# e.g., (4, 16) → (4, 4, 4) for 4 items with 4 tokens each [L0, L1, L2, Dedup]
num_items = total_tokens // self.num_semantic_layers
sem_tokens = sem_history.view(batch_size, num_items, self.num_semantic_layers)
```

**Data Transformation:**
- Input: `(B, 20)` flattened
- Output: `(B, 5, 4)` structured as items × layers
- Division: `20 // 4 = 5` items

**Verification:**
- ✅ Division uses `self.num_semantic_layers` which equals 4
- ✅ Comment shows correct example
- ✅ Reshape logic correct

---

### Stage 3: Embed Each Layer Separately
**Location:** `model.py:13-25` (InputLayer.__init__) and `75-78` (forward)

```python
# Initialization (lines 19-24)
self.sem_emb_layers = nn.ModuleList([
    nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),  # L0: 256 vocab
    nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),  # L1: 256 vocab
    nn.Embedding(config.sem_id_codebook_size, dim, padding_idx=0),  # L2: 256 vocab
    nn.Embedding(config.sem_id_dedup_size, dim, padding_idx=0),     # Dedup: 19 vocab
])

# Forward pass (lines 75-78)
for layer_idx in range(self.num_semantic_layers):  # 0, 1, 2, 3
    layer_tokens = sem_tokens[:, :, layer_idx]  # (B, num_items)
    layer_emb = self.sem_emb_layers[layer_idx](layer_tokens)  # (B, num_items, D)
    layer_embeddings.append(layer_emb)
```

**Embedding Tables:**
- Layer 0 (L0): Embedding(256, 64) - RQ-VAE layer
- Layer 1 (L1): Embedding(256, 64) - RQ-VAE layer
- Layer 2 (L2): Embedding(256, 64) - RQ-VAE layer
- Layer 3 (Dedup): Embedding(19, 64) - Deduplication column

**Verification:**
- ✅ 4 embedding tables created (not 3)
- ✅ First 3 tables use `sem_id_codebook_size` (256)
- ✅ 4th table uses `sem_id_dedup_size` (19)
- ✅ Loop iterates over all 4 layers
- ✅ Each layer embedded separately

---

### Stage 4: Combine Layer Embeddings
**Location:** `model.py:80-82` (InputLayer.forward)

```python
# Combine layer embeddings (sum across layers)
item_embeddings = torch.stack(layer_embeddings, dim=0).sum(dim=0)  # (B, num_items, D)
tokens.append(item_embeddings)
```

**Data Transformation:**
- Input: List of 4 tensors, each `(B, num_items, D)`
- Stack: `(4, B, num_items, D)`
- Sum over dim=0: `(B, num_items, D)` - final item embeddings

**Verification:**
- ✅ All 4 layer embeddings summed
- ✅ Output shape is item-level: `(B, num_items, D)`
- ✅ Each item gets representation from all 4 semantic tokens

---

### Stage 5: Convert Lengths (Token-level → Item-level)
**Location:** `model.py:146-154` (UniGCRModel.forward)

```python
# Extract sequence lengths (convert from token-level to item-level)
lengths_tokens = batch_dict.get('lengths')
if lengths_tokens is not None:
    # Convert token-level lengths to item-level lengths
    # e.g., 12 tokens = 4 items (12 // 3)  ⚠️ COMMENT OUTDATED - should say "// 4"
    lengths = lengths_tokens // self.config.sem_id_layers
else:
    # Fallback: assume no padding
    lengths = torch.full((batch_size,), num_items, dtype=torch.long, device=embeddings.device)
```

**Data Transformation:**
- Input: Token-level lengths, e.g., `[20, 16, 12, 20]`
- Division: `lengths // 4` → `[5, 4, 3, 5]` items
- Example: 20 tokens / 4 tokens per item = 5 items

**Verification:**
- ✅ Division uses `self.config.sem_id_layers` (equals 4)
- ✅ Logic correct
- ⚠️ **ISSUE**: Comment at line 150 says "12 // 3" but should say "20 // 4"

---

### Stage 6: HSTU Processing
**Location:** `model.py:160-173` (UniGCRModel.forward)

```python
# Prepare inputs for Research HSTU
past_ids = torch.arange(num_items, device=embeddings.device).unsqueeze(0).expand(batch_size, -1)
past_embeddings = embeddings  # Pre-computed item embeddings from InputLayer
past_payloads = {}

# Forward through Research HSTU
full_embeddings = self.backbone(
    past_lengths=lengths,
    past_ids=past_ids,           # Dummy IDs (API requirement)
    past_embeddings=past_embeddings,  # Actual item embeddings (pre-computed)
    past_payloads=past_payloads,
)  # (B, num_items, D)
```

**Data Flow:**
- Input: `(B, num_items, D)` item embeddings
- HSTU applies causal self-attention over items (not tokens!)
- Output: `(B, num_items, D)` contextualized item embeddings

**Verification:**
- ✅ HSTU receives item-level embeddings
- ✅ Item-level lengths used
- ✅ Output is item-level

---

### Stage 7: Predict All 4 Semantic Tokens
**Location:** `model.py:98-103` (init) and `175-184` (forward)

```python
# Initialization (lines 100-103)
self.gr_head_L0 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)      # vocab=256
self.gr_head_L1 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)      # vocab=256
self.gr_head_L2 = nn.Linear(config.embed_dim, config.sem_id_codebook_size)      # vocab=256
self.gr_head_Dedup = nn.Linear(config.embed_dim, config.sem_id_dedup_size)      # vocab=19

# Forward pass (lines 177-184)
logits_L0 = self.gr_head_L0(full_embeddings)      # (B, num_items, 256)
logits_L1 = self.gr_head_L1(full_embeddings)      # (B, num_items, 256)
logits_L2 = self.gr_head_L2(full_embeddings)      # (B, num_items, 256)
logits_Dedup = self.gr_head_Dedup(full_embeddings)  # (B, num_items, 19)

# Return as list since vocab sizes differ
logits_seq = [logits_L0, logits_L1, logits_L2, logits_Dedup]
```

**Prediction Structure:**
- 4 separate prediction heads (one per semantic layer)
- Each head predicts for ALL item positions (autoregressive)
- Output vocab sizes:
  - L0: 256 (RQ-VAE codebook)
  - L1: 256 (RQ-VAE codebook)
  - L2: 256 (RQ-VAE codebook)
  - Dedup: 19 (deduplication values 0-18)

**Verification:**
- ✅ 4 prediction heads (not 3)
- ✅ Correct vocab sizes (256, 256, 256, 19)
- ✅ All heads predict from same contextualized embeddings
- ✅ Output is list (not stacked tensor) due to different vocab sizes
- ✅ Each position predicts complete 4-token item

---

### Stage 8: User Representation
**Location:** `model.py:186-189` (UniGCRModel.forward)

```python
# User representation from last valid item position
batch_indices = torch.arange(batch_size, device=embeddings.device)
last_positions = lengths - 1
u = full_embeddings[batch_indices, last_positions, :]  # (B, D)
```

**Data Extraction:**
- Uses item-level lengths to find last valid position
- Extracts contextualized embedding at that position
- Output: `(B, D)` user representation

**Verification:**
- ✅ Uses item-level lengths (not token-level)
- ✅ Correct indexing
- ✅ Output shape correct

---

## Issues Found

### ⚠️ Issue 1: Outdated Docstring (lines 122-139)
**Location:** `model.py:122-139`

**Current (INCORRECT):**
```python
"""
Args:
    batch_dict: {
        'sem_history': (B, N_tokens) - flattened semantic tokens (num_items * 3)  ⚠️ WRONG: should be * 4
        ...
    }

Returns:
    logits_seq: (B, num_items, 3, vocab_size) - complete item predictions  ⚠️ WRONG: not a 3D tensor
        - dim 2 = [L0, L1, L2] semantic tokens per item  ⚠️ WRONG: missing Dedup
    ...
"""
```

**Should Be:**
```python
"""
Args:
    batch_dict: {
        'sem_history': (B, N_tokens) - flattened semantic tokens (num_items * 4)
        'lengths': (B,) - actual sequence length in TOKENS (optional)
        ... other fields (for atomic, profile - not used in Phase 1)
    }

Returns:
    For GR training:
        u: (B, D) - user representation from last item position
        logits_seq: List of 4 tensors with different vocab sizes
            - logits_L0: (B, num_items, 256)
            - logits_L1: (B, num_items, 256)
            - logits_L2: (B, num_items, 256)
            - logits_Dedup: (B, num_items, 19)
        candidate_embeddings: None - not used in Research HSTU
"""
```

### ⚠️ Issue 2: Outdated Comment (line 150)
**Current:** `# e.g., 12 tokens = 4 items (12 // 3)`
**Should Be:** `# e.g., 20 tokens = 5 items (20 // 4)`

---

## Summary

### ✅ Correct Implementation
1. **Config:** 4 layers with correct vocab sizes (256, 256, 256, 19)
2. **Input Processing:** Correctly reshapes flattened tokens to (B, num_items, 4)
3. **Embedding Tables:** 4 tables with layer-specific vocab sizes
4. **Layer Combination:** All 4 layers summed to create item embeddings
5. **Length Conversion:** Token-level → Item-level using division by 4
6. **HSTU Processing:** Item-level processing (not token-level)
7. **Prediction Heads:** 4 heads with correct output dimensions
8. **Output Format:** List of 4 tensors (handles different vocab sizes)

### ⚠️ Needs Fixing
1. **Docstring** (lines 122-139): Still mentions 3 layers, needs update
2. **Comment** (line 150): Example uses "// 3" instead of "// 4"

---

## Data Flow Diagram

```
Input: sem_history (B, 20)  [5 items × 4 tokens]
    ↓
Reshape: (B, 5, 4)  [items × layers]
    ↓
Embed Each Layer:
    Layer 0 → Emb(256, 64) → (B, 5, 64)
    Layer 1 → Emb(256, 64) → (B, 5, 64)
    Layer 2 → Emb(256, 64) → (B, 5, 64)
    Layer 3 → Emb(19, 64)  → (B, 5, 64)
    ↓
Sum Layers: (B, 5, 64)  [item embeddings]
    ↓
Convert Lengths: 20 tokens → 5 items
    ↓
HSTU Backbone: (B, 5, 64) → (B, 5, 64)  [contextualized]
    ↓
Predict 4 Tokens:
    Head L0 → (B, 5, 256)
    Head L1 → (B, 5, 256)
    Head L2 → (B, 5, 256)
    Head Dedup → (B, 5, 19)
    ↓
Output: List of 4 tensors + user repr (B, 64)
```

---

## Alignment with Original Data Flow

The implementation correctly follows the data flow we described:
1. ✅ **Stage 1:** Input flattened tokens
2. ✅ **Stage 2:** Reshape to items
3. ✅ **Stage 3:** Embed each of 4 layers separately
4. ✅ **Stage 4:** Combine layer embeddings
5. ✅ **Stage 5:** Convert lengths token→item
6. ✅ **Stage 6:** HSTU processes items (not tokens)
7. ✅ **Stage 7:** Predict all 4 tokens per item
8. ✅ **Stage 8:** Extract user representation

**Conclusion:** The code correctly implements 4-layer semantic IDs with deduplication. Only documentation needs minor updates.
