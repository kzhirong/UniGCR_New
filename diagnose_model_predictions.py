"""
Diagnose Model Prediction Collapse

This script analyzes:
1. User embedding diversity (are embeddings collapsed?)
2. Prediction head logits entropy (are predictions confident/diverse?)
3. Padding token investigation (why predicting code 0?)
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.config import UniGCRConfig
from src.data_amazon import get_dataloaders
from src.model import UniGCRModel

print("=" * 80)
print("MODEL PREDICTION DIAGNOSIS")
print("=" * 80)

# Load checkpoint
print("\n[Setup] Loading checkpoint...")
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
conf = checkpoint['config']

# Load data
train_dl, test_dl = get_dataloaders(conf)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UniGCRModel(conf)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"  Device: {device}")
print(f"  Test users: {len(test_dl.dataset)}")

# ============================================================
# Part 1: User Embedding Analysis
# ============================================================
print("\n" + "=" * 80)
print("PART 1: USER EMBEDDING ANALYSIS")
print("=" * 80)

print("\n[1.1] Collecting user embeddings from all test users...")

user_embeddings = []
user_ids = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dl, desc="Collecting embeddings")):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Get user representation
        u, _, _ = model.forward(batch)  # u: (batch, 256)

        user_embeddings.append(u.cpu())

        # Limit to 500 users for analysis (faster)
        if batch_idx >= 500 // test_dl.batch_size:
            break

user_embeddings = torch.cat(user_embeddings, dim=0)  # (N, 256)
print(f"  Collected embeddings: {user_embeddings.shape}")

# Compute statistics
mean_emb = user_embeddings.mean(dim=0)  # (256,)
std_emb = user_embeddings.std(dim=0)    # (256,)

print(f"\n[1.2] User Embedding Statistics:")
print(f"  Mean magnitude: {torch.norm(mean_emb).item():.4f}")
print(f"  Std per dimension (mean): {std_emb.mean().item():.4f}")
print(f"  Std per dimension (min): {std_emb.min().item():.6f}")
print(f"  Std per dimension (max): {std_emb.max().item():.4f}")

# Check for collapsed dimensions (std ≈ 0)
collapsed_dims = (std_emb < 0.01).sum().item()
print(f"  Collapsed dimensions (std < 0.01): {collapsed_dims} / 256 ({collapsed_dims/256*100:.1f}%)")

if collapsed_dims > 128:
    print(f"  ⚠️  WARNING: >50% dimensions collapsed! Embeddings are not diverse.")

# Cosine similarity analysis
print(f"\n[1.3] Pairwise Cosine Similarity (sample 100 users):")
sample_size = min(100, user_embeddings.size(0))
sample_embs = user_embeddings[:sample_size]

# Normalize
sample_embs_norm = F.normalize(sample_embs, p=2, dim=1)
similarity_matrix = torch.mm(sample_embs_norm, sample_embs_norm.t())  # (100, 100)

# Get upper triangle (exclude diagonal)
triu_indices = torch.triu_indices(sample_size, sample_size, offset=1)
similarities = similarity_matrix[triu_indices[0], triu_indices[1]].numpy()

print(f"  Mean similarity: {similarities.mean():.4f}")
print(f"  Median similarity: {np.median(similarities):.4f}")
print(f"  Std similarity: {similarities.std():.4f}")
print(f"  Min/Max: {similarities.min():.4f} / {similarities.max():.4f}")
print(f"\n  Interpretation:")
print(f"    < 0.3: Diverse (good)")
print(f"    0.3-0.7: Moderate")
print(f"    > 0.7: Collapsed (bad - all users similar)")

if similarities.mean() > 0.7:
    print(f"  ⚠️  WARNING: High similarity! User embeddings are collapsing.")

# Effective rank (measure of diversity)
# Rank ≈ 256 is good, rank ≈ 1 means collapsed to 1D subspace
print(f"\n[1.4] Effective Rank Analysis:")
_, S, _ = torch.svd(user_embeddings.t())  # Singular values
S_norm = S / S.sum()
effective_rank = torch.exp(-(S_norm * torch.log(S_norm + 1e-10)).sum()).item()
print(f"  Effective rank: {effective_rank:.1f} / 256")
print(f"  (256 = perfectly diverse, <10 = collapsed)")

if effective_rank < 10:
    print(f"  ⚠️  WARNING: Low effective rank! Embeddings span a low-dimensional subspace.")

# ============================================================
# Part 2: Prediction Head Logits Analysis
# ============================================================
print("\n" + "=" * 80)
print("PART 2: PREDICTION HEAD LOGITS ANALYSIS")
print("=" * 80)

print("\n[2.1] Collecting logits from all 4 prediction heads...")

all_logits_L0 = []
all_logits_L1 = []
all_logits_L2 = []
all_logits_Dedup = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dl, desc="Collecting logits")):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Get user representation
        u, _, _ = model.forward(batch)

        # Get logits from 4 heads
        logits_L0 = model.gr_head_L0(u)      # (B, 256)
        logits_L1 = model.gr_head_L1(u)      # (B, 256)
        logits_L2 = model.gr_head_L2(u)      # (B, 256)
        logits_Dedup = model.gr_head_Dedup(u) # (B, 19)

        all_logits_L0.append(logits_L0.cpu())
        all_logits_L1.append(logits_L1.cpu())
        all_logits_L2.append(logits_L2.cpu())
        all_logits_Dedup.append(logits_Dedup.cpu())

        if batch_idx >= 500 // test_dl.batch_size:
            break

all_logits_L0 = torch.cat(all_logits_L0, dim=0)
all_logits_L1 = torch.cat(all_logits_L1, dim=0)
all_logits_L2 = torch.cat(all_logits_L2, dim=0)
all_logits_Dedup = torch.cat(all_logits_Dedup, dim=0)

print(f"  Collected logits from {all_logits_L0.size(0)} users")

# Entropy analysis per head
print(f"\n[2.2] Entropy Analysis (uncertainty measure):")
print(f"  Higher entropy = more uncertain (flat distribution)")
print(f"  Lower entropy = more confident (peaked distribution)")
print("-" * 80)

def compute_entropy(logits):
    """Compute entropy: H = -Σ p(i) log p(i)"""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)  # (N,)
    return entropy

for name, logits in [('L0', all_logits_L0), ('L1', all_logits_L1),
                      ('L2', all_logits_L2), ('Dedup', all_logits_Dedup)]:
    entropy = compute_entropy(logits)
    vocab_size = logits.size(1)
    max_entropy = np.log(vocab_size)

    print(f"\n{name}:")
    print(f"  Mean entropy: {entropy.mean().item():.4f} / {max_entropy:.4f} ({entropy.mean().item()/max_entropy*100:.1f}%)")
    print(f"  Std entropy: {entropy.std().item():.4f}")
    print(f"  Min/Max: {entropy.min().item():.4f} / {entropy.max().item():.4f}")

    if entropy.mean().item() / max_entropy < 0.1:
        print(f"  ⚠️  Very low entropy (<10% of max)! Model is overconfident.")
    elif entropy.mean().item() / max_entropy > 0.9:
        print(f"  ⚠️  Very high entropy (>90% of max)! Model is uncertain/untrained.")

# Top predicted codes
print(f"\n[2.3] Most Frequently Predicted Codes:")
print("-" * 80)

for name, logits in [('L0', all_logits_L0), ('L1', all_logits_L1),
                      ('L2', all_logits_L2), ('Dedup', all_logits_Dedup)]:
    top_codes = torch.argmax(logits, dim=1)  # (N,) - most confident prediction per user
    code_counts = torch.bincount(top_codes, minlength=logits.size(1))

    # Get top-10 most predicted
    top10_indices = torch.argsort(code_counts, descending=True)[:10]

    print(f"\n{name}:")
    print(f"  {'Code':>6} {'Count':>8} {'%':>8}")
    for idx in top10_indices:
        count = code_counts[idx].item()
        percentage = count / logits.size(0) * 100
        print(f"  {idx.item():>6} {count:>8} {percentage:>7.2f}%")

    # Check if code 0 is over-predicted
    if top10_indices[0] == 0:
        print(f"  ⚠️  Code 0 (padding) is most predicted! This is wrong.")

# ============================================================
# Part 3: Padding Token Investigation
# ============================================================
print("\n" + "=" * 80)
print("PART 3: PADDING TOKEN INVESTIGATION")
print("=" * 80)

print("\n[3.1] Checking if code 0 appears in training data...")
semantic_ids = torch.load('data/semantic_id_kmean.pt')
for layer_idx, layer_name in enumerate(['L0', 'L1', 'L2', 'Dedup']):
    has_zero = (semantic_ids[layer_idx] == 0).any().item()
    zero_count = (semantic_ids[layer_idx] == 0).sum().item()
    print(f"  {layer_name}: Code 0 appears {zero_count} times ({'SHOULD BE 0!' if has_zero else 'OK'})")

print(f"\n[3.2] Checking embedding layer padding_idx...")
for layer_idx, layer_name in enumerate(['L0', 'L1', 'L2', 'Dedup']):
    emb_layer = model.input_layer.sem_emb_layers[layer_idx]
    padding_idx = emb_layer.padding_idx
    print(f"  {layer_name}: padding_idx = {padding_idx} ({'OK' if padding_idx == 0 else 'WRONG!'})")

print(f"\n[3.3] Checking prediction head bias for code 0...")
heads = [
    ('L0', model.gr_head_L0),
    ('L1', model.gr_head_L1),
    ('L2', model.gr_head_L2),
    ('Dedup', model.gr_head_Dedup)
]

for name, head in heads:
    bias_0 = head.bias[0].item()
    mean_bias = head.bias.mean().item()
    print(f"  {name}: bias[0] = {bias_0:.4f}, mean_bias = {mean_bias:.4f}")
    if bias_0 > mean_bias + 0.5:
        print(f"    ⚠️  Code 0 bias is unusually HIGH! Model biased toward padding.")

# ============================================================
# Part 4: Ground Truth vs Prediction Comparison
# ============================================================
print("\n" + "=" * 80)
print("PART 4: GROUND TRUTH vs PREDICTION COMPARISON")
print("=" * 80)

print("\n[4.1] Comparing predicted codes with actual target codes...")

# Collect ground truth codes from test set
target_codes_L0 = []
target_codes_L1 = []
target_codes_L2 = []
target_codes_Dedup = []

grid_mapper = test_dl.dataset.grid_mapper

with torch.no_grad():
    for batch_idx, batch in enumerate(test_dl):
        target_item_ids = batch.get('sem_target_eval')
        if target_item_ids is not None:
            for item_id in target_item_ids:
                codes = grid_mapper.mapping.get(item_id.item(), [0, 0, 0, 0])
                target_codes_L0.append(codes[0])
                target_codes_L1.append(codes[1])
                target_codes_L2.append(codes[2])
                target_codes_Dedup.append(codes[3])

        if batch_idx >= 500 // test_dl.batch_size:
            break

# Compare distributions
print(f"\n  {'Layer':<10} {'GT Top Code':<15} {'Pred Top Code':<15} {'Match?'}")
print("-" * 80)

pred_top_L0 = torch.bincount(torch.argmax(all_logits_L0, dim=1)).argmax().item()
pred_top_L1 = torch.bincount(torch.argmax(all_logits_L1, dim=1)).argmax().item()
pred_top_L2 = torch.bincount(torch.argmax(all_logits_L2, dim=1)).argmax().item()
pred_top_Dedup = torch.bincount(torch.argmax(all_logits_Dedup, dim=1)).argmax().item()

from collections import Counter
gt_top_L0 = Counter(target_codes_L0).most_common(1)[0][0]
gt_top_L1 = Counter(target_codes_L1).most_common(1)[0][0]
gt_top_L2 = Counter(target_codes_L2).most_common(1)[0][0]
gt_top_Dedup = Counter(target_codes_Dedup).most_common(1)[0][0]

print(f"  {'L0':<10} {gt_top_L0:<15} {pred_top_L0:<15} {'✓' if gt_top_L0 == pred_top_L0 else '✗'}")
print(f"  {'L1':<10} {gt_top_L1:<15} {pred_top_L1:<15} {'✓' if gt_top_L1 == pred_top_L1 else '✗'}")
print(f"  {'L2':<10} {gt_top_L2:<15} {pred_top_L2:<15} {'✓' if gt_top_L2 == pred_top_L2 else '✗'}")
print(f"  {'Dedup':<10} {gt_top_Dedup:<15} {pred_top_Dedup:<15} {'✓' if gt_top_Dedup == pred_top_Dedup else '✗'}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
