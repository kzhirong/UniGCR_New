"""
Analyze Training Data Distribution

This script analyzes:
1. Code distribution per layer (L0, L1, L2, Dedup)
2. Most common code combinations
3. Item frequency distribution in training sequences
"""

import torch
import json
import numpy as np
from collections import Counter, defaultdict

print("=" * 80)
print("TRAINING DATA ANALYSIS")
print("=" * 80)

# ============================================================
# Part 1: Semantic Code Distribution
# ============================================================
print("\n" + "=" * 80)
print("PART 1: SEMANTIC CODE DISTRIBUTION")
print("=" * 80)

# Load semantic_id_kmean.pt
semantic_ids = torch.load('data/semantic_id_kmean.pt')
print(f"\n[1.1] Loaded semantic_id_kmean.pt: {semantic_ids.shape}")
print(f"      Shape: [num_layers={semantic_ids.shape[0]}, num_items={semantic_ids.shape[1]}]")

num_layers, num_items = semantic_ids.shape

# Analyze each layer
layer_names = ['L0 (Coarse)', 'L1 (Subcategory)', 'L2 (Details)', 'Dedup (Collision)']

print(f"\n[1.2] Code Distribution per Layer:")
print("-" * 80)

for layer_idx in range(num_layers):
    layer_codes = semantic_ids[layer_idx].numpy()
    unique_codes = np.unique(layer_codes)
    code_counts = Counter(layer_codes)

    print(f"\n{layer_names[layer_idx]}:")
    print(f"  Unique codes used: {len(unique_codes)} (min={unique_codes.min()}, max={unique_codes.max()})")
    print(f"  Expected vocab size: {256 if layer_idx < 3 else 19}")
    print(f"  Dead codes (never used): {(256 if layer_idx < 3 else 19) - len(unique_codes)}")

    # Show top-10 most common codes
    most_common = code_counts.most_common(10)
    print(f"  Top-10 most frequent codes:")
    for code, count in most_common[:10]:
        percentage = (count / num_items) * 100
        print(f"    Code {code:3d}: {count:5d} items ({percentage:5.2f}%)")

    # Calculate entropy (measure of uniformity)
    # Entropy = -Σ p(i) * log2(p(i))
    # Max entropy = log2(vocab_size) for uniform distribution
    probs = np.array([code_counts[code] for code in unique_codes]) / num_items
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(unique_codes))
    entropy_ratio = entropy / max_entropy

    print(f"  Entropy: {entropy:.2f} / {max_entropy:.2f} = {entropy_ratio:.2%}")
    print(f"    (100% = perfectly uniform, <50% = highly skewed)")

# ============================================================
# Part 2: Code Combination Patterns
# ============================================================
print("\n" + "=" * 80)
print("PART 2: CODE COMBINATION PATTERNS")
print("=" * 80)

print(f"\n[2.1] Analyzing 4-layer code combinations...")

# Build combination strings for counting
combinations = []
for item_idx in range(num_items):
    combo = tuple(semantic_ids[:, item_idx].tolist())
    combinations.append(combo)

combo_counts = Counter(combinations)

print(f"  Total items: {num_items}")
print(f"  Unique combinations: {len(combo_counts)}")
print(f"  Collision rate: {((num_items - len(combo_counts)) / num_items * 100):.2f}%")
print(f"    (Items sharing same 4-layer code)")

# Most common combinations
print(f"\n[2.2] Top-20 Most Common Combinations:")
print("-" * 80)
print(f"  {'Rank':<6} {'L0':>4} {'L1':>4} {'L2':>4} {'Ded':>4} {'Count':>6} {'%':>6}")
print("-" * 80)

for rank, (combo, count) in enumerate(combo_counts.most_common(20), 1):
    percentage = (count / num_items) * 100
    print(f"  {rank:<6} {combo[0]:>4} {combo[1]:>4} {combo[2]:>4} {combo[3]:>4} {count:>6} {percentage:>5.2f}%")

# Check if model's common predictions match data
print(f"\n[2.3] Checking Model's Common Predictions:")
model_predictions = [
    (0, 0, 0, 0),      # Padding (shouldn't exist!)
    (14, 72, 227, 1),  # Common prediction
    (194, 81, 198, 2), # Common prediction
    (82, 69, 117, 0),  # Common prediction
]

for pred in model_predictions:
    count = combo_counts.get(pred, 0)
    if count > 0:
        percentage = (count / num_items) * 100
        rank = sorted(combo_counts.values(), reverse=True).index(count) + 1
        print(f"  {pred}: EXISTS in training data ({count} items, {percentage:.2f}%, rank #{rank})")
    else:
        print(f"  {pred}: NOT FOUND in training data (invalid combination!)")

# ============================================================
# Part 3: Item Frequency in Training Sequences
# ============================================================
print("\n" + "=" * 80)
print("PART 3: ITEM FREQUENCY IN TRAINING SEQUENCES")
print("=" * 80)

print(f"\n[3.1] Loading item2idx mapping...")
with open('data/item2idx.json', 'r') as f:
    item2idx = json.load(f)
print(f"  Loaded {len(item2idx)} ASIN → index mappings")

print(f"\n[3.2] Loading train_sequences.json...")
with open('data/train_sequences.json', 'r') as f:
    train_data = json.load(f)

print(f"  Total users: {len(train_data)}")

# Count item occurrences (convert ASIN → index)
item_counter = Counter()
total_interactions = 0

# Handle both dict and list formats
if isinstance(train_data, dict):
    for user_data in train_data.values():
        history = user_data.get('history', [])
        # Convert ASIN to index
        history_indices = [item2idx[asin] for asin in history if asin in item2idx]
        item_counter.update(history_indices)
        total_interactions += len(history_indices)
else:  # list format
    for user_data in train_data:
        history = user_data.get('history', [])
        # Convert ASIN to index
        history_indices = [item2idx[asin] for asin in history if asin in item2idx]
        item_counter.update(history_indices)
        total_interactions += len(history_indices)

print(f"  Total interactions: {total_interactions}")
print(f"  Unique items seen: {len(item_counter)}")
print(f"  Items in semantic_ids: {num_items}")
print(f"  Coverage: {len(item_counter) / num_items * 100:.2f}%")

# Item frequency distribution
frequencies = np.array(list(item_counter.values()))
print(f"\n[3.2] Item Frequency Statistics:")
print(f"  Mean: {frequencies.mean():.2f}")
print(f"  Median: {np.median(frequencies):.2f}")
print(f"  Std: {frequencies.std():.2f}")
print(f"  Min: {frequencies.min()}")
print(f"  Max: {frequencies.max()}")

# Top-20 most popular items
print(f"\n[3.3] Top-20 Most Popular Items:")
print("-" * 80)
print(f"  {'Rank':<6} {'Item ID':>8} {'Count':>8} {'%':>8} {'Codes (L0,L1,L2,Ded)'}")
print("-" * 80)

for rank, (item_id, count) in enumerate(item_counter.most_common(20), 1):
    percentage = (count / total_interactions) * 100
    codes = semantic_ids[:, item_id].tolist()
    print(f"  {rank:<6} {item_id:>8} {count:>8} {percentage:>7.2f}% {tuple(codes)}")

# Check if model predictions are popular items
print(f"\n[3.4] Are Model Predictions Mapping to Popular Items?")
print("  (Checking items that model frequently predicts)")

# Map common prediction combos to items
print(f"  Model often predicts these codes → which items do they map to?")
for pred in model_predictions[:3]:  # Skip padding
    count = combo_counts.get(pred, 0)
    if count > 0:
        # Find items with these codes
        matching_items = []
        for item_idx in range(num_items):
            if tuple(semantic_ids[:, item_idx].tolist()) == pred:
                matching_items.append(item_idx)

        if matching_items:
            item_frequencies = [item_counter.get(item_id, 0) for item_id in matching_items]
            avg_freq = np.mean(item_frequencies)
            print(f"  {pred} → Items {matching_items[:3]}... ({len(matching_items)} total)")
            print(f"    Avg frequency: {avg_freq:.1f} interactions")

# Power law check
print(f"\n[3.5] Power Law Distribution Check:")
percentile_80 = np.percentile(frequencies, 80)
top_20_percent = frequencies >= percentile_80
top_20_coverage = frequencies[top_20_percent].sum() / frequencies.sum() * 100
print(f"  Top 20% of items account for {top_20_coverage:.2f}% of interactions")
print(f"  (Typical e-commerce: 80% - 'Pareto Principle')")

if top_20_coverage > 80:
    print(f"  ⚠️  WARNING: Highly imbalanced! Few popular items dominate.")
    print(f"      Model may learn to predict popular items only.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
