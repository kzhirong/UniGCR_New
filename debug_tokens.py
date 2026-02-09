"""Debug script to check token values in data"""
import torch
from src.config import UniGCRConfig
from src.data_amazon import get_dataloaders

# Create config
conf = UniGCRConfig()
conf.use_semantic_seq = True
conf.grid_mapping_path = "data/semantic_id_kmean.pt"
conf.sem_id_layers = 4
conf.sem_id_codebook_size = 256
conf.sem_id_dedup_size = 19
conf.batch_size = 64
conf.data_path = "data/train_sequences.json"

# Get data
train_dl, _ = get_dataloaders(conf)

# Check first batch
batch = next(iter(train_dl))
sem_history = batch['sem_history']
sem_target = batch['sem_target']

print("=== Semantic History (Input) ===")
print(f"Shape: {sem_history.shape}")
print(f"Min value: {sem_history.min().item()}")
print(f"Max value: {sem_history.max().item()}")
print(f"Unique values: {torch.unique(sem_history).shape[0]}")

print("\n=== Semantic Target ===")
print(f"Shape: {sem_target.shape}")
print(f"Min value: {sem_target.min().item()}")
print(f"Max value: {sem_target.max().item()}")

# Reshape to see per-layer stats
B, total_tokens = sem_history.shape
num_items = total_tokens // 4
sem_tokens = sem_history.view(B, num_items, 4)

print("\n=== Per-Layer Analysis (sem_history) ===")
layer_offsets = [1, 257, 513, 769]
layer_names = ['L0', 'L1', 'L2', 'Dedup']
layer_vocab_sizes = [256, 256, 256, 19]

for layer_idx in range(4):
    layer_tokens_offset = sem_tokens[:, :, layer_idx]
    print(f"\n{layer_names[layer_idx]} (offset={layer_offsets[layer_idx]}, vocab_size={layer_vocab_sizes[layer_idx]}):")
    print(f"  Offset tokens - Min: {layer_tokens_offset.min().item()}, Max: {layer_tokens_offset.max().item()}")

    # Remove offset
    layer_tokens = layer_tokens_offset - layer_offsets[layer_idx]
    print(f"  After offset removal - Min: {layer_tokens.min().item()}, Max: {layer_tokens.max().item()}")

    # Check if any are out of bounds
    too_small = (layer_tokens < 0).sum().item()
    too_large = (layer_tokens >= layer_vocab_sizes[layer_idx]).sum().item()
    print(f"  Out of bounds: {too_small} too small, {too_large} too large")

    if too_large > 0:
        bad_indices = layer_tokens >= layer_vocab_sizes[layer_idx]
        bad_values = layer_tokens[bad_indices]
        print(f"  Bad values (>= {layer_vocab_sizes[layer_idx]}): {bad_values[:10]}")
