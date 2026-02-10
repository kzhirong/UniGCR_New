"""
Memory-optimized config for autoregressive training.

The autoregressive architecture requires significantly more memory than parallel prediction
because it loops through all sequence positions. Use this config for training.

Memory usage scales as: O(batch_size × max_items × 4_layers × embed_dim)
- Default config: 64 × 38 × 4 = ~10,000 forward steps per batch
- This config: 16 × 10 × 4 = ~640 forward steps per batch (15x reduction)
"""
from src.config import UniGCRConfig

def get_autoregressive_config():
    """Get memory-optimized config for autoregressive training."""
    conf = UniGCRConfig()

    # Memory optimizations
    conf.batch_size = 256  # Large batch fine: embed_dim=64 model is memory-light; AMP active
    conf.max_seq_len = 41  # 40 tokens = 10 items (reduced from 38 items)
                           # Formula: (max_seq_len - 1) must be divisible by 4
                           # 41 - 1 = 40, and 40 / 4 = 10 items ✓

    # Training settings
    conf.lr = 1e-3
    conf.epochs = 50
    conf.patience = 5

    # Model architecture (keep same as original)
    conf.embed_dim = 64
    conf.hstu_layers = 2
    conf.hstu_heads = 2

    # Semantic ID settings
    conf.use_semantic_seq = True
    conf.sem_id_layers = 4
    conf.sem_id_codebook_size = 256
    conf.sem_id_dedup_size = 19
    conf.grid_mapping_path = "data/semantic_id_kmean.pt"

    # Data settings
    conf.data_path = "data/train_sequences.json"

    return conf

if __name__ == "__main__":
    conf = get_autoregressive_config()
    print("=" * 60)
    print("AUTOREGRESSIVE TRAINING CONFIG")
    print("=" * 60)
    print(f"\nMemory-critical settings:")
    print(f"  batch_size: {conf.batch_size}")
    print(f"  max_seq_len: {conf.max_seq_len} tokens = {(conf.max_seq_len - 1) // 4} items")
    print(f"  Forward steps per batch: {conf.batch_size * ((conf.max_seq_len - 1) // 4) * 4}")
    print(f"\nModel settings:")
    print(f"  embed_dim: {conf.embed_dim}")
    print(f"  hstu_layers: {conf.hstu_layers}")
    print(f"  hstu_heads: {conf.hstu_heads}")
    print(f"\nTraining settings:")
    print(f"  lr: {conf.lr}")
    print(f"  epochs: {conf.epochs}")
    print(f"  patience: {conf.patience}")
    print(f"\nTo use this config in training:")
    print(f"  from config_autoregressive import get_autoregressive_config")
    print(f"  conf = get_autoregressive_config()")
