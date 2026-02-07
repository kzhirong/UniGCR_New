"""
Simple integration test for Research HSTU in Uni-GCR.

This script tests:
1. Model instantiation with Research HSTU
2. Forward pass with mock data
3. Output shape verification
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import using absolute imports
from src.config import UniGCRConfig
from src.model import UniGCRModel

def test_instantiation():
    """Test that model can be instantiated."""
    print("=" * 60)
    print("TEST 1: Model Instantiation")
    print("=" * 60)

    config = UniGCRConfig(
        # GR-only configuration
        enable_ctr=False,
        use_semantic_seq=True,
        use_atomic_seq=False,
        use_cat_profile=False,
        use_num_profile=False,

        # HSTU parameters
        embed_dim=64,
        hstu_layers=2,
        hstu_heads=2,
        dropout=0.1,
        max_seq_len=50,

        # Semantic ID parameters (RQ-VAE + Deduplication)
        sem_id_layers=4,  # 4 tokens per item (L0, L1, L2, Dedup)
        sem_id_codebook_size=256,  # Codebook size for L0, L1, L2
        sem_id_dedup_size=19,  # Deduplication column vocabulary
    )

    try:
        model = UniGCRModel(config)
        print("✓ Model instantiated successfully")
        print(f"  - Backbone type: {type(model.backbone).__name__}")
        print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model, config
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, config):
    """Test forward pass with mock data."""
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)

    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cpu':
        print("⚠️  Warning: Running on CPU. Research HSTU may use GPU-optimized kernels.")

    # Move model to device
    model = model.to(device)

    batch_size = 4
    num_items = 5  # 5 items in history
    num_tokens = num_items * config.sem_id_layers  # 5 items × 4 tokens = 20

    # Create mock batch with proper vocab sizes for each layer
    # Structure: [item0_L0, item0_L1, item0_L2, item0_Dedup, item1_L0, item1_L1, item1_L2, item1_Dedup, ...]
    sem_history = torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device)

    for i in range(num_items):
        # L0, L1, L2: vocab size 256
        sem_history[:, i*4 + 0] = torch.randint(1, config.sem_id_codebook_size, (batch_size,), device=device)
        sem_history[:, i*4 + 1] = torch.randint(1, config.sem_id_codebook_size, (batch_size,), device=device)
        sem_history[:, i*4 + 2] = torch.randint(1, config.sem_id_codebook_size, (batch_size,), device=device)
        # Dedup: vocab size 19 (values 0-18)
        sem_history[:, i*4 + 3] = torch.randint(0, config.sem_id_dedup_size, (batch_size,), device=device)

    batch_dict = {
        'sem_history': sem_history,  # (B, N_tokens)
        'lengths': torch.tensor([20, 16, 12, 20], device=device),  # Variable lengths in tokens (must be divisible by 4)
    }

    print(f"Input shapes:")
    print(f"  - sem_history: {batch_dict['sem_history'].shape} (flattened tokens)")
    print(f"  - lengths (tokens): {batch_dict['lengths']}")
    print(f"  - num_items: {num_items} ({num_tokens} tokens / {config.sem_id_layers} layers)")

    try:
        with torch.no_grad():
            u, logits_seq, candidate_embeddings = model(batch_dict)

        print(f"\nOutput shapes:")
        print(f"  - u (user repr): {u.shape}")
        print(f"  - logits_seq (list of 4 tensors):")
        for i, logits in enumerate(logits_seq):
            layer_name = ["L0", "L1", "L2", "Dedup"][i]
            print(f"      [{i}] {layer_name}: {logits.shape}")
        print(f"  - candidate_embeddings: {candidate_embeddings}")

        # Verify shapes (logits_seq is now a list of 4 tensors with different vocab sizes)
        assert u.shape == (batch_size, config.embed_dim), \
            f"User repr shape mismatch: expected ({batch_size}, {config.embed_dim}), got {u.shape}"

        assert isinstance(logits_seq, list) and len(logits_seq) == 4, \
            f"logits_seq should be list of 4 tensors, got {type(logits_seq)} with length {len(logits_seq) if isinstance(logits_seq, list) else 'N/A'}"

        # Check each layer's logits shape
        expected_shapes = [
            (batch_size, num_items, config.sem_id_codebook_size),  # L0: 256
            (batch_size, num_items, config.sem_id_codebook_size),  # L1: 256
            (batch_size, num_items, config.sem_id_codebook_size),  # L2: 256
            (batch_size, num_items, config.sem_id_dedup_size),     # Dedup: 19
        ]
        for i, (logits, expected_shape) in enumerate(zip(logits_seq, expected_shapes)):
            layer_name = ["L0", "L1", "L2", "Dedup"][i]
            assert logits.shape == expected_shape, \
                f"{layer_name} logits shape mismatch: expected {expected_shape}, got {logits.shape}"

        assert candidate_embeddings is None, \
            f"Candidate embeddings should be None for Research HSTU, got {candidate_embeddings}"

        print("\n✓ Forward pass successful - all shapes correct!")
        print(f"  ✓ User representations: {u.shape}")
        print(f"  ✓ Sequence logits (4 layers):")
        print(f"      - L0: {logits_seq[0].shape} (vocab=256)")
        print(f"      - L1: {logits_seq[1].shape} (vocab=256)")
        print(f"      - L2: {logits_seq[2].shape} (vocab=256)")
        print(f"      - Dedup: {logits_seq[3].shape} (vocab=19)")
        print(f"  ✓ Complete item predictions: All 4 semantic tokens [L0, L1, L2, Dedup]")
        print(f"  ✓ Item-level predictions: {num_items} items (not {num_tokens} tokens)")
        print(f"  ✓ Full autoregressive predictions at all {num_items} item positions")
        return True

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Research HSTU Integration Test for Uni-GCR")
    print("=" * 60 + "\n")

    # Test 1: Instantiation
    model, config = test_instantiation()
    if model is None:
        print("\n❌ Cannot proceed - model instantiation failed")
        return False

    # Test 2: Forward pass
    if not test_forward_pass(model, config):
        print("\n❌ Forward pass failed")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60 + "\n")

    print("Migration Summary:")
    print("  ✓ Switched from HSTUTransducer → Research HSTU")
    print("  ✓ Removed jagged tensor conversions (simpler pipeline)")
    print("  ✓ Fixed token-vs-item issue: HSTU now sees items, not tokens")
    print("  ✓ Separate embeddings per semantic layer (L0, L1, L2, Dedup)")
    print("  ✓ Four prediction heads for COMPLETE item prediction")
    print("    - Each position predicts all 4 semantic tokens [L0, L1, L2, Dedup]")
    print("    - L0, L1, L2: RQ-VAE layers (vocab=256 each)")
    print("    - Dedup: Collision resolution layer (vocab=19)")
    print("  ✓ Full autoregressive prediction at all ITEM positions")
    print("  ✓ Clean [B, num_items, D] tensor flow throughout")

    print("\nNext steps:")
    print("  1. Update data loading to provide 'lengths' field")
    print("  2. Update training loop to handle new forward() signature")
    print("  3. Test with actual Beauty dataset")
    print("  4. Implement beam search for inference (Phase 2)")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
