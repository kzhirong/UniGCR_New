"""
Simple integration test for HSTUTransducer in Uni-GCR.

This script tests:
1. Model instantiation with HSTUTransducer
2. Forward pass with mock data
3. Jagged tensor conversions
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

        # Semantic ID parameters
        sem_id_layers=3,  # 3 tokens per item
        sem_total_vocab=1000,  # Vocabulary size
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

    batch_size = 4
    seq_len = 15  # 5 items × 3 tokens/item

    # Create mock batch
    batch_dict = {
        'sem_history': torch.randint(1, 1000, (batch_size, seq_len)),  # (B, N)
        'lengths': torch.tensor([15, 12, 9, 15]),  # Variable lengths
        'num_target_tokens': torch.tensor([3, 3, 3, 3]),  # Last item (3 tokens) is target
    }

    print(f"Input shapes:")
    print(f"  - sem_history: {batch_dict['sem_history'].shape}")
    print(f"  - lengths: {batch_dict['lengths']}")
    print(f"  - num_target_tokens: {batch_dict['num_target_tokens']}")

    try:
        with torch.no_grad():
            u, logits_seq, candidate_embeddings = model(batch_dict)

        print(f"\nOutput shapes:")
        print(f"  - u (user repr): {u.shape} - Expected: ({batch_size}, {config.embed_dim})")
        print(f"  - logits_seq: {logits_seq.shape} - Expected: ({batch_size}, {seq_len}, {config.sem_total_vocab})")
        print(f"  - candidate_embeddings: {candidate_embeddings.shape} - Expected: ({sum(batch_dict['num_target_tokens'])}, {config.embed_dim})")

        # Verify shapes
        assert u.shape == (batch_size, config.embed_dim), f"User repr shape mismatch"
        assert logits_seq.shape == (batch_size, seq_len, config.sem_total_vocab), f"Logits shape mismatch"
        expected_targets = batch_dict['num_target_tokens'].sum().item()
        assert candidate_embeddings.shape == (expected_targets, config.embed_dim), f"Candidate embeddings shape mismatch"

        print("\n✓ Forward pass successful - all shapes correct!")
        return True

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jagged_conversions():
    """Test jagged tensor conversion utilities."""
    print("\n" + "=" * 60)
    print("TEST 3: Jagged Tensor Conversions")
    print("=" * 60)

    from utils.jagged_utils import padded_to_jagged, jagged_to_padded

    # Create test data
    batch_size = 3
    max_len = 10
    embed_dim = 64

    padded = torch.randn(batch_size, max_len, embed_dim)
    lengths = torch.tensor([5, 8, 3])

    print(f"Original padded shape: {padded.shape}")
    print(f"Lengths: {lengths.tolist()}")

    try:
        # Convert to jagged
        jagged, offsets = padded_to_jagged(padded, lengths)
        expected_total = lengths.sum().item()

        print(f"\nJagged tensor shape: {jagged.shape} - Expected: ({expected_total}, {embed_dim})")
        print(f"Offsets: {offsets.tolist()}")

        assert jagged.shape == (expected_total, embed_dim), "Jagged shape mismatch"

        # Convert back to padded
        reconstructed = jagged_to_padded(jagged, lengths, max_len)

        print(f"Reconstructed shape: {reconstructed.shape}")

        # Verify valid positions match
        for i in range(batch_size):
            L = lengths[i].item()
            if not torch.allclose(padded[i, :L], reconstructed[i, :L], atol=1e-6):
                print(f"✗ Mismatch at sequence {i}")
                return False

        print("\n✓ Jagged conversions successful - roundtrip preserves data!")
        return True

    except Exception as e:
        print(f"\n✗ Jagged conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HSTUTransducer Integration Test for Uni-GCR")
    print("=" * 60 + "\n")

    # Test 1: Instantiation
    model, config = test_instantiation()
    if model is None:
        print("\n❌ Cannot proceed - model instantiation failed")
        return False

    # Test 2: Jagged conversions
    if not test_jagged_conversions():
        print("\n❌ Jagged conversions failed")
        return False

    # Test 3: Forward pass
    if not test_forward_pass(model, config):
        print("\n❌ Forward pass failed")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60 + "\n")

    print("Next steps:")
    print("  1. Update data loading to provide 'lengths' and 'num_target_tokens'")
    print("  2. Update training loop to handle new forward() return signature")
    print("  3. Test with actual Beauty dataset")
    print("  4. Fix beam search methods for inference (Phase 2)")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
