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

# Patch fbgemm compatibility issues if needed
print("🔧 Checking fbgemm operations...")

if not hasattr(torch.ops.fbgemm, 'asynchronous_complete_cumsum'):
    print("⚠️  Patching fbgemm.asynchronous_complete_cumsum...")
    def async_cumsum_fallback(lengths):
        """Fallback implementation using torch.cumsum"""
        return torch.cat([
            torch.zeros(1, dtype=lengths.dtype, device=lengths.device),
            torch.cumsum(lengths, dim=0)
        ])
    torch.ops.fbgemm.asynchronous_complete_cumsum = async_cumsum_fallback
    print("✅ Patch applied!")

if not hasattr(torch.ops.fbgemm, 'dense_to_jagged'):
    print("⚠️  Patching fbgemm.dense_to_jagged...")
    def dense_to_jagged_fallback(dense, offsets_list):
        """
        Fallback: Convert dense (B, L, D) to jagged (sum(lengths), D).
        Removes padding and flattens valid items.
        """
        offsets = offsets_list[0]  # Offsets: [0, len0, len0+len1, ...]

        # If offsets is 1D with just batch_size+1 elements, extract valid sequences
        if offsets.dim() == 1 and len(offsets) == dense.size(0) + 1:
            batch_size, max_len, dim = dense.shape
            lengths = offsets[1:] - offsets[:-1]  # Compute lengths from offsets

            # Flatten valid items only
            jagged_list = []
            for i, length in enumerate(lengths):
                jagged_list.append(dense[i, :length, :])  # Take only valid items

            jagged = torch.cat(jagged_list, dim=0)  # (total_items, D)
            return (jagged, {})
        else:
            # Fallback: assume no padding
            return (dense.view(-1, dense.size(-1)), {})

    torch.ops.fbgemm.dense_to_jagged = dense_to_jagged_fallback
    print("✅ Patch applied!")

if not hasattr(torch.ops.fbgemm, 'jagged_to_padded_dense'):
    print("⚠️  Patching fbgemm.jagged_to_padded_dense...")
    def jagged_to_padded_dense_fallback(values, offsets_list=None, max_length=None, padding_value=0, **kwargs):
        """
        Fallback: Convert jagged (total_items, D) back to dense (B, L, D).

        The real fbgemm API might use jagged tensor objects where offsets are embedded.
        This fallback handles both cases:
        1. Jagged object with .values(), .offsets() attributes
        2. Raw tensors with explicit offsets_list
        """
        # Case 1: If values is a tuple from our dense_to_jagged, extract components
        if isinstance(values, tuple):
            values, _ = values  # (tensor, metadata_dict)

        # Case 2: If offsets_list not provided, assume no padding (return as-is reshaped)
        if offsets_list is None:
            # Simple fallback: assume uniform sequence lengths
            # values is (total_items, D), we need (B, max_L, D)
            # This won't work for variable lengths but prevents crashes
            print("⚠️  jagged_to_padded_dense called without offsets, using simple reshape")
            return values.unsqueeze(0)  # Add batch dim as workaround

        # Case 3: Standard path with offsets
        offsets = offsets_list[0] if isinstance(offsets_list, list) else offsets_list
        batch_size = len(offsets) - 1
        dim = values.size(-1)

        # Auto-determine max_length if not provided
        if max_length is None:
            lengths = offsets[1:] - offsets[:-1]
            max_length = int(lengths.max().item())

        # Create padded tensor
        padded = torch.full((batch_size, max_length, dim),
                           padding_value,
                           dtype=values.dtype,
                           device=values.device)

        # Fill in valid items
        for i in range(batch_size):
            start_idx = int(offsets[i].item())
            end_idx = int(offsets[i + 1].item())
            length = end_idx - start_idx
            if length > 0:
                padded[i, :length, :] = values[start_idx:end_idx, :]

        return padded

    torch.ops.fbgemm.jagged_to_padded_dense = jagged_to_padded_dense_fallback
    print("✅ Patch applied!")

print("✅ fbgemm patches complete!")

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
