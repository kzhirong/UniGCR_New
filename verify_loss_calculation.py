"""
Verify that loss calculation works correctly with 4-layer semantic IDs.

This script tests:
1. Model forward pass returns correct format (list of 4 tensors)
2. Loss calculation handles different vocab sizes correctly
3. Target reshaping works properly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from src.config import UniGCRConfig
from src.model import UniGCRModel

def test_loss_calculation():
    """Test that loss calculation works with 4-layer logits."""
    print("=" * 70)
    print("Testing Loss Calculation with 4-Layer Semantic IDs")
    print("=" * 70)

    # Initialize config and model
    config = UniGCRConfig(
        use_semantic_seq=True,
        use_atomic_seq=False,
        use_cat_profile=False,
        use_num_profile=False,
        sem_id_layers=4,
        sem_id_codebook_size=256,
        sem_id_dedup_size=19,
    )

    model = UniGCRModel(config)
    criterion = nn.CrossEntropyLoss()

    print("\n1. Model Configuration:")
    print(f"   ✓ Semantic layers: {config.sem_id_layers}")
    print(f"   ✓ Codebook size (L0-L2): {config.sem_id_codebook_size}")
    print(f"   ✓ Dedup size: {config.sem_id_dedup_size}")

    # Create mock batch with proper vocab sizes for each layer
    batch_size = 4
    num_items = 5
    num_tokens = num_items * config.sem_id_layers  # 20 tokens

    # Generate tokens with correct vocab sizes
    # Structure: [item0_L0, item0_L1, item0_L2, item0_Dedup, item1_L0, ...]
    sem_history = torch.zeros(batch_size, num_tokens, dtype=torch.long)
    for i in range(num_items):
        # L0, L1, L2: vocab 256
        sem_history[:, i*4 + 0] = torch.randint(1, 256, (batch_size,))
        sem_history[:, i*4 + 1] = torch.randint(1, 256, (batch_size,))
        sem_history[:, i*4 + 2] = torch.randint(1, 256, (batch_size,))
        # Dedup: vocab 19
        sem_history[:, i*4 + 3] = torch.randint(0, 19, (batch_size,))

    batch = {
        'sem_history': sem_history,
        'lengths': torch.tensor([20, 16, 12, 20])
    }

    # Create targets with same structure
    sem_target_flat = torch.zeros(batch_size, num_tokens, dtype=torch.long)
    for i in range(num_items):
        sem_target_flat[:, i*4 + 0] = torch.randint(1, 256, (batch_size,))
        sem_target_flat[:, i*4 + 1] = torch.randint(1, 256, (batch_size,))
        sem_target_flat[:, i*4 + 2] = torch.randint(1, 256, (batch_size,))
        sem_target_flat[:, i*4 + 3] = torch.randint(0, 19, (batch_size,))

    print(f"\n2. Input Shapes:")
    print(f"   ✓ sem_history: {batch['sem_history'].shape}")
    print(f"   ✓ sem_target (flattened): {sem_target_flat.shape}")

    # Forward pass
    with torch.no_grad():
        u, logits_seq, _ = model(batch)

    print(f"\n3. Model Output:")
    print(f"   ✓ User repr: {u.shape}")
    print(f"   ✓ Logits (list of {len(logits_seq)} tensors):")
    for i, logits in enumerate(logits_seq):
        layer_name = ["L0", "L1", "L2", "Dedup"][i]
        vocab_size = logits.size(-1)
        print(f"       [{i}] {layer_name}: {logits.shape} (vocab={vocab_size})")

    # Reshape targets from (B, num_items*4) to (B, num_items, 4)
    sem_target = sem_target_flat.view(batch_size, num_items, config.sem_id_layers)
    print(f"\n4. Target Reshaping:")
    print(f"   ✓ Original: {sem_target_flat.shape}")
    print(f"   ✓ Reshaped: {sem_target.shape} (B, num_items, 4)")

    # Compute loss for each layer
    print(f"\n5. Loss Calculation (per layer):")
    total_loss = 0.0
    layer_losses = []

    for layer_idx, logits_layer in enumerate(logits_seq):
        layer_name = ["L0", "L1", "L2", "Dedup"][layer_idx]

        # Extract targets for this layer
        targets_layer = sem_target[:, :, layer_idx]  # (B, num_items)

        # Flatten
        logits_flat = logits_layer.reshape(-1, logits_layer.size(-1))  # (B*num_items, vocab)
        targets_flat = targets_layer.reshape(-1)  # (B*num_items,)

        # Compute loss
        loss_layer = criterion(logits_flat, targets_flat)
        layer_losses.append(loss_layer.item())
        total_loss += loss_layer

        print(f"   ✓ {layer_name}: {loss_layer.item():.4f} (logits: {logits_flat.shape}, targets: {targets_flat.shape})")

    # Average over layers
    avg_loss = total_loss / config.sem_id_layers
    print(f"\n6. Final Loss:")
    print(f"   ✓ Sum: {total_loss.item():.4f}")
    print(f"   ✓ Average (÷ 4): {avg_loss.item():.4f}")

    # Verify shapes are correct
    print(f"\n7. Verification:")
    assert len(logits_seq) == 4, f"Expected 4 logits tensors, got {len(logits_seq)}"
    assert logits_seq[0].shape == (batch_size, num_items, 256), f"L0 shape mismatch"
    assert logits_seq[1].shape == (batch_size, num_items, 256), f"L1 shape mismatch"
    assert logits_seq[2].shape == (batch_size, num_items, 256), f"L2 shape mismatch"
    assert logits_seq[3].shape == (batch_size, num_items, 19), f"Dedup shape mismatch"
    assert sem_target.shape == (batch_size, num_items, 4), f"Target shape mismatch"
    print("   ✓ All shapes correct!")
    print("   ✓ Loss calculation successful!")
    print("   ✓ Different vocab sizes handled correctly!")

    print("\n" + "=" * 70)
    print("✅ Loss Calculation Test Passed!")
    print("=" * 70)

    print("\nKey Points:")
    print("  • Model returns list of 4 tensors (not stacked tensor)")
    print("  • Each layer has correct vocab size (256, 256, 256, 19)")
    print("  • Loss computed per layer and averaged")
    print("  • Targets reshaped from (B, N*4) to (B, N, 4)")
    print("  • CrossEntropyLoss handles each layer independently")

    return True

if __name__ == "__main__":
    try:
        success = test_loss_calculation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
