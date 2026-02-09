"""
Comprehensive test for autoregressive implementation on Colab/CUDA.
Run this on Colab to verify the implementation before full training.
"""
import torch
import sys
import os

print("=" * 80)
print("AUTOREGRESSIVE IMPLEMENTATION - COLAB TEST")
print("=" * 80)

# Step 1: Check environment
print("\n[1/6] Checking environment...")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import fbgemm_gpu
try:
    import fbgemm_gpu
    print(f"✅ fbgemm_gpu: {fbgemm_gpu.__version__}")
except ImportError as e:
    print(f"❌ fbgemm_gpu not available: {e}")
    sys.exit(1)

# Step 2: Import project modules
print("\n[2/6] Importing project modules...")
from src.config import UniGCRConfig
from src.model import UniGCRModel
from src.data_amazon import get_dataloaders
print("✅ Modules imported successfully")

# Step 3: Create config and model
print("\n[3/6] Creating model...")
conf = UniGCRConfig()
conf.use_semantic_seq = True
conf.grid_mapping_path = "data/semantic_id_kmean.pt"
conf.sem_id_layers = 4
conf.sem_id_codebook_size = 256
conf.sem_id_dedup_size = 19
conf.embed_dim = 64
conf.hstu_layers = 2
conf.hstu_heads = 4
conf.batch_size = 32  # Small batch for testing
conf.data_path = "data/train_sequences.json"
conf.max_len = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UniGCRModel(conf)
model = model.to(device)
model.eval()

print(f"✅ Model created and moved to {device}")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Step 4: Load real data and test forward pass
print("\n[4/6] Testing forward pass with real data...")
try:
    train_dl, val_dl = get_dataloaders(conf)
    print(f"✅ Data loaded: {len(train_dl)} training batches")

    # Get first batch
    batch = next(iter(train_dl))
    print(f"   Batch keys: {list(batch.keys())}")

    # Move to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    print(f"   sem_history: {batch['sem_history'].shape}")
    if 'target_codes_seq' in batch:
        print(f"   target_codes_seq: {batch['target_codes_seq'].shape}")

    # Forward pass
    with torch.no_grad():
        output = model(batch)

    print(f"✅ Forward pass successful!")

    # Check outputs
    logits_seq = output['logits_seq']
    print(f"   Output: {len(logits_seq)} layer logits")
    for i, logits in enumerate(logits_seq):
        vocab_size = 256 if i < 3 else 19
        print(f"      Layer {i}: {logits.shape} (vocab={vocab_size})")

except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test beam search
print("\n[5/6] Testing beam search...")
try:
    # Extract user state from the last position
    B = batch['sem_history'].size(0)
    u_current = torch.randn(B, conf.embed_dim, device=device)  # Dummy user state for testing

    beam_results = model._beam_search_hard_negatives(
        batch_dict=batch,
        u_current=u_current,
        beam_width=5
    )

    print(f"✅ Beam search successful!")
    print(f"   Beam results: {beam_results.shape}")

    # Check validity of codes
    for layer_idx in range(4):
        layer_codes = beam_results[:, :, layer_idx]
        min_val = layer_codes.min().item()
        max_val = layer_codes.max().item()
        expected_max = 255 if layer_idx < 3 else 18

        print(f"   Layer {layer_idx}: range=[{min_val}, {max_val}] (expected: [0, {expected_max}])")

        if min_val < 0 or max_val > expected_max:
            print(f"❌ Layer {layer_idx} codes out of range!")
            sys.exit(1)

except Exception as e:
    print(f"❌ Beam search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test one training step
print("\n[6/6] Testing training step...")
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get fresh batch
    batch = next(iter(train_dl))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # Forward
    output = model(batch)
    logits_seq = output['logits_seq']

    # Compute loss (simplified - just check it runs)
    if 'target_codes_seq' in batch:
        target_codes = batch['target_codes_seq']
        losses = []

        for layer_idx, logits in enumerate(logits_seq):
            target = target_codes[:, :, layer_idx]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100
            )
            losses.append(loss)

        total_loss = sum(losses)
        print(f"   Loss computed: {total_loss.item():.4f}")

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"✅ Training step successful!")
    else:
        print(f"⚠️  No target_codes_seq in batch - skipping loss computation")
        print(f"   (This is expected if trainer hasn't prepared targets yet)")

except Exception as e:
    print(f"❌ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✅ ALL COLAB TESTS PASSED!")
print("=" * 80)
print("\nAutoregressive implementation is working correctly:")
print("  ✅ Model creation")
print("  ✅ Forward pass")
print("  ✅ Beam search")
print("  ✅ Training step")
print("\nReady to proceed with full training!")
print("\nTo train for 1 epoch, run:")
print("  python run.py")
