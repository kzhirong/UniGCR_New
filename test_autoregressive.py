"""Test autoregressive implementation with dummy data"""
import torch
from src.config import UniGCRConfig
from src.model import UniGCRModel

print("=" * 60)
print("Testing Autoregressive Implementation")
print("=" * 60)

# Create config
conf = UniGCRConfig()
conf.use_semantic_seq = True
conf.sem_id_layers = 4
conf.sem_id_codebook_size = 256
conf.sem_id_dedup_size = 19
conf.embed_dim = 64
conf.hstu_layers = 2
conf.hstu_heads = 4
conf.max_len = 50

print(f"\n[1/5] Creating model...")
model = UniGCRModel(conf)
model.eval()
print(f"✅ Model created successfully")

# Create dummy batch
B = 4  # batch size
num_items = 10  # sequence length

print(f"\n[2/5] Creating dummy batch (B={B}, num_items={num_items})...")

# Semantic history: (B, num_items * 4) with offsets
sem_history = torch.randint(1, 788, (B, num_items * 4))

# Lengths
lengths = torch.tensor([num_items] * B)

# Target codes for teacher forcing: (B, num_items, 4) RAW codes (no offsets)
target_codes_seq = torch.stack([
    torch.randint(0, 256, (B, num_items)),  # L0
    torch.randint(0, 256, (B, num_items)),  # L1
    torch.randint(0, 256, (B, num_items)),  # L2
    torch.randint(0, 19, (B, num_items)),   # Dedup
], dim=-1)

batch = {
    'sem_history': sem_history,
    'lengths': lengths,
    'target_codes_seq': target_codes_seq
}

print(f"✅ Batch created:")
print(f"   - sem_history: {sem_history.shape}")
print(f"   - lengths: {lengths.shape}")
print(f"   - target_codes_seq: {target_codes_seq.shape}")

print(f"\n[3/5] Testing forward pass...")
try:
    with torch.no_grad():
        output = model(batch)
    print(f"✅ Forward pass successful!")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n[4/5] Checking output shapes...")
logits_seq = output['logits_seq']
print(f"Output logits_seq is a list of {len(logits_seq)} tensors:")
for i, logits in enumerate(logits_seq):
    expected_vocab = 256 if i < 3 else 19
    print(f"   Layer {i}: {logits.shape} (expected: ({B}, {num_items}, {expected_vocab}))")

    if logits.shape != (B, num_items, expected_vocab):
        print(f"❌ Shape mismatch!")
        exit(1)

print(f"✅ All shapes correct!")

print(f"\n[5/5] Testing beam search...")
try:
    # Get user representations from the model
    with torch.no_grad():
        # Run forward to get full_embeddings (user states)
        _ = model(batch)

        # Now test beam search with the last position's user state
        # In practice, this would be called from predict_ctr
        # Let's just verify the method exists and can be called
        u_current = torch.randn(B, conf.embed_dim)  # Dummy user state
        beam_results = model._beam_search_hard_negatives(
            batch_dict=batch,
            u_current=u_current,
            beam_width=5
        )

        print(f"✅ Beam search successful!")
        print(f"   Beam results shape: {beam_results.shape} (expected: ({B}, 5, 4))")

        if beam_results.shape != (B, 5, 4):
            print(f"❌ Shape mismatch!")
            exit(1)

        # Check if codes are in valid range
        for layer_idx in range(4):
            layer_codes = beam_results[:, :, layer_idx]
            min_val = layer_codes.min().item()
            max_val = layer_codes.max().item()
            expected_max = 255 if layer_idx < 3 else 18
            print(f"   Layer {layer_idx}: min={min_val}, max={max_val} (expected: 0-{expected_max})")

            if min_val < 0 or max_val > expected_max:
                print(f"❌ Codes out of range!")
                exit(1)

except Exception as e:
    print(f"❌ Beam search failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nAutoregressive implementation is working correctly.")
print("Ready to proceed with training.")
