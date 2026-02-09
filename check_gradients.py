"""
Check if gradients are flowing through all model components.
Run this to diagnose the fbgemm gradient issue.
"""
import torch
from src.config import UniGCRConfig
from src.model import UniGCRModel
from src.data_amazon import get_dataloaders

print("=" * 60)
print("GRADIENT FLOW CHECK")
print("=" * 60)

# Create model
conf = UniGCRConfig()
conf.use_semantic_seq = True
conf.grid_mapping_path = "data/semantic_id_kmean.pt"
conf.data_path = "data/train_sequences.json"
conf.batch_size = 4  # Small batch for testing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UniGCRModel(conf)
model = model.to(device)
model.train()

# Get a batch
train_dl, _ = get_dataloaders(conf)
batch = next(iter(train_dl))
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# Truncate for testing
batch['sem_history'] = batch['sem_history'][:, :20]
batch['sem_target'] = batch['sem_target'][:, :20]
batch['lengths'] = torch.full((4,), 20, dtype=torch.long, device=device)

print(f"\n[1/4] Running forward pass...")

# Prepare target codes
sem_target = batch['sem_target'].view(4, 5, 4)
layer_offsets = torch.tensor([1, 257, 513, 769], device=device)
target_codes_seq = sem_target - layer_offsets.view(1, 1, -1)
target_codes_seq = torch.where(
    target_codes_seq < 0,
    torch.tensor(-100, device=device),
    target_codes_seq
)
batch['target_codes_seq'] = target_codes_seq

# DEBUG: Check if we have any valid targets
print("\nTarget statistics:")
for layer_idx in range(4):
    layer_targets = target_codes_seq[:, :, layer_idx]
    num_valid = (layer_targets >= 0).sum().item()
    num_padding = (layer_targets == -100).sum().item()
    print(f"  Layer {layer_idx}: {num_valid} valid, {num_padding} padding (of {layer_targets.numel()} total)")
    if num_valid == 0:
        print(f"    ⚠️  WARNING: All targets are padding!")

total_valid = (target_codes_seq >= 0).sum().item()
if total_valid == 0:
    print(f"\n❌ CRITICAL: ALL targets are padding (-100)!")
    print("This will cause NaN loss. Using a longer sequence...")
    # Use longer sequence (need to reload the ORIGINAL batch)
    # Reload from dataloader to get full sequence
    batch = next(iter(train_dl))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Don't truncate this time - use full sequence
    print(f"Using full sequence: sem_target shape = {batch['sem_target'].shape}")

    # Reshape sem_target
    B = batch['sem_target'].size(0)
    num_tokens = batch['sem_target'].size(1)
    num_items = num_tokens // 4

    sem_target = batch['sem_target'].view(B, num_items, 4)
    target_codes_seq = sem_target - layer_offsets.view(1, 1, -1)
    target_codes_seq = torch.where(
        target_codes_seq < 0,
        torch.tensor(-100, device=device),
        target_codes_seq
    )
    batch['target_codes_seq'] = target_codes_seq
    print(f"✓ Using full sequence: {num_items} items ({num_tokens} tokens)")

    # Re-check target statistics
    print("\nTarget statistics (full sequence):")
    for layer_idx in range(4):
        layer_targets = target_codes_seq[:, :, layer_idx]
        num_valid = (layer_targets >= 0).sum().item()
        num_padding = (layer_targets == -100).sum().item()
        print(f"  Layer {layer_idx}: {num_valid} valid, {num_padding} padding")

    # Re-run forward with full sequence
    u, logits_seq, _ = model(batch)
    print(f"✓ Forward completed with full sequence: u shape = {u.shape}")
else:
    # Forward with original batch
    u, logits_seq, _ = model(batch)
    print(f"✓ Forward completed: u shape = {u.shape}")

print(f"\n[2/4] Computing loss...")

# Simple loss computation
loss = 0
for layer_idx, logits in enumerate(logits_seq):
    target = target_codes_seq[:, :, layer_idx].long()
    vocab_size = logits.size(-1)
    target = torch.clamp(target, -100, vocab_size - 1)

    layer_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        target.reshape(-1),
        ignore_index=-100
    )
    loss += layer_loss

loss_val = loss.item()
print(f"✓ Loss computed: {loss_val:.4f}")

# Check for NaN loss
if torch.isnan(loss) or torch.isinf(loss):
    print(f"\n❌ CRITICAL: Loss is {loss_val}!")
    print("This indicates a numerical instability issue.")
    print("\nChecking logits for NaN/Inf...")
    for layer_idx, logits in enumerate(logits_seq):
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        print(f"  Layer {layer_idx}: NaN={has_nan}, Inf={has_inf}")
        if has_nan or has_inf:
            print(f"    Min={logits.min().item():.4f}, Max={logits.max().item():.4f}")
    print("\nSkipping backward pass due to NaN loss.")
    exit(1)

print(f"\n[3/4] Running backward pass...")
loss.backward()
print(f"✓ Backward completed")

print(f"\n[4/4] Checking gradient flow...")

# Check if different components have gradients
components = {
    'GR Head L0': model.gr_head_L0.weight,
    'GR Head L1': model.gr_head_L1.weight,
    'GR Head L2': model.gr_head_L2.weight,
    'GR Head Dedup': model.gr_head_Dedup.weight,
    'Combiner L1': model.autoregressive_combiner_L1.weight,
    'Combiner L2': model.autoregressive_combiner_L2.weight,
    'Combiner Dedup': model.autoregressive_combiner_Dedup.weight,
    'Semantic Emb L0': model.input_layer.sem_emb_layers[0].weight,
    'Semantic Emb L1': model.input_layer.sem_emb_layers[1].weight,
    'Semantic Emb L2': model.input_layer.sem_emb_layers[2].weight,
}

# Check HSTU parameters
hstu_has_grad = False
for name, param in model.backbone.named_parameters():
    if param.grad is not None:
        hstu_has_grad = True
        break

print("\nGradient Status:")
print("-" * 60)

all_ok = True
for name, param in components.items():
    has_grad = param.grad is not None
    if has_grad:
        grad_norm = param.grad.norm().item()
        status = "✓" if grad_norm > 1e-8 else "⚠️ (near zero)"
        print(f"{name:25s}: {status:15s} (norm={grad_norm:.6f})")
    else:
        print(f"{name:25s}: ✗ NO GRADIENT")
        all_ok = False

print(f"\nHSTU Backbone: {'✓ HAS GRADIENTS' if hstu_has_grad else '✗ NO GRADIENTS (CRITICAL!)'}")

print("\n" + "=" * 60)
if all_ok and hstu_has_grad:
    print("✅ ALL COMPONENTS HAVE GRADIENTS")
    print("The fbgemm warning is harmless - gradients are flowing correctly.")
elif not hstu_has_grad:
    print("❌ HSTU BACKBONE HAS NO GRADIENTS!")
    print("This is the problem - gradients are blocked at the backbone.")
    print("\nPOSSIBLE SOLUTIONS:")
    print("1. The fbgemm operations need .detach() to be removed")
    print("2. Use requires_grad=True on HSTU inputs")
    print("3. Switch to a different HSTU implementation")
else:
    print("⚠️ SOME COMPONENTS MISSING GRADIENTS")
    print("Check the components marked with ✗ above.")

print("=" * 60)
