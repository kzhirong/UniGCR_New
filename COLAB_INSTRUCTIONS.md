# Running UniGCR with 4-Layer Semantic IDs in Colab

## Quick Start (Copy-Paste These Cells)

### Cell 1: Setup Environment
```python
# Clone repo and install dependencies
!git clone -b Zhirong https://github.com/kzhirong/UniGCR_New.git
%cd UniGCR_New

# Install PyTorch dependencies
!pip install torch numpy pandas scikit-learn tqdm --quiet

# Install generative_recommenders
!pip install git+https://github.com/facebookresearch/generative-recommenders.git --quiet

print("✅ Setup complete!")
```

### Cell 2: Patch fbgemm Compatibility Issue
```python
# Patch for missing asynchronous_complete_cumsum
import torch

# Add fallback implementation if fbgemm operation is missing
if not hasattr(torch.ops.fbgemm, 'asynchronous_complete_cumsum'):
    print("⚠️  fbgemm.asynchronous_complete_cumsum not found, adding fallback...")

    # Register a custom operation as fallback
    def async_cumsum_fallback(lengths: torch.Tensor) -> torch.Tensor:
        """Fallback implementation using torch.cumsum"""
        # Prepend 0 and compute cumsum
        return torch.cat([torch.zeros(1, dtype=lengths.dtype, device=lengths.device),
                         torch.cumsum(lengths, dim=0)])

    # Monkey patch it
    torch.ops.fbgemm.asynchronous_complete_cumsum = async_cumsum_fallback
    print("✅ Fallback registered!")
else:
    print("✅ fbgemm.asynchronous_complete_cumsum available!")
```

### Cell 3: Test Model with 4-Layer Semantic IDs
```python
import torch
from src.config import UniGCRConfig
from src.model import UniGCRModel

print("=" * 60)
print("Testing 4-Layer Semantic ID Model")
print("=" * 60)

# Initialize
config = UniGCRConfig(
    use_semantic_seq=True,
    use_atomic_seq=False,
    use_cat_profile=False,
    use_num_profile=False,
)
model = UniGCRModel(config).cuda()
print("✅ Model initialized")

# Create test batch with PROPER vocab sizes
batch_size = 4
num_items = 5
num_tokens = num_items * 4  # 20 tokens

# Generate tokens with correct vocab ranges
sem_history = torch.zeros(batch_size, num_tokens, dtype=torch.long, device='cuda')
for i in range(num_items):
    # L0, L1, L2: vocab 256 (values 1-255)
    sem_history[:, i*4 + 0] = torch.randint(1, 256, (batch_size,), device='cuda')
    sem_history[:, i*4 + 1] = torch.randint(1, 256, (batch_size,), device='cuda')
    sem_history[:, i*4 + 2] = torch.randint(1, 256, (batch_size,), device='cuda')
    # Dedup: vocab 19 (values 0-18)
    sem_history[:, i*4 + 3] = torch.randint(0, 19, (batch_size,), device='cuda')

batch = {
    'sem_history': sem_history,
    'lengths': torch.tensor([20, 16, 12, 20], device='cuda')
}

# Forward pass
with torch.no_grad():
    u, logits_seq, _ = model(batch)

# Display results
print(f"\n✅ Forward pass successful!")
print(f"   User repr: {u.shape}")
print(f"   Logits: {len(logits_seq)} tensors")
for i, logits in enumerate(logits_seq):
    layer_name = ["L0", "L1", "L2", "Dedup"][i]
    print(f"      {layer_name}: {logits.shape}")

# Verify shapes
assert len(logits_seq) == 4
assert logits_seq[0].shape == (batch_size, num_items, 256)
assert logits_seq[1].shape == (batch_size, num_items, 256)
assert logits_seq[2].shape == (batch_size, num_items, 256)
assert logits_seq[3].shape == (batch_size, num_items, 19)

print("\n" + "=" * 60)
print("🎉 All tests passed! Ready for training.")
print("=" * 60)
```

### Cell 4: Test Loss Calculation
```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# Create targets with proper vocab sizes
sem_target = torch.zeros(batch_size, num_tokens, dtype=torch.long, device='cuda')
for i in range(num_items):
    sem_target[:, i*4 + 0] = torch.randint(1, 256, (batch_size,), device='cuda')
    sem_target[:, i*4 + 1] = torch.randint(1, 256, (batch_size,), device='cuda')
    sem_target[:, i*4 + 2] = torch.randint(1, 256, (batch_size,), device='cuda')
    sem_target[:, i*4 + 3] = torch.randint(0, 19, (batch_size,), device='cuda')

# Reshape targets to (B, num_items, 4)
sem_target = sem_target.view(batch_size, num_items, 4)

# Compute loss per layer
total_loss = 0.0
print("\nLoss per layer:")
for layer_idx, logits_layer in enumerate(logits_seq):
    layer_name = ["L0", "L1", "L2", "Dedup"][layer_idx]
    targets_layer = sem_target[:, :, layer_idx]

    logits_flat = logits_layer.reshape(-1, logits_layer.size(-1))
    targets_flat = targets_layer.reshape(-1)

    loss = criterion(logits_flat, targets_flat)
    total_loss += loss
    print(f"  {layer_name}: {loss.item():.4f}")

avg_loss = total_loss / 4
print(f"\n✅ Average loss: {avg_loss.item():.4f}")

# Test backward pass
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()
avg_loss.backward()
optimizer.step()

print("✅ Backward pass successful!")
print("✅ Ready for full training!")
```

---

## After Tests Pass

Run your training script:
```python
# If you have run.py:
!python run.py --dataset beauty --use_semantic_seq --epochs 10 --batch_size 64

# Or use your custom training code
```

---

## Key Points

1. **4 Semantic Layers:** [L0, L1, L2, Dedup] not 3
2. **Different Vocab Sizes:**
   - L0, L1, L2: 256 values (1-255)
   - Dedup: 19 values (0-18)
3. **Loss Calculation:** Computed per layer and averaged
4. **Model Output:** List of 4 tensors (not stacked)

---

## Troubleshooting

### If you see "CUDA device-side assert":
- Make sure Dedup tokens are in range 0-18 (not 0-255)

### If you see "asynchronous_complete_cumsum not found":
- Run Cell 2 (the patch) before testing

### If model loading fails:
- Check that you pulled latest code with `item_embedding_dim` property fix
