# DeepSpeed Configuration Files

This directory contains different DeepSpeed configurations for various training scenarios.

## Available Configs

### 1. `deepspeed_single_gpu.json` - **Single GPU (Recommended for Colab)**
**Use when:** Training on 1 GPU (like Colab T4/V100/A100)

**Features:**
- ✅ FP16 mixed precision (faster training)
- ✅ AdamW optimizer with warmup
- ✅ No ZeRO optimizer (not needed for single GPU)
- ✅ Minimal overhead, maximum speed

**Command:**
```bash
python run.py --deepspeed_config configs/deepspeed_single_gpu.json
```

---

### 2. `deepspeed_config.json` - **Multi-GPU (ZeRO Stage 2)**
**Use when:** Training on 2-8 GPUs in a cluster

**Features:**
- ✅ FP16 mixed precision
- ✅ ZeRO Stage 2 (optimizer state partitioning)
- ✅ Efficient multi-GPU training
- ✅ Good for models that fit in GPU memory

**Command:**
```bash
deepspeed --num_gpus=4 run.py --deepspeed_config configs/deepspeed_config.json
```

---

### 3. `deepspeed_zero3.json` - **Multi-GPU (ZeRO Stage 3)**
**Use when:** Training HUGE models (billions of parameters) that don't fit in GPU memory

**Features:**
- ✅ FP16 mixed precision
- ✅ ZeRO Stage 3 (full parameter partitioning)
- ✅ Offload optimizer + parameters to CPU
- ✅ Can train models larger than GPU memory
- ⚠️ Slower than Stage 2 due to CPU offloading

**Command:**
```bash
deepspeed --num_gpus=8 run.py --deepspeed_config configs/deepspeed_zero3.json
```

---

## Configuration Breakdown

### Key Parameters

| Parameter | What it does |
|-----------|-------------|
| `"train_batch_size": "auto"` | Uses batch size from your args |
| `"lr": "auto"` | Uses learning rate from your args (default: 1e-4) |
| `"fp16": {"enabled": true}` | Uses half-precision for speed |
| `"zero_optimization": {"stage": 2}` | Partitions optimizer across GPUs |
| `"gradient_clipping": 1.0` | Clips gradients to prevent explosion |
| `"warmup_num_steps": 500` | Warmup learning rate for 500 steps |

### ZeRO Stages Explained

**Stage 0:** No optimization (regular distributed training)
**Stage 1:** Partition optimizer states
**Stage 2:** Partition optimizer + gradients (most common)
**Stage 3:** Partition optimizer + gradients + parameters (huge models)

---

## For Your Current Model (187k params)

**Recommendation:** You don't need DeepSpeed for this small model!

✅ **Best option:** Just use regular PyTorch
```bash
python run.py  # No DeepSpeed flag
```

🔶 **If you want to try DeepSpeed anyway:**
```bash
python run.py --deepspeed_config configs/deepspeed_single_gpu.json
```

DeepSpeed benefits become significant for:
- Models with 100M+ parameters
- Multi-GPU training
- Models that don't fit in GPU memory

---

## Customizing Configs

### Change Learning Rate
```json
"optimizer": {
  "params": {
    "lr": 5e-4  // Change from "auto" to fixed value
  }
}
```

### Disable FP16 (if GPU doesn't support it)
```json
"fp16": {
  "enabled": false
}
```

### Gradient Accumulation (simulate larger batch)
```json
"gradient_accumulation_steps": 4  // 4x larger effective batch
```

---

## Troubleshooting

**Error: "CUDA out of memory"**
→ Reduce `train_micro_batch_size_per_gpu` or use ZeRO Stage 3

**Error: "mpi4py not found"**
→ Install: `pip install mpi4py`

**Slow training with ZeRO-3**
→ Use ZeRO-2 instead (or regular PyTorch for small models)

---

## References

- [DeepSpeed Configuration Guide](https://www.deepspeed.ai/docs/config-json/)
- [ZeRO Optimization](https://www.deepspeed.ai/tutorials/zero/)
