import torch
import torch.distributed as dist
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score, log_loss


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
        return int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode.")
        return 0


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gather_tensors(tensor):
    """Gather tensors from all ranks for distributed evaluation."""
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def compute_gr_metrics(pred_indices, target_indices, k=10):
    """
    Compute Hit@K and NDCG@K.

    Args:
        pred_indices: (B, K) predicted item IDs
        target_indices: (B,) ground truth item IDs
    Returns:
        (hit_rate, ndcg): scalar means over batch
    """
    if isinstance(pred_indices, torch.Tensor):
        pred_indices = pred_indices.cpu().numpy()
    if isinstance(target_indices, torch.Tensor):
        target_indices = target_indices.cpu().numpy()

    hits, ndcgs = [], []
    for i in range(len(target_indices)):
        target = target_indices[i]
        preds = pred_indices[i]
        hit_mask = (preds == target)
        if np.any(hit_mask):
            hits.append(1.0)
            rank = np.where(hit_mask)[0][0]
            ndcgs.append(1.0 / np.log2(rank + 2.0))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)

    return np.mean(hits), np.mean(ndcgs)


def compute_ctr_metrics(logits, labels):
    """
    Compute AUC and LogLoss for CTR evaluation.

    Args:
        logits: (N,) raw model scores
        labels: (N,) binary labels
    Returns:
        (auc, logloss)
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
    else:
        probs = 1.0 / (1.0 + np.exp(-logits))

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    try:
        auc = roc_auc_score(labels, probs)
        ll = log_loss(labels, probs)
    except ValueError:
        auc, ll = 0.5, 0.0

    return auc, ll
