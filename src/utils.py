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
    """
    在分布式评估时，将所有 GPU 的结果收集到一起。
    """
    if not dist.is_initialized():
        return tensor
    
    # 收集到所有 GPU，或者只收集到 Rank 0
    # 这里使用 all_gather 简单处理
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)

def compute_gr_metrics(pred_indices, target_indices, k=10):
    """
    计算 HitRate@K 和 NDCG@K
    pred_indices: (B, K)
    target_indices: (B,)
    """
    # 转 numpy
    if isinstance(pred_indices, torch.Tensor):
        pred_indices = pred_indices.cpu().numpy()
    if isinstance(target_indices, torch.Tensor):
        target_indices = target_indices.cpu().numpy()

    hits = []
    ndcgs = []
    
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
    计算 AUC 和 LogLoss
    logits: (N,) 模型输出的原始分数
    labels: (N,) 0/1 标签
    """
    if isinstance(logits, torch.Tensor):
        # Apply Sigmoid for probabilities
        probs = torch.sigmoid(logits).cpu().numpy()
    else:
        probs = 1.0 / (1.0 + np.exp(-logits))
        
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    try:
        auc = roc_auc_score(labels, probs)
        ll = log_loss(labels, probs)
    except ValueError:
        # 防止只有一个类别导致报错
        auc, ll = 0.5, 0.0
        
    return auc, ll
