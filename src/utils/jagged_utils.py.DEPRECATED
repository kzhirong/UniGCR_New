"""
Utilities for converting between padded batches and jagged tensors.

Jagged tensors are used by HSTUTransducer for efficient variable-length sequence processing.
Instead of padding all sequences to max_len, jagged format concatenates all valid tokens.

Example:
    Padded format:
        Seq 1: [tok1, tok2, tok3, PAD, PAD]  length=3
        Seq 2: [tok4, tok5, PAD, PAD, PAD]   length=2
        Shape: (2, 5, D)

    Jagged format:
        [tok1, tok2, tok3, tok4, tok5]
        Shape: (5, D)
        Offsets: [0, 3, 5]  # cumsum([0, 3, 2])
"""

import torch
from typing import Tuple


def padded_to_jagged(
    padded_tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert padded batch tensor to jagged tensor format.

    Args:
        padded_tensor: Padded batch of shape (B, N, D) where:
            - B: batch size
            - N: max sequence length (with padding)
            - D: embedding dimension
        lengths: Actual sequence lengths, shape (B,)

    Returns:
        jagged_tensor: Concatenated valid tokens, shape (total_len, D)
            where total_len = sum(lengths)
        offsets: Cumulative offsets for each sequence, shape (B+1,)
            Computed as cumsum([0] + lengths)

    Example:
        >>> padded = torch.randn(2, 5, 64)
        >>> lengths = torch.tensor([3, 2])
        >>> jagged, offsets = padded_to_jagged(padded, lengths)
        >>> jagged.shape
        torch.Size([5, 64])  # 3 + 2 = 5
        >>> offsets
        tensor([0, 3, 5])
    """
    batch_size = padded_tensor.shape[0]
    device = padded_tensor.device
    dtype = padded_tensor.dtype

    # Collect valid (non-padded) tokens from each sequence
    jagged_list = []
    for i in range(batch_size):
        valid_len = lengths[i].item()
        jagged_list.append(padded_tensor[i, :valid_len, :])

    # Concatenate all valid tokens
    jagged_tensor = torch.cat(jagged_list, dim=0)  # (total_len, D)

    # Compute offsets using fbgemm if available, else manual cumsum
    try:
        # Use fbgemm for efficiency (same as HSTUTransducer uses)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    except (AttributeError, RuntimeError):
        # Fallback: manual cumsum
        # Prepend 0, then cumsum: [0, len[0], len[0]+len[1], ...]
        offsets = torch.cat([
            torch.zeros(1, dtype=lengths.dtype, device=device),
            torch.cumsum(lengths, dim=0)
        ])

    return jagged_tensor, offsets


def jagged_to_padded(
    jagged_tensor: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """
    Convert jagged tensor back to padded batch format.

    Args:
        jagged_tensor: Concatenated tokens, shape (total_len, D)
        lengths: Sequence lengths for each sample, shape (B,)
        max_len: Maximum sequence length for padding

    Returns:
        padded_tensor: Padded batch, shape (B, max_len, D)
            Positions beyond each sequence's length are filled with zeros

    Example:
        >>> jagged = torch.randn(5, 64)
        >>> lengths = torch.tensor([3, 2])
        >>> padded = jagged_to_padded(jagged, lengths, max_len=5)
        >>> padded.shape
        torch.Size([2, 5, 64])
        >>> # padded[0, :3, :] contains first 3 tokens
        >>> # padded[0, 3:, :] is all zeros (padding)
        >>> # padded[1, :2, :] contains next 2 tokens
        >>> # padded[1, 2:, :] is all zeros (padding)
    """
    batch_size = lengths.shape[0]
    embed_dim = jagged_tensor.shape[1]
    device = jagged_tensor.device
    dtype = jagged_tensor.dtype

    # Initialize padded tensor with zeros
    padded = torch.zeros(
        batch_size, max_len, embed_dim,
        device=device,
        dtype=dtype
    )

    # Fill in valid positions from jagged tensor
    offset = 0
    for i in range(batch_size):
        length = lengths[i].item()
        padded[i, :length, :] = jagged_tensor[offset:offset+length, :]
        offset += length

    return padded


def compute_jagged_params(
    lengths: torch.Tensor,
    num_targets: torch.Tensor,
) -> dict:
    """
    Compute parameters needed for HSTUTransducer forward pass.

    Args:
        lengths: Total sequence length per sample (history + targets), shape (B,)
        num_targets: Number of target tokens per sample, shape (B,)

    Returns:
        Dictionary with:
            - max_uih_len: Maximum history length across batch (int)
            - max_targets: Maximum target tokens across batch (int)
            - total_uih_len: Total history tokens across batch (int)
            - total_targets: Total target tokens across batch (int)

    Example:
        >>> lengths = torch.tensor([15, 12, 9])
        >>> num_targets = torch.tensor([3, 3, 3])
        >>> params = compute_jagged_params(lengths, num_targets)
        >>> params['max_uih_len']
        12  # max(15-3, 12-3, 9-3) = max(12, 9, 6)
        >>> params['total_uih_len']
        27  # (15-3) + (12-3) + (9-3) = 12+9+6
        >>> params['total_targets']
        9  # 3 + 3 + 3
    """
    # History lengths = total_length - num_targets
    history_lengths = lengths - num_targets

    return {
        'max_uih_len': history_lengths.max().item(),
        'max_targets': num_targets.max().item(),
        'total_uih_len': history_lengths.sum().item(),
        'total_targets': num_targets.sum().item(),
    }
