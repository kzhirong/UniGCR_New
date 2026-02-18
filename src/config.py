from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class UniGCRConfig:
    # Task switches
    enable_ctr: bool = False
    use_semantic_seq: bool = True
    use_atomic_seq: bool = False
    use_cat_profile: bool = False
    use_num_profile: bool = False

    # CTR attention modules
    ctr_use_self_attn: bool = True
    ctr_use_cross_attn: bool = True

    # Semantic ID / GRID
    # 4 layers: L0, L1, L2 (K-Quantization, vocab=256 each) + Dedup (collision resolution, vocab=19)
    sem_id_layers: int = 4
    sem_id_codebook_size: int = 256
    sem_id_dedup_size: int = 19
    grid_mapping_path: str = "data/semantic_id_kmean.pt"

    # Atomic ID (optional auxiliary input)
    num_atomic_items: int = 0
    max_atomic_len: int = 50

    # User profile (optional)
    cat_feature_vocab_sizes: List[int] = field(default_factory=lambda: [1000, 20])
    num_feature_size: int = 5

    # Model architecture
    embed_dim: int = 256
    # max_seq_len must satisfy: (max_seq_len - 1) % sem_id_layers == 0
    # 153 - 1 = 152 = 38 items × 4 tokens
    max_seq_len: int = 153
    hstu_layers: int = 4
    hstu_heads: int = 4
    dropout: float = 0.1
    attn_alpha: float = 1.0

    # Research HSTU (Meta ICML'24) parameters
    hstu_normalization: str = "rel_bias"
    hstu_linear_config: str = "uvqk"
    hstu_linear_activation: str = "silu"
    hstu_enable_rel_bias: bool = True

    # Training
    patience: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # CTR loss weights
    memory_bank_size: int = 20
    num_hard_negatives: int = 5
    temp: float = 0.07
    loss_alpha: float = 1.0
    loss_beta: float = 0.5

    # Runtime (filled automatically by AmazonBeautyDataset)
    sem_total_vocab: int = 0

    @property
    def num_semantic_tokens_per_item(self) -> int:
        return self.sem_id_layers
