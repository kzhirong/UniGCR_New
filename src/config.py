from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class UniGCRConfig:
    # --- [任务开关] ---
    enable_ctr: bool = False          # 是否开启 CTR 联合训练 (Phase 1: GR-only)
    use_semantic_seq: bool = True     # 是否使用 GRID Semantic ID
    use_atomic_seq: bool = False      # 是否使用 Atomic ID
    use_cat_profile: bool = False     # 是否使用类别用户画像 (Phase 1: disabled)
    use_num_profile: bool = False     # 是否使用数值用户画像 (Phase 1: disabled)
    
    # --- [CTR 模块微调] ---
    ctr_use_self_attn: bool = True    # Candidate-Aware Self-Attention
    ctr_use_cross_attn: bool = True   # User-Centric Cross-Attention
    
    # --- [Semantic ID / GRID] ---
    # Note: num_layers and codebook_sizes are auto-detected from semantic_ids.json
    # These values are only fallback if auto_detect=False in GridMapper
    # 4 layers: [L0, L1, L2, Dedup]
    # - L0, L1, L2: RQ-VAE layers (256 codebook each)
    # - Dedup: Handles ID collisions when multiple items map to same [L0,L1,L2] (19 values: 0-18)
    sem_id_layers: int = 4  # 4 layers including deduplication
    sem_id_codebook_size: int = 256  # Codebook size for L0, L1, L2
    sem_id_dedup_size: int = 19  # Deduplication column vocabulary (0-18)
    grid_mapping_path: str = "data/beauty/semantic_ids.json"
    
    # --- [Atomic ID] ---
    num_atomic_items: int = 0
    max_atomic_len: int = 50
    
    # --- [User Profile] ---
    cat_feature_vocab_sizes: List[int] = field(default_factory=lambda: [1000, 20]) 
    num_feature_size: int = 5
    
    # --- [模型参数] ---
    embed_dim: int = 64
    max_seq_len: int = 153  # Must be: (max_seq_len - 1) % 4 == 0 for 4-layer semantic IDs
                             # 153 - 1 = 152, and 152 / 4 = 38 items ✓
    hstu_layers: int = 2
    hstu_heads: int = 2
    dropout: float = 0.1
    attn_alpha: float = 1.0

    # --- [Research HSTU Advanced Parameters (Optional)] ---
    # These follow Meta's ICML'24 paper recommendations
    # Base model: num_blocks=2, num_heads=1, dqk=dv=embed_dim
    # Large model: num_blocks=8, num_heads=2, dqk=dv=embed_dim//2
    hstu_normalization: str = "rel_bias"       # Normalization strategy
    hstu_linear_config: str = "uvqk"            # Linear layer configuration
    hstu_linear_activation: str = "silu"        # SiLU (Swish) activation
    hstu_enable_rel_bias: bool = True          # Enable relative attention bias
    
    # --- [训练参数] ---
    patience: int = 30  # Increased to allow conservative scheduled sampling (epochs 1-30)
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # --- [Uni-GCR Loss] ---
    memory_bank_size: int = 20
    num_hard_negatives: int = 5
    temp: float = 0.07  
    loss_alpha: float = 1.0 
    loss_beta: float = 0.5 
    
    # --- [运行时动态填充] ---
    sem_total_vocab: int = 0

    @property
    def num_semantic_tokens_per_item(self) -> int:
        """Number of tokens per item (layers in semantic ID)."""
        return self.sem_id_layers
