from dataclasses import dataclass, field
from typing import List
import torch

try:
    from generative_recommenders.modeling.sequential.hstu import HSTUConfig
except ImportError:
    HSTUConfig = None

@dataclass
class UniGCRConfig:
    # --- [任务开关] ---
    enable_ctr: bool = True           # 是否开启 CTR 联合训练
    use_semantic_seq: bool = True     # 是否使用 GRID Semantic ID
    use_atomic_seq: bool = False      # 是否使用 Atomic ID
    use_cat_profile: bool = True      # 是否使用类别用户画像
    use_num_profile: bool = True      # 是否使用数值用户画像
    
    # --- [CTR 模块微调] ---
    ctr_use_self_attn: bool = True    # Candidate-Aware Self-Attention
    ctr_use_cross_attn: bool = True   # User-Centric Cross-Attention
    
    # --- [Semantic ID / GRID] ---
    sem_id_layers: int = 3
    sem_id_codebook_size: int = 256
    grid_mapping_path: str = "data/beauty/semantic_ids.json"
    
    # --- [Atomic ID] ---
    num_atomic_items: int = 0
    max_atomic_len: int = 50
    
    # --- [User Profile] ---
    cat_feature_vocab_sizes: List[int] = field(default_factory=lambda: [1000, 20]) 
    num_feature_size: int = 5
    
    # --- [模型参数] ---
    embed_dim: int = 64
    max_seq_len: int = 150  # 需足够容纳 (Sem_Len + Atom_Len + Profile_Len)
    hstu_layers: int = 2
    hstu_heads: int = 2
    dropout: float = 0.1
    attn_alpha: float = 1.0
    
    # --- [训练参数] ---
    patience: int = 3
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
    
    def to_hstu_config(self):
        if HSTUConfig is None:
            raise ImportError("generative_recommenders not installed.")
        return HSTUConfig(
            embedding_dim=self.embed_dim,
            num_heads=self.hstu_heads,
            num_blocks=self.hstu_layers,
            dropout_rate=self.dropout,
            linear_dropout_rate=0.0,
            attn_dropout_rate=0.0,
            forward_dropout_rate=self.dropout,
            normalization="layer_norm",
            activation="silu",
            max_seq_len=self.max_seq_len,
            attn_alpha=self.attn_alpha 
        )
