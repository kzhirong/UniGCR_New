"""
Builder functions for creating Research HSTU and its components.

This module creates all the necessary components for using Research HSTU
from generative_recommenders for autoregressive generative retrieval.

Note: We use Option B architecture where UniGCR's InputLayer handles
the 3-token-per-item semantic ID structure, and HSTU receives pre-computed
item-level embeddings.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .config import UniGCRConfig

# IMPORTANT: Import fbgemm_gpu BEFORE using torch.ops.fbgemm
# This ensures the custom ops are registered
try:
    import fbgemm_gpu
    # Verify critical ops are available
    torch.ops.fbgemm.asynchronous_complete_cumsum
    torch.ops.fbgemm.dense_to_jagged
    torch.ops.fbgemm.jagged_to_padded_dense
except (ImportError, AttributeError) as e:
    print(f"[WARNING] fbgemm_gpu operations not available: {e}")
    print("Research HSTU requires fbgemm_gpu. Install with:")
    print("  pip install fbgemm-gpu==1.1.0 --index-url https://download.pytorch.org/whl/cu124")

# Import from generative_recommenders - Research HSTU
try:
    from generative_recommenders.research.modeling.sequential.hstu import HSTU
    from generative_recommenders.research.modeling.sequential.embedding_modules import (
        EmbeddingModule,
    )
    from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
        InputFeaturesPreprocessorModule,
        LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    )
    from generative_recommenders.research.modeling.sequential.output_postprocessors import (
        OutputPostprocessorModule,
        L2NormEmbeddingPostprocessor,
    )
    from generative_recommenders.research.rails.similarities.module import SimilarityModule
    from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (
        DotProductSimilarity,
    )
    GENERATIVE_RECOMMENDERS_AVAILABLE = True
except ImportError:
    GENERATIVE_RECOMMENDERS_AVAILABLE = False
    HSTU = None
    EmbeddingModule = object
    InputFeaturesPreprocessorModule = object
    OutputPostprocessorModule = object
    SimilarityModule = object


class PassthroughEmbeddingModule(nn.Module):
    """
    Passthrough embedding module for Research HSTU.

    UniGCR's InputLayer already handles semantic token embedding (4 tokens per item).
    This module satisfies HSTU's API requirement but doesn't do actual embedding lookup.

    Architecture rationale:
    - UniGCR uses RQ-VAE semantic IDs: 4 tokens per item [L0, L1, L2, Dedup]
    - HSTU expects: 1 ID per item
    - Solution: InputLayer converts 4-token → 1-embedding, HSTU uses pre-computed embeddings
    """

    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Dimension of item embeddings (must match InputLayer output)
        """
        super().__init__()
        self._embedding_dim = embedding_dim

    @property
    def item_embedding_dim(self) -> int:
        """Property required by Research HSTU."""
        return self._embedding_dim

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        HSTU's API requires this method, but we won't use it.
        Our embeddings are pre-computed by UniGCR's InputLayer.

        Returns zero embeddings (will be overridden by past_embeddings parameter).

        Args:
            item_ids: (B, N) item IDs

        Returns:
            Zero embeddings of shape (B, N, D)
        """
        batch_size, seq_len = item_ids.shape
        return torch.zeros(
            batch_size, seq_len, self._embedding_dim,
            dtype=torch.float32,
            device=item_ids.device
        )

    def get_item_embedding_dim(self) -> int:
        """Return embedding dimension (backward compatibility)."""
        return self._embedding_dim


def build_embedding_module(config: UniGCRConfig) -> PassthroughEmbeddingModule:
    """
    Build passthrough embedding module.

    NOTE: Actual embeddings are computed by UniGCR's InputLayer which handles
    the 3-token-per-item semantic ID format from RQ-VAE.

    Args:
        config: UniGCR configuration

    Returns:
        PassthroughEmbeddingModule instance
    """
    return PassthroughEmbeddingModule(embedding_dim=config.embed_dim)


def build_similarity_module() -> SimilarityModule:
    """
    Build similarity module for Research HSTU.

    Used by HSTU for computing similarity scores between user and item embeddings.
    Dot product similarity is standard for recommendation tasks.

    Returns:
        DotProductSimilarity instance
    """
    if not GENERATIVE_RECOMMENDERS_AVAILABLE:
        raise ImportError("generative_recommenders not available")

    return DotProductSimilarity()


def build_input_preprocessor(config: UniGCRConfig) -> InputFeaturesPreprocessorModule:
    """
    Build input features preprocessor for Research HSTU.

    Adds learnable positional embeddings to item embeddings and applies dropout.
    This is similar to transformer positional encodings but learned during training.

    Args:
        config: UniGCR configuration

    Returns:
        LearnablePositionalEmbeddingInputFeaturesPreprocessor instance
    """
    if not GENERATIVE_RECOMMENDERS_AVAILABLE:
        raise ImportError("generative_recommenders not available")

    return LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=config.max_seq_len,
        embedding_dim=config.embed_dim,
        dropout_rate=config.dropout,
    )


def build_output_postprocessor(config: UniGCRConfig) -> OutputPostprocessorModule:
    """
    Build output postprocessor for Research HSTU.

    Applies L2 normalization to output embeddings for better similarity computation.
    This is standard practice in recommendation systems using dot product similarity.

    Args:
        config: UniGCR configuration

    Returns:
        L2NormEmbeddingPostprocessor instance
    """
    if not GENERATIVE_RECOMMENDERS_AVAILABLE:
        raise ImportError("generative_recommenders not available")

    return L2NormEmbeddingPostprocessor(
        embedding_dim=config.embed_dim,
        eps=1e-6,
    )


def build_research_hstu(config: UniGCRConfig) -> HSTU:
    """
    Build Research HSTU model for autoregressive next-item prediction.

    Architecture:
        Input: Pre-computed item embeddings from UniGCR's InputLayer (B, N, D)
        Processing: HSTU layers with causal self-attention
        Output: Contextualized item embeddings (B, N, D) for all positions

    Configuration follows Meta's ICML'24 paper recommendations:
    - Base model: 2 blocks, 1 head, dqk=dv=embedding_dim
    - Large model: 8 blocks, 2 heads, dqk=dv=embedding_dim//2

    Reference:
        Paper: "Actions Speak Louder than Words: Trillion-Parameter Sequential
               Transducers for Generative Recommendations" (ICML'24)
        Repo: https://github.com/meta-recsys/generative-recommenders
        File: generative_recommenders/research/modeling/sequential/hstu.py

    Args:
        config: UniGCR configuration

    Returns:
        HSTU instance ready for training
    """
    if not GENERATIVE_RECOMMENDERS_AVAILABLE:
        raise ImportError(
            "generative_recommenders not available. "
            "Please install: pip install generative-recommenders"
        )

    # Build required modules
    embedding_module = build_embedding_module(config)
    similarity_module = build_similarity_module()
    input_preproc = build_input_preprocessor(config)
    output_postproc = build_output_postprocessor(config)

    # Determine attention and linear dimensions
    # For base models, typically dqk = dv = embedding_dim
    # For large models, can reduce to embedding_dim // 2
    attention_dim = config.embed_dim  # dqk
    linear_dim = config.embed_dim      # dv

    # Create HSTU
    hstu = HSTU(
        # Sequence configuration
        max_sequence_len=config.max_seq_len,
        max_output_len=config.max_seq_len,
        embedding_dim=config.embed_dim,

        # Architecture configuration
        num_blocks=config.hstu_layers,      # Number of STU layers (2 for base, 8 for large)
        num_heads=config.hstu_heads,         # Number of attention heads (1 for base, 2 for large)
        linear_dim=linear_dim,               # Dimension of linear/value projections (dv)
        attention_dim=attention_dim,         # Dimension of query/key projections (dqk)

        # Layer configuration
        normalization="rel_bias",            # Relative bias normalization
        linear_config="uvqk",                # Linear layer config: u, v, q, k projections
        linear_activation="silu",            # SiLU (Swish) activation

        # Dropout
        linear_dropout_rate=config.dropout,
        attn_dropout_rate=config.dropout,

        # Required modules
        embedding_module=embedding_module,
        similarity_module=similarity_module,
        input_features_preproc_module=input_preproc,
        output_postproc_module=output_postproc,

        # Optional features
        enable_relative_attention_bias=True,  # Use temporal/positional bias
        concat_ua=False,                      # No user attributes concatenation
        verbose=False,                        # Disable verbose logging
    )

    return hstu


# For backward compatibility / alternative builder interface
def build_hstu_transducer(config: UniGCRConfig) -> HSTU:
    """
    Legacy function name for backward compatibility.
    Now builds Research HSTU instead of HSTUTransducer.

    Args:
        config: UniGCR configuration

    Returns:
        HSTU instance (Research HSTU, not HSTUTransducer)
    """
    return build_research_hstu(config)
