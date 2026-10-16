"""QFormerConfig: Q-Former settings, used when ``projector_type: qformer``."""

from hale_core.config.sections.common import StrictModel


class QFormerConfig(StrictModel):
    """Q-Former settings, used when ``projector_type: qformer``."""

    num_queries: int = 32
    hidden_dim: int | None = None  # None -> min(llm_dim, 768)
    num_layers: int = 2
    num_heads: int = 8
    cross_attention_freq: int = 1  # cross-attend to vision in every N-th layer (BLIP-2 uses 2)
    ffn_mult: int = 4
    dropout: float = 0.0
