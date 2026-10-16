""":class:`GatedCrossAttentionConfig`."""

from hale_core.config.sections.common import StrictModel


class GatedCrossAttentionConfig(StrictModel):
    """Gated cross-attention settings, used when ``projector_type: gated_cross_attention``."""

    num_latents: int = 64
    hidden_dim: int | None = None  # None -> min(llm_dim, 768)
    num_layers: int = 2
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    gate_init: float = 1.0
