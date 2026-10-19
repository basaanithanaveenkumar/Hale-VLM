"""Builders of the vision-to-LLM token connectors, registered in ``PROJECTORS``."""

from __future__ import annotations

import torch.nn as nn

from hale_vlm.config.sections.vision import TOKEN_CONNECTOR_TYPES, VisionConfig
from hale_vlm.exceptions import ModelConfigError
from hale_vlm.vision.connectors.gated_cross_attention import GatedCrossAttentionProjector
from hale_vlm.vision.connectors.qformer import QFormer
from hale_vlm.vision.vision_registries import PROJECTORS


@PROJECTORS.register_module(name="qformer")
def build_qformer(vision_dim: int, llm_dim: int, cfg: VisionConfig) -> QFormer:
    """BLIP-2 style learned-query connector from ``cfg.qformer``."""
    q = cfg.qformer
    return QFormer(
        vision_dim,
        llm_dim,
        num_queries=q.num_queries,
        hidden_dim=q.hidden_dim,
        num_layers=q.num_layers,
        num_heads=q.num_heads,
        cross_attention_freq=q.cross_attention_freq,
        ffn_mult=q.ffn_mult,
        dropout=q.dropout,
    )


@PROJECTORS.register_module(name="gated_cross_attention")
def build_gated_cross_attention(
    vision_dim: int, llm_dim: int, cfg: VisionConfig
) -> GatedCrossAttentionProjector:
    """Flamingo style tanh-gated cross-attention connector from ``cfg.gated_cross_attention``."""
    g = cfg.gated_cross_attention
    return GatedCrossAttentionProjector(
        vision_dim,
        llm_dim,
        num_latents=g.num_latents,
        hidden_dim=g.hidden_dim,
        num_layers=g.num_layers,
        num_heads=g.num_heads,
        ffn_mult=g.ffn_mult,
        dropout=g.dropout,
        gate_init=g.gate_init,
    )


def build_vision_connector(vision_dim: int, llm_dim: int, cfg: VisionConfig) -> nn.Module:
    """Return the token connector selected by ``cfg.projector_type``.

    Raises:
        ModelConfigError: if ``projector_type`` is not a token connector.
    """
    if cfg.projector_type not in TOKEN_CONNECTOR_TYPES:
        raise ModelConfigError(
            f"projector_type={cfg.projector_type!r} is not a token connector; "
            f"expected one of {TOKEN_CONNECTOR_TYPES}"
        )
    return PROJECTORS.get(cfg.projector_type)(vision_dim, llm_dim, cfg)
