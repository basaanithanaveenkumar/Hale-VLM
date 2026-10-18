"""Build a vision-to-LLM token connector from :class:`VisionConfig`."""

from __future__ import annotations

from torch import nn

from hale_vlm.config.sections.vision import VisionConfig
from hale_vlm.exceptions import VLMConfigError
from hale_vlm.vision.connectors.gated_cross_attention import (
    GatedCrossAttentionProjector,  # noqa: F401
)
from hale_vlm.vision.connectors.qformer import QFormer  # noqa: F401  (import registers it)
from hale_vlm.vision.connectors.registry import VISION_CONNECTORS


def build_vision_connector(vision_dim: int, llm_dim: int, cfg: VisionConfig) -> nn.Module:
    """Build the connector registered for ``cfg.projector_type`` (Factory over a Registry).

    Adding a connector means registering a class with
    :func:`~hale_vlm.vision.connectors.registry.register_vision_connector`; this function does not
    change.

    Args:
        vision_dim: channel size of the vision encoder tokens.
        llm_dim: hidden size of the language model.
        cfg: vision config; ``projector_type`` selects the connector.

    Returns:
        The connector module mapping ``[B, N, vision_dim]`` to ``[B, T, llm_dim]``.

    Raises:
        VLMConfigError: if ``projector_type`` is not a registered token connector.
    """
    if cfg.projector_type not in VISION_CONNECTORS:
        raise VLMConfigError(
            f"projector_type={cfg.projector_type!r} is not a token connector; "
            f"expected one of {sorted(VISION_CONNECTORS.keys())}"
        )
    return VISION_CONNECTORS.get(cfg.projector_type).from_config(vision_dim, llm_dim, cfg)
