"""Registries of vision towers and vision-to-LLM projectors (selected by name in the config).

``vision.encoder`` picks a :data:`VISION_TOWERS` entry and ``vision.projector_type`` picks a
:data:`PROJECTORS` entry. New encoders or connectors are added by registering a builder, never by
editing an ``if``/``elif`` chain (open/closed).

Builder signatures:
    * tower: ``(cfg: VisionConfig) -> nn.Module`` exposing ``hidden_size``;
    * projector: ``(vision_dim: int, llm_dim: int, cfg: VisionConfig) -> nn.Module``.
"""

from __future__ import annotations

from hale_core.registry import NamedRegistry

VISION_TOWERS = NamedRegistry("vision_tower", scope="hale_vlm")
PROJECTORS = NamedRegistry("vision_projector", scope="hale_vlm")

__all__ = ["PROJECTORS", "VISION_TOWERS"]
