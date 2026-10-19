"""VisionConfig: Vision tower and projector settings."""

from typing import Literal

from hale_core.config.sections.common import StrictModel
from pydantic import Field

from .gated_cross_attention_config import GatedCrossAttentionConfig
from .local_backbone_config import LocalBackboneConfig
from .q_former_config import QFormerConfig


class VisionConfig(StrictModel):
    """Vision tower and projector settings.

    Encoder types:
      siglip  — SigLIP pretrained encoder (HuggingFace, default)
      clip    — CLIP pretrained encoder (HuggingFace)
      local   — hale_core native backbone (any registered architecture)

    When encoder="local", the ``local_backbone`` section controls the backbone.
    """

    encoder: Literal["siglip", "clip", "local"] = "siglip"
    model_id: str = "google/siglip-base-patch16-224"  # used only for siglip/clip
    image_size: int = 224
    freeze_encoder: bool = True  # top-level freeze (siglip/clip)
    projector_type: Literal["mlp", "linear", "qformer", "gated_cross_attention"] = "mlp"
    projector_hidden_dim: int | None = None
    projector_dropout: float = 0.0
    num_image_tokens: int = 256
    qformer: QFormerConfig = Field(default_factory=QFormerConfig)
    gated_cross_attention: GatedCrossAttentionConfig = Field(
        default_factory=GatedCrossAttentionConfig
    )
    local_backbone: LocalBackboneConfig = Field(default_factory=LocalBackboneConfig)

    def resolved_num_image_tokens(self) -> int:
        """Image tokens the LLM sees: the connector's own count, else ``num_image_tokens``."""
        if self.projector_type == "qformer":
            return self.qformer.num_queries
        if self.projector_type == "gated_cross_attention":
            return self.gated_cross_attention.num_latents
        return self.num_image_tokens

    def build_vision_tower(self):
        """Build the vision tower registered under :attr:`encoder` (see ``VISION_TOWERS``)."""
        from hale_vlm.vision.encoders import build_vision_tower

        return build_vision_tower(self)
