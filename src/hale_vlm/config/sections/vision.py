from typing import Literal, Optional

from hale_core.config.sections.common import StrictModel
from pydantic import Field

#: ``projector_type`` values that compress vision tokens into a fixed number of learned tokens.
TOKEN_CONNECTOR_TYPES = ("qformer", "gated_cross_attention")

#: ``encoder`` values that use HuggingFace pretrained models (online/cached download).
HF_ENCODER_TYPES = ("siglip", "clip")

#: ``encoder`` value that uses a hale_core native backbone (no HuggingFace required).
LOCAL_ENCODER_TYPE = "local"


class QFormerConfig(StrictModel):
    """Q-Former settings, used when ``projector_type: qformer``."""

    num_queries: int = 32
    hidden_dim: int | None = None  # None -> min(llm_dim, 768)
    num_layers: int = 2
    num_heads: int = 8
    cross_attention_freq: int = 1  # cross-attend to vision in every N-th layer (BLIP-2 uses 2)
    ffn_mult: int = 4
    dropout: float = 0.0


class GatedCrossAttentionConfig(StrictModel):
    """Gated cross-attention settings, used when ``projector_type: gated_cross_attention``."""

    num_latents: int = 64
    hidden_dim: int | None = None  # None -> min(llm_dim, 768)
    num_layers: int = 2
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    gate_init: float = 1.0  # 0.0 = Flamingo-style identity start; >0 lets images flow at step 0


class LocalBackboneConfig(StrictModel):
    """Settings for a hale_core native backbone (encoder: local).

    Use this when you want full control over backbone architecture and weights
    without the HuggingFace transformers dependency.

    Available backbone names (see hale_core.nn.vision.list_vision_backbones()):
      CNN family:     resnet50, convnext_base, mobile_one_s2, efficientnet_v2_l, etc.
      ViT family:     vit_large, dino_vit_giant, eva_02_large, deit3_huge, etc.
      Swin family:    swinv2_tiny, swinv2_base, swinv2_large, swinv2_giant
      Hybrid family:  coatnet_2, maxvit_base, maxvit_xlarge
      DCN family:     resnet101_dcn, internimage_large

    Example YAML::

        vision:
          encoder: local
          image_size: 336
          local_backbone:
            backbone_name: vit_large
            freeze_encoder: false
            freeze_up_to_stage: 12     # freeze first 12 blocks, fine-tune later ones
            pretrained_path: null
    """

    backbone_name:       str          = "convnext_base"
    drop_rate:           float        = 0.0
    freeze_encoder:      bool         = True
    freeze_up_to_stage:  int          = -1
    pretrained_path:     Optional[str] = None
    return_patch_tokens: bool         = True   # include patch tokens for cross-attn


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
    freeze_encoder: bool = True                        # top-level freeze (siglip/clip)
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
        """Build and return the configured vision tower.

        For encoder in ("siglip", "clip") — returns the HuggingFace VisionTower.
        For encoder == "local"            — returns LocalVisionTower.
        """
        if self.encoder == "local":
            from hale_vlm.vision.local_encoders import build_local_vision_tower
            lb = self.local_backbone
            return build_local_vision_tower(
                backbone_name=lb.backbone_name,
                img_size=self.image_size,
                drop_rate=lb.drop_rate,
                freeze_encoder=lb.freeze_encoder,
                freeze_up_to_stage=lb.freeze_up_to_stage,
                pretrained_path=lb.pretrained_path,
                return_patch_tokens=lb.return_patch_tokens,
            )
        # HuggingFace encoder (siglip / clip)
        from hale_vlm.vision.encoders import build_vision_tower
        return build_vision_tower(self)
