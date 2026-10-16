"""LocalBackboneConfig: Settings for a hale_core native backbone (encoder: local)."""

from hale_core.config.sections.common import StrictModel


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

    backbone_name: str = "convnext_base"
    drop_rate: float = 0.0
    freeze_encoder: bool = True
    freeze_up_to_stage: int = -1
    pretrained_path: str | None = None
    return_patch_tokens: bool = True
