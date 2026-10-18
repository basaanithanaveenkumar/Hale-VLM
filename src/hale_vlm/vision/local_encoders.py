"""
LocalVisionTower — VisionTower backed by hale_core native backbones.

Why two VisionTower implementations?
--------------------------------------
The existing VisionTower (encoders.py) loads pretrained SigLIP/CLIP weights
via HuggingFace transformers, which is ideal when using publicly-released
vision encoders out of the box.

LocalVisionTower uses any backbone registered in hale_core.nn.vision — giving
full control over architecture, pretrained weights, and fine-tuning strategy.
Use it when:
  - You want a larger-scale backbone (ViT-22B, SwinV2-G, ConvNeXt-H, etc.)
  - You have custom pretrained weights (EVA-02 CLIP, DINOv2, etc.)
  - You want to fine-tune the vision encoder end-to-end (no HF dependency)
  - You are running on an environment without internet (local .pt weights)

Config integration:

    vision:
      encoder: local                         # use LocalVisionTower
      backbone_name: convnext_base           # any registered backbone
      image_size: 224
      drop_rate: 0.1
      freeze_encoder: false
      freeze_up_to_stage: 2                  # freeze first 3 stages
      pretrained_path: /path/to/weights.pt   # optional local checkpoint
      return_patch_tokens: true              # include patch-level tokens

Architecture:

    Input images (normalised)
         ↓
    hale_core backbone (ConvNeXt / SwinV2 / ViT / etc.)
         ↓  BackboneOutput
    pooled_output  [B, embed_dim]   → global image embedding → projector
    patch_tokens   [B, N, D]        → per-patch tokens for cross-attention
    feature_maps   [B, C_i, H_i, W_i] → multi-scale features (if needed)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LocalVisionTower(nn.Module):
    """Vision encoder backed by a hale_core registered backbone.

    Unlike the HuggingFace-based VisionTower, this class owns the backbone
    weights and can be fine-tuned within the VLM training loop.

    Args:
        backbone_name:
            Name registered in hale_core.nn.vision, e.g. "convnext_base",
            "vit_large", "swinv2_tiny", "dino_vit_giant".

        img_size:
            Input spatial resolution.  Standard: 224 (ImageNet), 336 (hi-res VLM).

        drop_rate:
            Dropout on backbone pooled output.

        freeze_encoder:
            Freeze ALL backbone parameters.  Use for fixed feature extraction.

        freeze_up_to_stage:
            Freeze backbone stages 0..N.  -1 = no stage freezing.

        pretrained_path:
            Path to local .pt or .safetensors checkpoint for backbone weights.

        return_patch_tokens:
            If True, include patch-level token sequence in the output.
            Required for cross-attention-based projectors (Q-Former, GCA).
    """

    def __init__(
        self,
        backbone_name: str = "convnext_base",
        img_size: int = 224,
        drop_rate: float = 0.0,
        freeze_encoder: bool = True,
        freeze_up_to_stage: int = -1,
        pretrained_path: str | None = None,
        return_patch_tokens: bool = True,
    ) -> None:
        super().__init__()

        from hale_core.nn.vision import build_vision_backbone

        self.backbone = build_vision_backbone(
            backbone_name,
            img_size=img_size,
            in_chans=3,
            drop_rate=drop_rate,
        )
        self.backbone_name = backbone_name
        self.return_patch_tokens = return_patch_tokens

        # Load pretrained weights
        if pretrained_path:
            from hale_core.config.sections.backbone import _load_pretrained

            _load_pretrained(self.backbone, pretrained_path)

        # Apply freezing strategy
        if freeze_encoder:
            self.backbone.freeze_all()
            self.backbone.eval()
        elif freeze_up_to_stage >= 0:
            self.backbone.freeze_up_to_stage(freeze_up_to_stage)

        self._freeze_encoder = freeze_encoder

    @property
    def hidden_size(self) -> int:
        """Output embedding dimension (compatible with existing VisionTower API)."""
        return self.backbone.embed_dim

    @property
    def patch_token_dim(self) -> int:
        """Dimension of each patch token for cross-attention projectors."""
        return self.backbone.embed_dim

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Run vision encoder.

        Args:
            pixel_values: [B, 3, H, W] normalised float images.

        Returns:
            Patch-level token sequence [B, N, D] if return_patch_tokens is True
            and the backbone supports it (ViT family).
            Otherwise returns global embedding [B, D] as a single "token".

        Note: the VLM projector expects [B, N, D]; this method returns that
        shape in all cases — for CNNs without patch tokens, the pooled embedding
        is unsqueezed to [B, 1, D].
        """
        if self._freeze_encoder:
            with torch.no_grad():
                out = self.backbone(pixel_values)
        else:
            out = self.backbone(pixel_values)

        # Prefer patch tokens if available (ViT, DINOv2, EVA-02, DeiT III)
        if self.return_patch_tokens and out.patch_tokens is not None:
            return out.patch_tokens  # [B, N, D]

        # CNN backbones (ConvNeXt, ResNet, etc.) return None for patch_tokens.
        # Reshape the spatial feature map into a token sequence.
        if out.feature_maps:
            # Use the finest-resolution feature map as "patch tokens"
            feat = out.feature_maps[0]  # [B, C, H, W]
            B, C, H, W = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            return tokens

        # Fallback: wrap pooled embedding as a single-token sequence
        return out.pooled_output.unsqueeze(1)  # [B, 1, D]

    def get_feature_maps(self, pixel_values: Tensor) -> list[Tensor]:
        """Return multi-scale feature maps for dense tasks.

        Useful if the VLM has a dense grounding head that reads spatial features.

        Args:
            pixel_values: [B, 3, H, W]

        Returns:
            list of [B, C_i, H_i, W_i] tensors, finest → coarsest.
        """
        if self._freeze_encoder:
            with torch.no_grad():
                out = self.backbone(pixel_values)
        else:
            out = self.backbone(pixel_values)
        return out.feature_maps


def build_local_vision_tower(
    backbone_name: str = "convnext_base",
    img_size: int = 224,
    drop_rate: float = 0.0,
    freeze_encoder: bool = True,
    freeze_up_to_stage: int = -1,
    pretrained_path: str | None = None,
    return_patch_tokens: bool = True,
) -> LocalVisionTower:
    """Factory function for building LocalVisionTower from keyword args.

    Example::

        tower = build_local_vision_tower("vit_large", img_size=336, freeze_encoder=False)
        tokens = tower(images)   # [B, N, 1024]
    """
    return LocalVisionTower(
        backbone_name=backbone_name,
        img_size=img_size,
        drop_rate=drop_rate,
        freeze_encoder=freeze_encoder,
        freeze_up_to_stage=freeze_up_to_stage,
        pretrained_path=pretrained_path,
        return_patch_tokens=return_patch_tokens,
    )
