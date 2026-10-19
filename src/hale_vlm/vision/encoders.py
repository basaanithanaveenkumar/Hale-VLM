"""Vision towers: Hugging Face SigLIP / CLIP encoders, plus the registry-driven builder."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel, SiglipVisionModel

from hale_vlm.config.sections.vision import VisionConfig
from hale_vlm.vision.vision_registries import VISION_TOWERS

# encoder name -> Hugging Face model class; anything else loads through AutoModel.
_HF_ENCODER_CLASSES: dict[str, type] = {"siglip": SiglipVisionModel, "clip": CLIPVisionModel}


class VisionTower(nn.Module):
    """Frozen (by default) Hugging Face vision encoder producing patch-level embeddings.

    Args:
        cfg: vision settings; ``model_id`` is loaded with the class of ``cfg.encoder``.
    """

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_type = cfg.encoder
        self.model = self._load_encoder(cfg.model_id, cfg.encoder)
        if cfg.freeze_encoder:
            self.model.requires_grad_(False)
            self.model.eval()

    @staticmethod
    def _load_encoder(model_id: str, encoder: str) -> nn.Module:
        return _HF_ENCODER_CLASSES.get(encoder, AutoModel).from_pretrained(model_id)

    @property
    def hidden_size(self) -> int:
        """Width of the patch embeddings."""
        return int(self.model.config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 3, H, W)`` pixels to ``(B, N_patches, hidden_size)`` embeddings."""
        if self.cfg.freeze_encoder:
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
        else:
            outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state


@VISION_TOWERS.register_module(name=["siglip", "clip"])
def _build_hf_tower(cfg: VisionConfig) -> VisionTower:
    return VisionTower(cfg)


@VISION_TOWERS.register_module(name="local")
def _build_local_tower(cfg: VisionConfig) -> nn.Module:
    # Imported lazily: it pulls in the hale_core.nn.vision backbone zoo.
    from hale_vlm.vision.local_encoders import build_local_vision_tower

    lb = cfg.local_backbone
    return build_local_vision_tower(
        backbone_name=lb.backbone_name,
        img_size=cfg.image_size,
        drop_rate=lb.drop_rate,
        freeze_encoder=lb.freeze_encoder,
        freeze_up_to_stage=lb.freeze_up_to_stage,
        pretrained_path=lb.pretrained_path,
        return_patch_tokens=lb.return_patch_tokens,
    )


def build_vision_tower(cfg: VisionConfig) -> nn.Module:
    """Build the vision tower registered under ``cfg.encoder``.

    Args:
        cfg: vision settings.

    Returns:
        A module with ``hidden_size`` and ``forward(pixel_values)``.

    Raises:
        hale_core.exceptions.RegistryError: for an unregistered encoder name.
    """
    return VISION_TOWERS.get(cfg.encoder)(cfg)
