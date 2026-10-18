"""Shared base of the Hale-VLM deployment adapters (Template Method over one built VLM)."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.exceptions import ConfigError
from loguru import logger
from torch import nn

from hale_vlm.deploy.calibration.calibration_image_loader import CalibrationImageLoader

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hale_vlm.config import VLMRunConfig

#: Prompts used to build realistic text-side calibration inputs.
CALIBRATION_PROMPTS = (
    "describe this image .",
    "what color is the shape ?",
    "what is on the background ?",
    "is it a square or a circle ?",
)


class VLMPartAdapter(ModelAdapter):
    """Build the :class:`~hale_vlm.models.vlm.HaleVLM` of a run config once and expose one part.

    Subclasses return the exportable part (:meth:`build_part`), its signature and calibration
    inputs; this base owns config loading, checkpoint loading into the *full* VLM (training
    checkpoints hold the whole model) and the shared geometry (image size, visual token count,
    hidden size, vocabulary, context length).

    Args:
        run_config: VLM run config YAML (the training config; ``inherits`` supported).
        run_overrides: nested mapping merged over the run config (e.g. smaller ``max_length``).
        tiny_assets: directory of offline tiny SigLIP + Llama assets (CI / laptops); created on
            demand and merged *before* ``run_overrides``.
        max_batch: largest batch the exported graphs must accept.
        opt_text_len: prompt length used for tracing and TensorRT kernel selection.
        calibration_images: folder of images for quantization calibration (``None``: random).
        checkpoint: trainer checkpoint of the full VLM (``model_state_dict`` or plain state dict).
    """

    def __init__(
        self,
        run_config: str,
        run_overrides: Mapping[str, Any] | None = None,
        tiny_assets: str | None = None,
        max_batch: int = 4,
        opt_text_len: int = 32,
        calibration_images: str | None = None,
        **base: Any,
    ) -> None:
        base.setdefault("strict", False)  # LoRA/PEFT naming differs between train and export builds
        super().__init__(**base)
        self.run_config = run_config
        self.run_overrides = dict(run_overrides or {})
        self.tiny_assets = tiny_assets
        self.max_batch = max_batch
        self.opt_text_len = opt_text_len
        self.calibration_images = calibration_images

    # ------------------------------------------------------------------ the VLM
    @cached_property
    def run_cfg(self) -> VLMRunConfig:
        """The validated :class:`VLMRunConfig` (YAML + tiny-asset and run overrides), built once."""
        from hale_core.config.load import load_yaml_tree
        from hale_core.config.merge import deep_merge
        from pydantic import ValidationError

        from hale_vlm.config import VLMRunConfig

        path = Path(self.run_config)
        if not path.is_file():
            raise ConfigError(f"VLM run config not found: {path.resolve()}")
        raw = load_yaml_tree(path)
        if self.tiny_assets:
            from hale_vlm.deploy.testing.tiny_vlm_assets import TinyVLMAssets

            raw = deep_merge(raw, TinyVLMAssets(self.tiny_assets).ensure().run_overrides())
        raw = deep_merge(raw, self.run_overrides)
        try:
            return VLMRunConfig.model_validate(raw)
        except ValidationError as exc:  # report schema violations as a config problem
            raise ConfigError(f"invalid VLM run config {path}: {exc}") from exc

    @cached_property
    def vlm(self) -> nn.Module:
        """The float32 evaluation-mode :class:`HaleVLM` shared by every part, built once."""
        from hale_vlm.models.vlm import HaleVLM

        model = HaleVLM.from_config(self.run_cfg).float().eval()
        logger.info(
            "built HaleVLM for deployment (llm={}, vision={})",
            self.run_cfg.model.llm.backbone,
            self.run_cfg.model.vision.encoder,
        )
        return model

    @cached_property
    def num_image_tokens(self) -> int:
        """Visual tokens actually produced (patches after projector truncation), measured once."""
        size = self.image_size
        with torch.no_grad():
            return int(self.vlm.encode_images(torch.zeros(1, 3, size, size)).shape[1])

    @property
    def image_size(self) -> int:
        """Square input resolution of the vision tower."""
        return int(self.run_cfg.model.vision.image_size)

    @property
    def hidden_size(self) -> int:
        """Hidden size of the language model."""
        return int(self.vlm.llm.hidden_size)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the language model."""
        return int(self.vlm.llm.vocab_size)

    @property
    def max_length(self) -> int:
        """Maximum sequence length of the model config."""
        return int(self.run_cfg.model.max_length)

    # ------------------------------------------------------------------ ModelAdapter
    def build_model(self) -> nn.Module:
        """Build the part module to export (loads the VLM, then calls :meth:`build_part`)."""
        return self.build_part(self.vlm)

    @abstractmethod
    def build_part(self, vlm: nn.Module) -> nn.Module:
        """The exportable sub-module of ``vlm``."""

    def load_checkpoint(self, model: nn.Module, path: Path) -> None:
        """Load a training checkpoint into the full VLM (the part shares its modules)."""
        super().load_checkpoint(self.vlm, path)

    def forward_for_export(self, model: nn.Module, inputs: dict[str, torch.Tensor]) -> Any:
        """Run the part on keyword tensors (the export graph's calling convention).

        Args:
            model: the module returned by :meth:`build_model`.
            inputs: tensors keyed by the signature's input names.
        """
        return model(*inputs.values())

    # ------------------------------------------------------------------ calibration helpers
    def calibration_images_batches(self, num_batches: int) -> Iterator[torch.Tensor] | None:
        """Image batches from the configured calibration folder.

        Args:
            num_batches: number of batches to produce.

        Returns:
            An iterator of ``[B, 3, H, W]`` tensors, or ``None`` when no folder is configured.
        """
        if not self.calibration_images:
            return None
        return CalibrationImageLoader(self.calibration_images, self.image_size).batches(num_batches)

    def tokenize(self, prompts: Sequence[str], length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise prompts to fixed-length ids and mask.

        Args:
            prompts: texts to encode.
            length: padded / truncated sequence length.

        Returns:
            ``(input_ids, attention_mask)`` of shape ``[len(prompts), length]``.
        """
        tokens = self.vlm.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return tokens["input_ids"].long(), tokens["attention_mask"].long()
