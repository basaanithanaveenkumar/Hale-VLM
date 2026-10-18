"""Deploy the VLM's vision tower + projector (the per-image engine)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from hale_core.deploy.core.io_signature import IOSignature
from torch import nn

from hale_vlm.deploy.adapters.vlm_part_adapter import VLMPartAdapter
from hale_vlm.deploy.modules.vision_encoder_module import VisionEncoderModule
from hale_vlm.deploy.registry import ADAPTERS


@ADAPTERS.register_module()
class VisionEncoderAdapter(VLMPartAdapter):
    """``pixel_values [batch, 3, S, S] → image_embeds [batch, N, D]`` (see :class:`VLMPartAdapter`).

    Targets: TensorRT (FP16 / INT8, Jetson DLA), MIGraphX, Ryzen AI NPU, Snapdragon NPU, MLX.
    """

    def build_part(self, vlm: nn.Module) -> nn.Module:
        """Wrap the trained VLM's sub-modules in the exportable part module.

        Args:
            vlm: the built :class:`HaleVLM`.

        Returns:
            The ``nn.Module`` that is traced / exported.
        """
        return VisionEncoderModule(
            vlm.vision, vlm.projector, self.run_cfg.model.vision.resolved_num_image_tokens()
        )

    def default_signature(self) -> IOSignature:
        """The part's input/output tensor signature (names, dtypes, dynamic axes)."""
        size = self.image_size
        return IOSignature.from_config(
            {
                "inputs": [
                    {
                        "name": "pixel_values",
                        "shape": ["batch", 3, size, size],
                        "dtype": "float32",
                        "value_range": [-1.0, 1.0],
                    }
                ],
                "outputs": ["image_embeds"],
                "dims": {"batch": [1, 1, self.max_batch]},
            }
        )

    def calibration_batches(self, num_batches: int) -> Iterator[dict[str, torch.Tensor]] | None:
        """Calibration inputs for quantisation.

        Args:
            num_batches: number of batches to produce.

        Returns:
            An iterator of input dicts, or ``None`` when no calibration data is configured.
        """
        images = self.calibration_images_batches(num_batches)
        return None if images is None else ({"pixel_values": px} for px in images)

    def describe(self) -> dict[str, Any]:
        """Metadata recorded with the artifact: image size, token count, hidden size."""
        return {
            "image_size": self.image_size,
            "image_tokens": self.num_image_tokens,
            "hidden": self.hidden_size,
        }
