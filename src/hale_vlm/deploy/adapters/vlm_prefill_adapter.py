"""Deploy the full VLM prefill (images + prompt → logits) as one graph."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from hale_core.deploy.core.io_signature import IOSignature
from torch import nn

from hale_vlm.deploy.adapters.vlm_part_adapter import CALIBRATION_PROMPTS, VLMPartAdapter
from hale_vlm.deploy.modules.vision_encoder_module import VisionEncoderModule
from hale_vlm.deploy.modules.vlm_prefill_module import VLMPrefillModule
from hale_vlm.deploy.registry import ADAPTERS


@ADAPTERS.register_module()
class VLMPrefillAdapter(VLMPartAdapter):
    """``(pixel_values, input_ids, attention_mask) → logits [batch, N + T, V]``.

    The single-graph form: simplest app integration (one engine, no embedding hand-off) and the
    end-to-end parity reference for the split deployment. The prompt layout is image-first
    (see :class:`VLMPrefillModule`).
    """

    def build_part(self, vlm: nn.Module) -> nn.Module:
        vision = VisionEncoderModule(
            vlm.vision, vlm.projector, self.run_cfg.model.vision.resolved_num_image_tokens()
        )
        return VLMPrefillModule(vision, vlm.llm, vlm.image_token_id)

    def default_signature(self) -> IOSignature:
        size = self.image_size
        max_text = max(1, self.max_length - self.num_image_tokens)
        return IOSignature.from_config(
            {
                "inputs": [
                    {
                        "name": "pixel_values",
                        "shape": ["batch", 3, size, size],
                        "dtype": "float32",
                        "value_range": [-1.0, 1.0],
                    },
                    {
                        "name": "input_ids",
                        "shape": ["batch", "text"],
                        "dtype": "int64",
                        "value_range": [0, max(1, self.vlm.image_token_id)],
                    },  # text ids only (no <image>)
                    {
                        "name": "attention_mask",
                        "shape": ["batch", "text"],
                        "dtype": "int64",
                        "value_range": [1, 2],
                    },
                ],
                "outputs": ["logits"],
                "dims": {
                    "batch": [1, 1, self.max_batch],
                    "text": [1, min(self.opt_text_len, max_text), max_text],
                },
            }
        )

    def calibration_batches(self, num_batches: int) -> Iterator[dict[str, torch.Tensor]] | None:
        images = self.calibration_images_batches(num_batches)
        if images is None:
            return None
        return self._batches(images)

    def _batches(self, images) -> Iterator[dict[str, torch.Tensor]]:
        for i, px in enumerate(images):
            prompts = [
                CALIBRATION_PROMPTS[(i + j) % len(CALIBRATION_PROMPTS)] for j in range(px.shape[0])
            ]
            ids, mask = self.tokenize(prompts, self.opt_text_len)
            yield {"pixel_values": px, "input_ids": ids, "attention_mask": mask}
