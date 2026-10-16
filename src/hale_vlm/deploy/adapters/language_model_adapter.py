"""Deploy the VLM's language model (prefill graph and the HF checkpoint for LLM runtimes)."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any, Literal

import torch
from loguru import logger
from torch import nn

from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.exceptions import DeployError
from hale_vlm.deploy.adapters.vlm_part_adapter import CALIBRATION_PROMPTS, VLMPartAdapter
from hale_vlm.deploy.modules.language_model_module import LanguageModelModule
from hale_vlm.deploy.registry import ADAPTERS


@ADAPTERS.register_module()
class LanguageModelAdapter(VLMPartAdapter):
    """Prefill graph ``(inputs_embeds | input_ids, attention_mask) → logits`` plus :meth:`hf_checkpoint`.

    Two deployment routes:

    * **graph** (ONNX → TensorRT / MIGraphX / QNN, or the MLX bridge) - ``mode: embeds`` runs on
      the merged ``[visual ; text]`` sequence the vision engine feeds;
    * **LLM runtime** (TensorRT-LLM, mlx-lm, Genie) - :meth:`hf_checkpoint` hands the text tower
      (with the ``<image>`` token in its vocabulary, LoRA merged) to
      ``HuggingFaceCheckpointStage``; those runtimes add KV cache, paged attention, batching.

    Args:
        mode: ``embeds`` (VLM prefill) or ``ids`` (text-only LM).
    """

    def __init__(self, mode: Literal["embeds", "ids"] = "embeds", **base: Any) -> None:
        super().__init__(**base)
        self.mode = mode

    def build_part(self, vlm: nn.Module) -> nn.Module:
        return LanguageModelModule(vlm.llm, self.mode)

    def default_signature(self) -> IOSignature:
        first = ({"name": "inputs_embeds", "shape": ["batch", "seq", self.hidden_size], "dtype": "float32",
                  "value_range": [-0.05, 0.05]} if self.mode == "embeds" else
                 {"name": "input_ids", "shape": ["batch", "seq"], "dtype": "int64", "value_range": [0, self.vocab_size]})
        return IOSignature.from_config({
            "inputs": [first, {"name": "attention_mask", "shape": ["batch", "seq"], "dtype": "int64",
                               "value_range": [1, 2]}],
            "outputs": ["logits"],
            "dims": {"batch": [1, 1, self.max_batch],
                     "seq": [1, min(self.num_image_tokens + self.opt_text_len, self.max_length), self.max_length]}})

    def hf_checkpoint(self, model: nn.Module) -> tuple[Any, Any]:
        """The HF causal LM (LoRA merged on a copy) and its tokenizer, ``<image>`` included."""
        hf_model = self.vlm.llm.model
        if not hasattr(hf_model, "save_pretrained"):
            raise DeployError(f"LLM backbone {self.run_cfg.model.llm.backbone!r} is not a Hugging Face model; "
                              "use the graph route (ONNX / MLX bridge) instead")
        if hasattr(hf_model, "merge_and_unload"):
            logger.info("merging LoRA adapters into a copy of the LLM for export")
            hf_model = copy.deepcopy(hf_model).merge_and_unload()
        return hf_model, self.vlm.tokenizer

    def calibration_batches(self, num_batches: int) -> Iterator[dict[str, torch.Tensor]] | None:
        images = self.calibration_images_batches(num_batches)
        if images is None:
            return None
        return self._batches(images)

    def _batches(self, images) -> Iterator[dict[str, torch.Tensor]]:
        vlm = self.vlm
        for i, px in enumerate(images):
            ids, mask = self.tokenize([CALIBRATION_PROMPTS[i % len(CALIBRATION_PROMPTS)]] * px.shape[0],
                                      self.opt_text_len)
            with torch.no_grad():
                image = vlm.encode_images(px)
                if self.mode == "ids":
                    yield {"input_ids": ids, "attention_mask": mask}
                    continue
                embeds = torch.cat([image, vlm.llm.embed_tokens(ids)], dim=1)
            full_mask = torch.cat([mask.new_ones(px.shape[0], image.shape[1]), mask], dim=1)
            yield {"inputs_embeds": embeds, "attention_mask": full_mask}
