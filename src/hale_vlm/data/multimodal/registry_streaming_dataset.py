"""The :class:`RegistryStreamingDataset` class."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.data.registry_stream import iter_registry_vlm_samples
from hale_vlm.data.types import Modality, VLMSample

from .multimodal_sample import MultimodalSample


class RegistryStreamingDataset(IterableDataset):
    """Stream all registered datasets sequentially with bounded sample counts."""

    def __init__(self, cfg: VLMRunConfig, tokenizer) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_token = cfg.model.llm.image_token
        self.max_length = cfg.model.max_length
        self.image_size = cfg.model.vision.image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __iter__(self) -> Iterator[MultimodalSample]:
        data_cfg = self.cfg.data
        for sample in iter_registry_vlm_samples(
            data_cfg,
            image_size=self.image_size,
        ):
            yield self._encode(sample)

    def _encode(self, sample: VLMSample) -> MultimodalSample:
        pixel_values = self._visual_tensor(sample)
        prompt = self._build_prompt(sample)
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return MultimodalSample(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            modality=sample.modality.value,
            dataset=sample.dataset,
        )

    def _build_prompt(self, sample: VLMSample) -> str:
        if sample.modality == Modality.TEXT:
            return sample.text
        if sample.modality == Modality.VIDEO:
            return f"User: {self.image_token}\n{sample.text}\nAssistant: {sample.text}"
        if sample.modality == Modality.MULTI_IMAGE:
            tokens = " ".join(self.image_token for _ in sample.images) or self.image_token
            return f"User: {tokens}\n{sample.text}\nAssistant: {sample.text}"
        return f"User: {self.image_token}\n{sample.text}\nAssistant: {sample.text}"

    def _visual_tensor(self, sample: VLMSample) -> torch.Tensor:
        frames = sample.visual_frames
        if not frames:
            return torch.zeros(3, self.image_size, self.image_size)
        tensors = [self.transform(frame) for frame in frames]
        if sample.modality == Modality.VIDEO and len(tensors) > 1:
            return torch.stack(tensors)
        return tensors[0]
