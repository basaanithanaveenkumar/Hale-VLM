"""The :class:`EncodedRegistryDataset` class."""

from __future__ import annotations

from torch.utils.data import Dataset

from hale_vlm.config.run import VLMRunConfig

from .multimodal_sample import MultimodalSample
from .registry_streaming_dataset import RegistryStreamingDataset


class EncodedRegistryDataset(Dataset):
    """Materialize a bounded slice from the sequential registry stream for map-style training."""

    def __init__(self, cfg: VLMRunConfig, tokenizer) -> None:
        self.samples = list(RegistryStreamingDataset(cfg, tokenizer))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalSample:
        return self.samples[idx]
