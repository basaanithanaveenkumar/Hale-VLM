"""Multimodal dataset and dataloader utilities."""

from __future__ import annotations

# ``AutoTokenizer`` is reachable here so tests can patch
# ``hale_vlm.data.multimodal.AutoTokenizer.from_pretrained`` (load_backbone_tokenizer uses it).
from transformers import AutoTokenizer

from ._common import _collate as _collate
from .encoded_registry_dataset import EncodedRegistryDataset
from .multimodal_data_module import MultimodalDataModule
from .multimodal_dataset import MultimodalDataset
from .multimodal_sample import MultimodalSample
from .registry_streaming_dataset import RegistryStreamingDataset

__all__ = [
    "AutoTokenizer",
    "EncodedRegistryDataset",
    "MultimodalDataModule",
    "MultimodalDataset",
    "MultimodalSample",
    "RegistryStreamingDataset",
]
