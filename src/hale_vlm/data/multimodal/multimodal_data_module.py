"""MultimodalDataModule: HaleBlocks-compatible data module for VLM training."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import torch.distributed as dist
from datasets import load_dataset
from hale_core.data.webdataset import WebDatasetAdapter
from hale_core.nn.llm import load_backbone_tokenizer
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.llm.backbones import resolve_llm_config

from ._common import _collate
from .encoded_registry_dataset import EncodedRegistryDataset
from .multimodal_dataset import MultimodalDataset


class MultimodalDataModule:
    """HaleBlocks-compatible data module for VLM training."""

    def __init__(self, cfg: VLMRunConfig, tokenizer=None) -> None:
        self.cfg = cfg
        if tokenizer is None:
            # The backbone's own tokenizer (Hub repo, or the one a HaloDiffusionLLM checkpoint
            # was trained with), so token ids always match the model's vocabulary.
            self.tokenizer = load_backbone_tokenizer(
                resolve_llm_config(cfg.model.llm).to_backbone_config()
            )
        else:
            self.tokenizer = tokenizer
        self._train: Dataset | IterableDataset | None = None
        self._val: Dataset | None = None

    def _load_split(self, split: str) -> list[dict[str, Any]]:
        data_cfg = self.cfg.data
        if data_cfg.source == "overfit":
            text = data_cfg.overfit_text or "A red square on a white background."
            size = self.cfg.model.vision.image_size
            image = Image.new("RGB", (size, size), "white")
            copies = data_cfg.n_overfit_copies if split == data_cfg.train_split else 4
            return [
                {"image": image.copy(), "text": text, "dataset": "overfit"} for _ in range(copies)
            ]

        ds = load_dataset(
            data_cfg.dataset,
            data_cfg.subset,
            split=split,
            cache_dir=data_cfg.cache_dir,
        )
        records = []
        for row in ds:
            image = row.get("image")
            text = row.get(data_cfg.text_field) or row.get("caption") or row.get("text")
            if image is None or text is None:
                continue
            records.append({"image": image, "text": text, "dataset": data_cfg.dataset})
            limit = data_cfg.train_size if split == data_cfg.train_split else data_cfg.val_size
            if limit is not None and len(records) >= limit:
                break
        return records

    # ------------------------------------------------------------------ datasets
    def _uses_registry(self) -> bool:
        return self.cfg.data.source in {
            "registry",
            "phase_registry",
            "vla_registry",
            "mixed_registry",
        }

    def _build_train_dataset(self) -> Dataset:
        if self._uses_registry():
            return EncodedRegistryDataset(self.cfg, self.tokenizer)
        return self._caption_dataset(self.cfg.data.train_split)

    def _build_val_dataset(self) -> Dataset:
        return self._caption_dataset(self.cfg.data.val_split)

    def _caption_dataset(self, split: str) -> MultimodalDataset:
        return MultimodalDataset(
            self._load_split(split),
            tokenizer=self.tokenizer,
            image_token=self.cfg.model.llm.image_token,
            image_size=self.cfg.model.vision.image_size,
            max_length=self.cfg.model.max_length,
        )

    def dataset_fingerprint(self, split: str) -> str:
        """Everything that changes the encoded samples of ``split``.

        Data source/selection, prompt template token, image size, sequence length and the
        tokenizer. Training hyper-parameters, LoRA and loader settings are deliberately
        excluded so they do not trigger a re-conversion.
        """
        data = self.cfg.data.model_dump(mode="json", exclude={"webdataset", "loader"})
        spec = {
            "split": split,
            "data": data,
            "max_length": self.cfg.model.max_length,
            "image_size": self.cfg.model.vision.image_size,
            "image_token": self.cfg.model.llm.image_token,
            "tokenizer": getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__name__),
        }
        return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()

    def _prepare(self, factory, split: str, *, training: bool) -> Dataset:
        """The dataset from ``factory``, or its WebDataset stream with ``data.webdataset.enabled``
        (then ``factory`` - downloads, image decoding, tokenisation - only runs when the shards
        are missing or stale)."""
        fingerprint = self.dataset_fingerprint(split)
        source = self.cfg.data.source if self._uses_registry() else self.cfg.data.dataset
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{source}-{split}") + f"-{fingerprint[:12]}"
        return WebDatasetAdapter(self.cfg.data.webdataset).prepare(
            factory,
            name,
            training=training,
            loader_workers=self.cfg.data.loader.num_workers,
            fingerprint=fingerprint,
        )

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        kwargs: dict[str, Any] = {"batch_size": self.cfg.train.batch_size, "collate_fn": _collate}
        if isinstance(dataset, IterableDataset):
            pass  # streams (registry, WebDataset) order / shuffle / split themselves
        elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # DDP / Ray: a disjoint 1/world_size of the samples per rank; the HaleBlocks trainer
            # calls sampler.set_epoch(epoch) every epoch.
            kwargs["sampler"] = DistributedSampler(dataset, shuffle=shuffle)
        else:
            kwargs["shuffle"] = shuffle
        return DataLoader(dataset, **kwargs)

    # ------------------------------------------------------------------ HaleBlocks DataModule
    def train_loader(self) -> DataLoader:
        """Build the training :class:`DataLoader` (registry stream or captioned HF dataset)."""
        if self._train is None:
            self._train = self._prepare(
                self._build_train_dataset, self.cfg.data.train_split, training=True
            )
        # Registry sources keep their streaming order; caption datasets are shuffled.
        return self._loader(self._train, shuffle=not self._uses_registry())

    def val_loader(self, train_loader: DataLoader | None = None) -> DataLoader:
        """Build the validation :class:`DataLoader`.

        Args:
            train_loader: the training loader, reused for its collate function when given.
        """
        del train_loader
        if self._val is None:
            self._val = self._prepare(
                self._build_val_dataset, self.cfg.data.val_split, training=False
            )
        return self._loader(self._val, shuffle=False)

    def viz_prompt(self, step: int) -> str:
        """Return the text prompt used for the visualisation sample at training ``step``.

        Args:
            step: current training step.
        """
        del step
        return f"User: {self.cfg.model.llm.image_token}\nDescribe the image.\nAssistant:"
