"""Turn a folder of images into normalised ``pixel_values`` batches (the training transform)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from hale_core.deploy.exceptions import ConfigError
from loguru import logger

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class CalibrationImageLoader:
    """Resize → tensor → normalise with mean = std = 0.5, exactly as ``MultimodalDataModule``.

    Quantization calibrates activation ranges, so calibration images must go through the same
    preprocessing as training/serving; using the data module's transform keeps them in sync.

    Args:
        directory: folder searched recursively for images.
        image_size: square resize target (``model.vision.image_size``).
        batch_size: images per batch.
    """

    def __init__(self, directory: str | Path, image_size: int, batch_size: int = 1) -> None:
        self.directory = Path(directory)
        self.image_size = image_size
        self.batch_size = batch_size
        if not self.directory.is_dir():
            raise ConfigError(f"calibration image folder not found: {self.directory}")

    def files(self) -> list[Path]:
        """Sorted image files found in the calibration folder."""
        return sorted(p for p in self.directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

    def batches(self, num_batches: int) -> Iterator[torch.Tensor]:
        """Yield normalised image batches.

        Args:
            num_batches: number of batches to produce (capped by the number of files available).

        Yields:
            ``[B, 3, H, W]`` float tensors.
        """
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        files = self.files()
        if not files:
            raise ConfigError(f"no images in {self.directory}")
        logger.info(
            "calibrating on {} images from {}",
            min(len(files), num_batches * self.batch_size),
            self.directory,
        )
        for start in range(0, min(len(files), num_batches * self.batch_size), self.batch_size):
            chunk = files[start : start + self.batch_size]
            yield torch.stack([transform(Image.open(f).convert("RGB")) for f in chunk])
