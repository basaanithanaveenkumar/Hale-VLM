"""MultimodalDataset: Image + caption pairs with prompt templating."""

from __future__ import annotations

from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .multimodal_sample import MultimodalSample


class MultimodalDataset(Dataset):
    """Image + caption pairs with prompt templating."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        tokenizer,
        image_token: str,
        image_size: int,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MultimodalSample:
        row = self.records[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        pixel_values = self.transform(image)

        caption = row["text"]
        prompt = f"User: {self.image_token}\nDescribe the image.\nAssistant: {caption}"
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
            modality="image",
            dataset=row.get("dataset", "local"),
        )
