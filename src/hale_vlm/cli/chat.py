"""Image+text chat CLI (``hale-vlm-chat``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from hale_vlm.config import load_vlm_config
from hale_vlm.models.vlm import build_vlm


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    """Load ``path`` as a normalised ``[1, 3, image_size, image_size]`` tensor."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def main() -> None:
    """Answer one prompt about one image with the configured VLM and print the reply.

    Example:
        ``hale-vlm-chat configs/base.yaml --image cat.png --prompt "What is this?"``
    """
    parser = argparse.ArgumentParser(description="Chat with a Hale-VLM model")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config overrides, e.g. model.llm.dtype=float32",
    )
    args = parser.parse_args()

    cfg = load_vlm_config(args.config, args.cfg_options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_vlm(cfg).to(device)
    model.eval()

    tokenizer = model.tokenizer
    image_token = cfg.model.llm.image_token
    user_prompt = f"User: {image_token}\n{args.prompt}\nAssistant:"
    encoded = tokenizer(user_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    pixel_values = _load_image(args.image, cfg.model.vision.image_size).to(device)

    with torch.no_grad():
        text_embeds = model.llm.embed_tokens(input_ids)
        image_embeds = model.encode_images(pixel_values)
        inputs_embeds, attention_mask = model.merge_image_embeddings(
            input_ids,
            text_embeds,
            image_embeds,
        )
        outputs = model.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )

    # The reply is the CLI's product, so it goes to stdout; diagnostics go through loguru.
    sys.stdout.write(tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n")


if __name__ == "__main__":
    main()
