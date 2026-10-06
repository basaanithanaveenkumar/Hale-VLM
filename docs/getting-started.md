# Getting started

## Install

```bash
git clone https://github.com/basaanithanaveenkumar/Hale-VLM && cd Hale-VLM
uv sync --extra dev          # pulls hale-blocks from GitHub
```

Python ≥ 3.12. Full training needs a GPU with bf16 support and Hugging Face access to the
backbone and SigLIP weights.

## Inspect a config

```python
from hale_vlm.config import load_vlm_config
cfg = load_vlm_config("configs/base.yaml")
print(cfg.model.llm.backbone, cfg.model.llm.use_lora, cfg.model.vision.encoder)
```

## Train

```bash
uv run hale-vlm-train configs/qwen3_8b_overfit.yaml        # smoke test
uv run hale-vlm-train configs/smolvlm_vision.yaml          # SmolVLM vision stage
uv run hale-vlm-train configs/deepseek_r1_qwen_7b.yaml     # reasoning backbone
uv run hale-vlm-train configs/vlm_with_robotics_pretrain.yaml
```

The first log line reports what is trainable:
`trainable components: vision_frozen=True llm_mode=lora trainable=... / ...`.

## Chat

```bash
uv run hale-vlm-chat configs/base.yaml --image cat.jpg --prompt "What is in this image?"
```

## Develop

```bash
pre-commit install
uv run pytest tests/unit -q
uv run pytest tests/smoke -q -m smoke
uv run pytest tests/integration -q -m integration
uv run ruff check src tests
```

Read [known issues](known-issues.md) before interpreting training results.
