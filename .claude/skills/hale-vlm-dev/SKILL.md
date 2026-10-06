---
name: hale-vlm-dev
description: Set up, navigate and test Hale-VLM — a parameter-efficient VLM (frozen SigLIP + MLP projector + LoRA on Qwen3-8B / DeepSeek-R1-Distill-Qwen-7B) built on HaleBlocks registries. Use when starting work in this repo, changing model/config/training code, or debugging plugin registration.
---

# Hale-VLM development

## Environment

```bash
uv sync --extra dev            # installs hale-blocks from git (see [tool.uv.sources])
pre-commit install
```

Python ≥ 3.12. Real training needs a GPU and Hugging Face access to `Qwen/Qwen3-8B` or
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and `google/siglip-base-patch16-224`.

## How it plugs into HaleBlocks

HaleBlocks (`hale_core`) owns config loading, registries and the generic `Trainer`.
Hale-VLM registers plugins on import (`hale_vlm/bootstrap.py::register_vlm_plugins`):

| Registry | Registered here |
|---|---|
| `register_model` | `qwen3_8b_vlm`, `deepseek_r1_qwen_7b_vlm` → `HaleVLM` (`models/vlm.py`) |
| `register_loss` | same variant names → `_vlm_loss` (HF causal-LM loss) |
| `register_trainer("vlm")` | `VLMTrainer`: gives the optimiser only `model.trainable_parameters()` |
| `register_dataset` (local `NamedRegistry("vlm_dataset")`) | 19 SmolVLM datasets + 1 disabled (`data/datasets/builtin.py`) |
| VLA registry | SmolVLA community / simulation / real-world datasets (`data/datasets/vla_builtin.py`) |

The `variant:` key in YAML selects the model, loss and trainer. If you add a variant, add it to
`VLM_VARIANTS` in `models/vlm.py` so that both the model and the loss get registered.

## Code map

| Path | What |
|---|---|
| `config/` | `VLMRunConfig` = HaleBlocks `RunConfig` + `model.vision` (`VisionConfig`) + `model.llm` (`LLMConfig`) + `VLMDataConfig` |
| `vision/encoders.py` | `VisionTower` (SigLIP or CLIP via `transformers`) |
| `vision/projector.py` | `VisionProjector`: MLP `d_v → 2·d_llm → d_llm` (GELU) or linear |
| `llm/backbones.py` | `LLMBackbone` wraps `AutoModelForCausalLM` + tokenizer, `embed_tokens` |
| `llm/adapters.py` | `apply_lora` (PEFT), `configure_llm_trainability`, default LoRA targets |
| `models/vlm.py` | `HaleVLM`: `encode_images`, `merge_image_embeddings`, `forward`, `trainable_parameters` |
| `data/` | `catalog.py` / `vla_catalog.py` presets, adapters, `SequentialMultiDatasetStream`, `robotics_vlm.py`, `multimodal.py` data module |
| `training/` | loss, `VLMTrainer`, evaluator |
| `cli/train.py`, `cli/chat.py` | `hale-vlm-train`, `hale-vlm-chat` |

## Tests

```bash
uv run pytest tests/unit -q
uv run pytest tests/smoke -q -m smoke
uv run pytest tests/integration -q -m integration   # tiny models from tests/helpers/tiny_models.py
uv run ruff check src tests
```

Use `tests/helpers/tiny_models.py` for anything that needs a model. Never download 8B
weights in tests.

## Known issues (verify before relying on them)

1. **Image merge overwrites text.** `merge_image_embeddings` replaces `num_image_tokens`
   positions starting at the *first* `<image>` token, but the prompt contains only
   **one** `<image>` placeholder. The `num_image_tokens − 1` text tokens after it are
   therefore dropped (195 with the default 196). Fix options: expand the placeholder to
   `num_image_tokens` copies when tokenising (then realign `labels`), or splice by
   inserting rather than overwriting and extend `labels` with `-100`.
2. **Loss on the whole prompt.** `RegistryStreamingDataset._encode` sets
   `labels = input_ids` (only padding masked), so user and prompt tokens are trained
   too. The prompt template also repeats `sample.text` as the assistant answer.
3. Multi-image and video samples encode only the first frame (`_visual_tensor`
   returns `tensors[0]` unless the modality is video).
