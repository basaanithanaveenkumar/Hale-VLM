# Hale-VLM

Vision-language models built on dense transformer LLMs, using [HaleBlocks](https://github.com/basaanithanaveenkumar/HaleBlocks) for training infrastructure and reusable transformer components.

## Resources

| | |
|---|---|
| Paper (arXiv source) | [`paper/main.tex`](paper/main.tex) — build with `make -C paper` |
| Project page | [basaanithanaveenkumar.github.io/Hale-VLM](https://basaanithanaveenkumar.github.io/Hale-VLM/) ([source](project-page/index.html)) |
| Documentation | [`docs/`](docs/README.md) — getting started, configuration, data, [known issues](docs/known-issues.md) |
| Architecture diagrams | [`docs/architecture.md`](docs/architecture.md) (Mermaid) |
| Blog | [Giving an 8B reasoning model eyes for about 1% of its parameters](docs/blog/2026-09-28-eyes-for-a-reasoning-llm.md) |
| Claude Code skills | [`.claude/skills/`](.claude/skills) — `hale-vlm-dev`, `hale-vlm-train`, `hale-vlm-datasets`, `hale-publish` |

## Supported LLM backbones

| Backbone | Params | Architecture | Notes |
|----------|--------|--------------|-------|
| **Qwen3-8B** | 8B dense | Transformer | Lowest integration risk, best tooling |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B dense | Transformer | Reasoning-heavy VLM |

Both backbones load via HuggingFace `transformers` and are connected to a frozen SigLIP vision encoder through a trainable MLP projector. The LLM is **not** fully fine-tuned — by default only **LoRA adapters** on the LLM plus the **projector** (and optionally the vision tower) are trained.

## Architecture

```text
Image ──► Vision Tower (SigLIP, frozen) ──► Projector (trainable) ──► LLM embeddings
                                                                          │
Text  ──► Tokenizer ──────────────────────────────────────────────────────┘
                                                                          │
                                                                          ▼
                                                    Causal LM + LoRA adapters (trainable)
```

### Fine-tuning policy (default)

| Component | Default | Config |
|-----------|---------|--------|
| Vision encoder | Frozen | `model.vision.freeze_encoder: true` |
| Projector | Trainable | always |
| LLM base weights | Frozen | `model.llm.freeze_llm: true` |
| LLM LoRA adapters | Trainable | `model.llm.use_lora: true` |

Hale-VLM plugs into HaleBlocks registries for config, models, losses, and the training loop.

## Install

```bash
uv sync --extra dev
```

The project depends on HaleBlocks from git:

```bash
uv add "hale-blocks @ git+https://github.com/basaanithanaveenkumar/HaleBlocks.git"
```

## Quick start

Load a config and inspect the VLM schema:

```python
from hale_vlm.config import load_vlm_config

cfg = load_vlm_config("configs/base.yaml")
print(cfg.model.llm.backbone, cfg.model.vision.encoder)
```

Train on a tiny overfit loop (no GPU weights download required for config validation; full training needs GPU + HF model access):

```bash
uv run hale-vlm-train configs/qwen3_8b_overfit.yaml
```

Chat with an image after training or checkpoint load:

```bash
uv run hale-vlm-chat configs/base.yaml --image path/to/image.jpg --prompt "What is in this image?"
```

## Configs

| File | Purpose |
|------|---------|
| `configs/base.yaml` | Shared defaults for Qwen3-8B VLM |
| `configs/qwen3_8b_overfit.yaml` | Small overfit run for local smoke testing |
| `configs/deepseek_r1_qwen_7b.yaml` | DeepSeek-R1-Distill-Qwen-7B reasoning backbone |
| `configs/smolvlm_all.yaml` | Full SmolVLM paper mixture (19 datasets, streaming) |
| `configs/smolvlm_vision.yaml` | Vision training stage only |
| `configs/smolvlm_video.yaml` | Video fine-tuning stage only |
| `configs/smolvlm_context.yaml` | Long-context extension stage only |
| `configs/registry_mix.yaml` | Alias for `smolvlm_all.yaml` |

Configs use HaleBlocks YAML inheritance (`inherits:`) and the `vlm` config schema.

## Dataset registry (SmolVLM paper)

All datasets from [SmolVLM (arXiv:2504.05299)](https://arxiv.org/abs/2504.05299) are registered with stage-aware skeleton adapters. Streaming + `max_samples_per_dataset` keeps local disk usage bounded; the sequential mixer prefetches the next dataset while the current one is consumed.

### Vision training stage (§4.1)

| Registry name | HuggingFace path | Role |
|---------------|------------------|------|
| `the-cauldron` | `HuggingFaceM4/the_cauldron` | Laurençon et al. (2024) vision mixture |
| `docmatix` | `HuggingFaceM4/Docmatix` | OCR & documents |
| `mathwriting` | `google/mathwriting` | Handwritten math OCR (added) |
| `llava-onevision-data` | `lmms-lab/LLaVA-OneVision-Data` | Captioning / instructions |

### Video fine-tuning stage (§4.1)

| Registry name | HuggingFace path | Role |
|---------------|------------------|------|
| `llava-video-178k` | `lmms-lab/LLaVA-Video-178K` | Visual description |
| `videostar` | `orrzohar/Video-STaR` | Visual description |
| `vript` | `Mutonix/Vript` | Visual description |
| `sharegpt4video` | `ShareGPT4Video/ShareGPT4Video` | Visual description |
| `vista-400k` | `TIGER-Lab/VISTA-400K` | Temporal understanding |
| `moviechat` | `Enxin/MovieChat-1K_train` | Narrative comprehension |
| `finevideo` | `HuggingFaceFV/finevideo` | Narrative comprehension |
| `m4-instruct-data` | `lmms-lab/M4-Instruct-Data` | Multi-image |
| `mammoth` | `MAmmoTH-VL/MAmmoTH-VL-Instruct-12M` | Multi-image |
| `magpie` | `Magpie-Align/Magpie-Pro-300K-Filtered` | Text SFT (14%; Xu et al. 2024) |

### Long-context extension (§2.2)

| Registry name | HuggingFace path |
|---------------|------------------|
| `dolma-books` | `allenai/dolma` (books) |
| `the-stack` | `bigcode/the-stack` |
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` |
| `dclm` | `mlfoundations/dclm-baseline-1.0` |
| `smollm2-math` | `HuggingFaceTB/smollm-corpus` (math) |

### Rejected (§3.3 — registered but disabled)

| Registry name | Reason |
|---------------|--------|
| `smoltalk` | Reusing LLM-SFT SmolTalk hurt video (-3.7%) and image (-6.5%) performance |

```python
from hale_vlm.data import (
    SMOLVLM_VISION_DATASETS,
    SequentialMixConfig,
    SequentialMultiDatasetStream,
    list_datasets,
)

# All 19 enabled datasets
print(len(list_datasets()), "datasets")

# Vision stage only
stream = SequentialMultiDatasetStream(
    SequentialMixConfig(
        dataset_names=list(SMOLVLM_VISION_DATASETS),
        max_samples_per_dataset=128,
        prefetch_workers=2,
        streaming=True,
    )
)
```

Set `data.source: registry` and `data.registry_stage: all|vision|video|context` in YAML.

## VLA dataset registry (SmolVLA)

Separate registry for robotics / vision-language-action data ([SmolVLA paper](https://arxiv.org/abs/2506.01844)):

| Stage | Count | Examples |
|-------|-------|----------|
| Community pretraining | 534 HF paths (paper: 481) | `satvikahuja--mixer_on_off_new_1`, `aergogo--so100_pick_place`, … |
| Simulation | 2 | `libero`, `metaworld-mt50` |
| Real-world (authors) | 4 | `svla-so100-pickplace`, `svla-so100-stacking`, `svla-so100-sorting`, `svla-so101-pickplace` |

```python
from hale_vlm.data.vla_catalog import SMOLVLA_ALL_DATASETS, SMOLVLA_REAL_WORLD_DATASETS
from hale_vlm.data.vla_registry import list_vla_datasets, build_vla_dataset

print(len(list_vla_datasets()), "VLA datasets")
adapter = build_vla_dataset("svla-so100-pickplace")
```

YAML presets: `configs/smolvla_all.yaml`, `smolvla_community.yaml`, `smolvla_simulation.yaml`, `smolvla_real_world.yaml`.

### Robotics data in VLM training

Robotics datasets can be converted into VLM samples (image + task instruction) for pretraining, finetuning, or instruction tuning:

```yaml
data:
  source: mixed_registry          # VLM + robotics
  registry_stage: vision
  vla_registry_stage: real_world
  robotics_vlm_mode: pretraining  # off | pretraining | finetuning | instruction_tuning
```

Or add robotics to an existing VLM registry run:

```yaml
data:
  source: registry
  registry_stage: all
  robotics_vlm_mode: finetuning
  vla_registry_stage: real_world
```

Presets: `configs/vlm_with_robotics_pretrain.yaml`, `configs/vlm_with_robotics_finetune.yaml`.

## Layout

```text
src/hale_vlm/
  config/         VLMRunConfig (vision + llm sections)
  vision/         SigLIP/CLIP encoder, MLP projector
  llm/            HuggingFace Qwen3 / DeepSeek-R1 loaders
  models/         HaleVLM assembly + variant registration
  data/           Dataset registry, streaming sequential mixer, data module
  training/       VLM loss + evaluator plugins
  cli/            train and chat entry points
configs/          YAML experiment configs
tests/            smoke and unit tests
```

## Develop

```bash
uv sync --extra dev
pre-commit install
pre-commit run --all-files          # run all hooks once
uv run pytest tests/unit -q
uv run pytest tests/smoke -q -m smoke
uv run pytest tests/integration -q -m integration
uv run ruff check src tests
```

## License

MIT
