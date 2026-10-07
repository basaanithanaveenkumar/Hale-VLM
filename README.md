# Hale-VLM

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Hale--VLM-181717?logo=github&logoColor=white)](https://github.com/basaanithanaveenkumar/Hale-VLM)
[![Project Page](https://img.shields.io/badge/🌐_Project-Page-4A90D9)](https://basaanithanaveenkumar.github.io/Hale-VLM/)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b?logo=arxiv&logoColor=white)](https://github.com/basaanithanaveenkumar/Hale-VLM/blob/main/paper/main.tex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

</div>

## How it works

A frozen image encoder reads the photo and converts it to a sequence of visual tokens. A tiny trainable MLP projector translates those tokens into the language model's vocabulary space. The frozen LLM — with small LoRA patches on its attention layers — reads both the visual tokens and your text prompt and generates an answer. Only ~1% of the total weights are ever updated.

```mermaid
flowchart LR
  subgraph INPUT["What you give it"]
    IMG["🖼️ Photo\n(224 × 224)"]
    PROMPT["💬 Question or instruction"]
  end

  subgraph FROZEN["Frozen — never updated"]
    ENC["SigLIP encoder\nreads pixels → 196 visual tokens"]
    LLM["Qwen3-8B or DeepSeek-R1-7B\n(language understanding)"]
  end

  subgraph TRAINED["Trained — ~1% of total weights"]
    PROJ["MLP projector\n(connects vision to language)"]
    LORA["LoRA adapters\n(fine-tune attention layers)"]
  end

  IMG --> ENC --> PROJ --> LLM
  PROMPT --> LLM
  LORA -.->|"patched onto"| LLM
  LLM --> ANS["💬 Answer"]
```

> **Why freeze most of it?** The LLM already knows language deeply. You only need to teach it to *look* — the projector + LoRA do that in a fraction of the compute.

> **Note:** This repo has been merged into [Halo-VLM](https://github.com/basaanithanaveenkumar/Halo-VLM). Use the Halo-VLM repo for ongoing work — it contains both `src/hale_vlm/` and `src/halo_vlm/`.

Vision-language models built on Qwen3 and DeepSeek-R1 LLM backbones, powered by HaleBlocks.

See the Halo-VLM README for install and usage.

## Resources

| | |
|---|---|
| Paper (arXiv source) | [`paper/main.tex`](paper/main.tex) — build with `make -C paper` |
| Project page | [basaanithanaveenkumar.github.io/Hale-VLM](https://basaanithanaveenkumar.github.io/Hale-VLM/) ([source](project-page/index.html)) |
| Documentation | [`docs/`](docs/README.md) — getting started, configuration, data, [known issues](docs/known-issues.md) |
| Architecture diagrams | [`docs/architecture.md`](docs/architecture.md) (Mermaid) |
| Blog | [Giving an 8B reasoning model eyes for about 1% of its parameters](docs/blog/2026-09-28-eyes-for-a-reasoning-llm.md) |
| Claude Code skills | [`.claude/skills/`](.claude/skills) — `hale-vlm-dev`, `hale-vlm-train`, `hale-vlm-datasets`, `hale-publish` |

## Vision projectors

`model.vision.projector_type` selects how vision features reach the LLM: `mlp`, `linear`, `qformer` (BLIP-2 style learned queries) or
`gated_cross_attention` (Flamingo style). See [docs/vision_connectors.md](docs/vision_connectors.md).
