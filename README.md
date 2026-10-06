# Hale-VLM

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
