# Hale-VLM documentation

Hale-VLM builds vision-language models on dense 7–8B LLMs (Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B)
with a frozen SigLIP encoder, a trainable MLP projector and LoRA adapters. It plugs into
[HaleBlocks](https://github.com/basaanithanaveenkumar/HaleBlocks) registries.

| Page | Contents |
|---|---|
| [Getting started](getting-started.md) | install, configs, training, chat |
| [Architecture](architecture.md) | Mermaid diagrams: model, trainability, plugin wiring, data flow |
| [Configuration](configuration.md) | `VLMRunConfig` fields |
| [Data](data.md) | SmolVLM and SmolVLA registries, streaming mixer, robotics-to-VLM modes |
| [Known issues](known-issues.md) | input-pipeline issues to fix before trusting results |
| [Blog](blog/README.md) | long-form posts |

Also: the [paper](../paper/main.tex) and the [project page](../project-page/index.html).
