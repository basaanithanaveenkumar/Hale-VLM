---
name: hale-vlm-train
description: Configure and run Hale-VLM training and chat — YAML inheritance, backbone choice (Qwen3-8B vs DeepSeek-R1-Distill-Qwen-7B), LoRA/freeze policy, data sources (overfit, SmolVLM registry stages, SmolVLA robotics, mixed), and the chat CLI. Use when asked to train, fine-tune, change what is trainable, or pick a data mixture.
---

# Training Hale-VLM

## Entry points

```bash
uv run hale-vlm-train configs/qwen3_8b_overfit.yaml
uv run hale-vlm-chat configs/base.yaml --image path/to/img.jpg --prompt "What is in this image?"
```

## Configs

YAML files use HaleBlocks inheritance (`inherits: base.yaml`) and deep-merge the child over
the parent.

| Config | Use |
|---|---|
| `base.yaml` | Qwen3-8B, SigLIP-base (196 image tokens), LoRA, overfit data, 1000 steps, lr 2e-5 |
| `qwen3_8b_overfit.yaml` | smoke test |
| `deepseek_r1_qwen_7b.yaml` | reasoning backbone |
| `smolvlm_all.yaml` / `smolvlm_vision.yaml` / `smolvlm_video.yaml` / `smolvlm_context.yaml` | SmolVLM mixture by stage (`registry_mix.yaml` is an alias of `_all`) |
| `smolvla_*.yaml` | SmolVLA robotics registry (all, community, simulation, real_world) |
| `vlm_with_robotics_pretrain.yaml` / `_finetune.yaml` | VLM data + robotics frames converted to VLM samples |

## What is trained

| Setting | Default | Effect |
|---|---|---|
| `model.vision.freeze_encoder` | true | set false to fine-tune SigLIP too |
| projector | always trainable | MLP `768 → 2·d → d` |
| `model.llm.freeze_llm` | true | base weights frozen |
| `model.llm.use_lora` | true | LoRA r=16, α=32, dropout 0.05 on q,k,v,o,gate,up,down |
| `model.llm.lora_target_modules` | null | override targets |

`VLMTrainer` passes only `trainable_parameters()` to the optimiser. Check the
`trainable components: ...` log line at start-up to confirm the policy.

Approximate trainable parameters with the defaults (computed from the backbone shapes):
Qwen3-8B about 43.6M LoRA + 39.9M projector; DeepSeek-R1-Distill-Qwen-7B about 40.4M
LoRA + 31.2M projector.

## Data sources (`data.source`)

| Value | Meaning |
|---|---|
| `overfit` | repeats `overfit_text` `n_overfit_copies` times (sanity check) |
| `huggingface` | a single HF dataset from `data.*` fields |
| `registry` | SmolVLM registry; `registry_stage: all\|vision\|video\|context` or explicit `registry_datasets` |
| `vla_registry` | SmolVLA registry; `vla_registry_stage: all\|community\|simulation\|real_world` |
| `mixed_registry` | VLM registry plus robotics frames |

`robotics_vlm_mode: off|pretraining|finetuning|instruction_tuning` converts robotics frames into
image + instruction samples (`data/robotics_vlm.py`). Registry data is streamed dataset by
dataset with background warm-up of the next one. `max_samples_per_dataset` bounds disk use.

## Tips

- Start with `qwen3_8b_overfit.yaml`. The loss should approach 0 within a few hundred steps.
- `model.llm.reasoning_mode: true` is intended for the DeepSeek-R1 backbone.
- Watch the known issues in the `hale-vlm-dev` skill (image merge overwrites text; loss
  covers the whole prompt) before interpreting results.
