# Configuration

`VLMRunConfig` extends the HaleBlocks `RunConfig`. Unknown keys are rejected. YAML files
can `inherits:` a parent file and override fields.

## `model.vision` (`VisionConfig`)

| Field | Default | Meaning |
|---|---|---|
| `encoder` | `siglip` | `siglip` or `clip` |
| `model_id` | `google/siglip-base-patch16-224` | HF id |
| `image_size` | 224 | |
| `freeze_encoder` | true | freeze the vision tower |
| `projector_type` | `mlp` | `mlp` (d_v → hidden → d) or `linear` |
| `projector_hidden_dim` | null → 2·d_llm | MLP hidden width |
| `projector_dropout` | 0.0 | |
| `num_image_tokens` | 256 (schema) / 196 (`base.yaml`) | visual tokens kept per image |

## `model.llm` (`LLMConfig`)

| Field | Default | Meaning |
|---|---|---|
| `backbone` | `qwen3-8b` | `qwen3-8b`, `deepseek-r1-qwen-7b` or `custom` |
| `model_id` | `Qwen/Qwen3-8B` | HF id |
| `dtype` | `bfloat16` | |
| `freeze_llm` | true | freeze base weights |
| `use_lora` | true | wrap with PEFT LoRA |
| `lora_r`, `lora_alpha`, `lora_dropout` | 16, 32, 0.05 | |
| `lora_target_modules` | null → q,k,v,o,gate,up,down | |
| `attn_implementation` | `sdpa` | |
| `image_token` | `<image>` | added to the tokenizer if missing |
| `reasoning_mode` | false | |

## `data` (`VLMDataConfig`)

| Field | Default | Meaning |
|---|---|---|
| `source` | `huggingface` | `huggingface`, `overfit`, `registry`, `vla_registry`, `mixed_registry` |
| `registry_stage` | `all` | `all`, `vision`, `video`, `context` |
| `registry_datasets` | all 19 | explicit list |
| `vla_registry_stage` | `all` | `all`, `community`, `simulation`, `real_world` |
| `vla_registry_datasets` | all | explicit list |
| `robotics_vlm_mode` | `off` | `off`, `pretraining`, `finetuning`, `instruction_tuning` |
| `max_samples_per_dataset` | 256 | streaming cap per dataset |
| `streaming`, `prefetch_workers`, `max_video_frames` | true, 2, 8 | |

## `train` (from HaleBlocks, as used in `base.yaml`)

`steps: 1000`, `batch_size: 2`, `lr: 2.0e-5`, `grad_clip: 1.0`, `resume: false`,
`checkpoint_every_epoch: true`.
