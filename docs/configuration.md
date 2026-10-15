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
| `projector_type` | `mlp` | `mlp` (d_v → hidden → d), `linear`, `qformer` or `gated_cross_attention` (see [vision_connectors.md](vision_connectors.md)) |
| `projector_hidden_dim` | null → 2·d_llm | MLP hidden width |
| `projector_dropout` | 0.0 | |
| `num_image_tokens` | 256 (schema) / 196 (`base.yaml`) | visual tokens kept per image (`mlp`/`linear`; the token connectors use their own count) |
| `qformer.num_queries` | 32 | learned queries, used when `projector_type: qformer` |
| `gated_cross_attention.num_latents` | 64 | learned latents, used when `projector_type: gated_cross_attention` |

## `model.llm` (`LLMConfig`)

| Field | Default | Meaning |
|---|---|---|
| `backbone` | `qwen3-8b` | any name in HaleBlocks' registry (`hale_core.nn.llm.list_llm_backbones()`, 47 Hugging Face + 2 HaloDiffusionLLM), or `custom` |
| `model_id` | `Qwen/Qwen3-8B` | HF id; left at the default, the backbone's registered repo is used |
| `checkpoint` | null | HaloDiffusionLLM checkpoint for `hale-*` backbones |
| `lora_train_embeddings` | false | also train embeddings / LM head (learn `<image>`) |
| `gradient_checkpointing` | false | |
| `dtype` | `bfloat16` | |
| `freeze_llm` | true | freeze base weights |
| `use_lora` | true | wrap with PEFT LoRA |
| `lora_r`, `lora_alpha`, `lora_dropout` | 16, 32, 0.05 | |
| `lora_target_modules` | null → q,k,v,o,gate,up,down | |
| `attn_implementation` | `sdpa` | |
| `image_token` | `<image>` | added to the tokenizer if missing |
| `reasoning_mode` | false | |

Registered backbones include text LLMs (SmolLM2/3, Qwen2.5/3, Llama 3.x, Gemma 2/3, Phi-3.5/4-mini(-reasoning),
Mistral 7B, Granite 4.2, DeepSeek-R1-Distill). They also include the **text towers of natively multimodal 2026
models**: Qwen3.5 0.8B/2B/4B (base + instruct), Gemma 4 E2B/E4B, Ministral 3 3B/8B, Cosmos-Reason2-2B,
LFM2.5-VL 450M/1.6B/3B and SmolVLM2. For those, only the decoder and LM head are loaded, and the VLM attaches its own
vision encoder and projector. Hybrid decoders (Qwen3.5 linear attention, LFM2.5 convolutions, Gemma 4 sliding
window) are supported. Gemma 4 receives the token ids alongside the spliced image embeddings automatically.
Examples: `configs/examples/qwen3_5_4b_base_lora.yaml`, `gemma4_e4b_lora.yaml`, `cosmos_reason2_2b_lora.yaml`.
Full table, selection rule and picks: HaleBlocks `docs/llm_backbones.md`. Known limitation: `chat` generation
starting from spliced image embeddings does not work with `gemma4-*` (Gemma 4 cannot recover its per-layer token
ids from them). Training, evaluation and text-only chat work.

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
