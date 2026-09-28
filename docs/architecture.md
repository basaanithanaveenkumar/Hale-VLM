# Architecture

Mermaid diagrams (they render on GitHub). The same figures appear on the
[project page](../project-page/index.html) and in Section 3 of the [paper](../paper/main.tex).

## 1. Model

```mermaid
flowchart LR
  IMG["image 224×224"] --> SIG["SigLIP-B/16<br/>(frozen)"]
  SIG -->|"196 × 768"| PROJ["MLP projector<br/>768 → 2d → d<br/>(trainable)"]
  TXT["prompt with &lt;image&gt;"] --> TOK["tokenizer + LLM embeddings<br/>(frozen)"]
  PROJ --> MERGE["merge visual tokens<br/>at the &lt;image&gt; position"]
  TOK --> MERGE
  MERGE --> LLM["Qwen3-8B or DeepSeek-R1-Distill-Qwen-7B<br/>frozen base + LoRA r=16<br/>q,k,v,o,gate,up,down"]
  LLM --> LOSS["causal LM loss"]
```

## 2. What is trained

```mermaid
flowchart TB
  subgraph Frozen
    V["SigLIP vision tower<br/>freeze_encoder: true"]
    B["LLM base weights<br/>freeze_llm: true"]
  end
  subgraph Trainable
    P["projector (always)"]
    L["LoRA adapters<br/>use_lora: true"]
  end
  T["VLMTrainer"] -->|"optimizer(model.trainable_parameters())"| P
  T --> L
```

Trainable budget (computed from the backbone shapes): Qwen3-8B 43.6M LoRA + 39.9M projector ≈ 83.5M
(≈1.0%); DeepSeek-R1-Distill-Qwen-7B 40.4M + 31.2M ≈ 71.6M (≈0.9%).

## 3. Plugin wiring with HaleBlocks

```mermaid
flowchart LR
  Y["YAML (inherits: base.yaml)"] --> C["load_vlm_config → VLMRunConfig<br/>(strict pydantic)"]
  C --> R{"HaleBlocks registries"}
  BOOT["register_vlm_plugins()"] --> R
  R -->|"variant: qwen3_8b_vlm"| M["HaleVLM"]
  R -->|"loss: same variant name"| LS["_vlm_loss"]
  R -->|"trainer: vlm"| TR["VLMTrainer"]
  R -->|"vlm_dataset / vla_dataset"| DS["dataset adapters"]
  M & LS & TR & DS --> FIT["Trainer.fit()"]
```

## 4. Data flow

```mermaid
flowchart TB
  CAT["catalog.py stage presets<br/>vision · video · context"] --> SEQ["SequentialMultiDatasetStream<br/>stream ≤ max_samples_per_dataset<br/>prefetch next dataset"]
  VCAT["vla_catalog.py<br/>community · simulation · real_world"] --> CONV["vla_sample_to_vlm<br/>robotics_vlm_mode"]
  CONV --> MIX["mixed_registry"]
  SEQ --> MIX
  MIX --> ENC["encode: transform image(s),<br/>build prompt, tokenize (max_length)"]
  ENC --> BATCH["pixel_values, input_ids,<br/>attention_mask, labels"]
  BATCH --> MODEL["HaleVLM.forward"]
```

## Known issues

See [known-issues.md](known-issues.md). The merge step in diagram 1 currently
**overwrites** the text that follows `<image>` instead of inserting the visual tokens.
