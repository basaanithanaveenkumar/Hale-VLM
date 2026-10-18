# Architecture

Mermaid diagrams (render on GitHub). They match the [project page](../project-page/index.html)
and Section 3 of the [paper](../paper/main.tex).
For known pipeline issues see [known-issues.md](known-issues.md).

---

## 1. End-to-end model

A frozen vision encoder converts the image to 196 visual tokens. A trainable MLP projector translates those tokens into the LLM's embedding space. The frozen LLM base (Qwen3-8B or DeepSeek-R1-Distill-Qwen-7B) with small LoRA adapters generates the answer.

```mermaid
flowchart LR
  subgraph INPUT["Inputs"]
    IMG["Image\n[B, 3, 224, 224]"]
    PROMPT["Prompt text\n(tokenised)"]
  end

  subgraph VISION["Vision tower (frozen)"]
    PATCH["Patch embedding\n14×14 = 196 patches\n3×16×16 → 768"]
    SIGLIP["SigLIP-B/16 transformer\n12 layers, heads=12, dim=768\nno CLS token — all 196 patches kept"]
    VIS_OUT["196 × 768 visual features"]
    PATCH --> SIGLIP --> VIS_OUT
  end

  subgraph PROJ["MLP projector (trainable)"]
    FC1["Linear 768 → 2d\nGELU activation"]
    FC2["Linear 2d → d\n(d = LLM hidden dim)"]
    FC1 --> FC2
  end

  subgraph LLM["Language model"]
    TOK_EMB["Token embedding\n(frozen)"]
    MERGE["Merge: replace &lt;image&gt; slot\nwith 196 visual tokens"]
    LAYERS["32 transformer layers\nfrozen base weights\n+ LoRA adapters (see §2)"]
    LM_HEAD["LM head → vocab logits"]
    TOK_EMB --> MERGE --> LAYERS --> LM_HEAD
  end

  IMG --> PATCH
  VIS_OUT --> FC1
  FC2 -->|"196 × d tokens"| MERGE
  PROMPT --> TOK_EMB
  LM_HEAD --> ANS["Answer tokens"]
```

---

## 2. LoRA adapter placement

Only ~1% of total parameters are updated: the MLP projector (always) and LoRA r=16 adapters on 7 linear modules in every transformer layer.

```mermaid
flowchart TB
  subgraph FROZEN["Frozen — never updated during training"]
    SIG["SigLIP-B/16 weights\n(87M params)"]
    LLM_BASE["LLM base weights\nQwen3-8B: ~8B params\nDeepSeek-R1-7B: ~7B params"]
  end

  subgraph TRAINED["Trainable — ~1% of total"]
    PROJECTOR["MLP projector\n768 → 2d → d\n(~40M params)"]
    subgraph LORA["LoRA r=16 on every layer"]
      Q["q_proj  ΔW = BA, rank 16"]
      K["k_proj  ΔW = BA, rank 16"]
      V["v_proj  ΔW = BA, rank 16"]
      O["o_proj  ΔW = BA, rank 16"]
      GA["gate_proj  ΔW = BA, rank 16"]
      UP["up_proj  ΔW = BA, rank 16"]
      DN["down_proj  ΔW = BA, rank 16"]
    end
  end

  subgraph COUNTS["Trainable parameter counts"]
    QW["Qwen3-8B: 43.6M LoRA + 39.9M proj = 83.5M (≈1.0%)"]
    DW["DeepSeek-R1-7B: 40.4M LoRA + 31.2M proj = 71.6M (≈0.9%)"]
  end
```

---

## 3. Plugin wiring (HaleBlocks registry)

```mermaid
flowchart LR
  YAML["YAML config\n(inherits: base.yaml)\nvariant: qwen3_8b_vlm"]
  BOOT["register_vlm_plugins()"]
  REG{"HaleBlocks VariantRegistry"}

  YAML & BOOT --> REG
  REG -->|"variant: qwen3_8b_vlm"| M["HaleVLM model"]
  REG -->|"loss: qwen3_8b_vlm"| LS["_vlm_loss function"]
  REG -->|"trainer: vlm"| TR["VLMTrainer"]
  REG -->|"vlm_dataset / vla_dataset"| DS["dataset adapters"]

  M & LS & TR & DS --> FIT["Trainer.fit()"]
```

---

## 4. SmolVLM data pipeline (three stages)

```mermaid
flowchart TB
  subgraph PRETRAIN["Stage 1 — vision pretraining\n(image–caption pairs)"]
    D1["LAION-COCO\n(600M image-text pairs)"]
    D2["COYO-700M\n(700M image-text pairs)"]
    D3["ShareGPT4V\n(100K high-quality captions)"]
  end

  subgraph FINETUNE["Stage 2 — instruction tuning\n(multi-turn Q&A)"]
    D4["LLaVA-Instruct-150K"]
    D5["TextVQA, VQAv2, GQA"]
    D6["DocVQA, ChartQA, InfoVQA"]
    D7["COCO-QA, ScienceQA"]
  end

  subgraph LONGCTX["Stage 3 — long context\n(high-resolution & multi-image)"]
    D8["LLaVA-OneVision"]
    D9["Multi-image conversations"]
  end

  subgraph LOADER["SequentialMultiDatasetStream"]
    SEQ["Stream dataset 1 up to max_samples\nwarm up dataset 2 in background\nthen switch — no shuffle buffer"]
  end

  PRETRAIN & FINETUNE & LONGCTX --> LOADER

  subgraph ENCODE["Sample encoding"]
    TR_IMG["transform image → [3, 224, 224]"]
    BUILD["build ChatML prompt\nwith &lt;image&gt; placeholder"]
    TOK["tokenise (max_length=2048)\nset labels=-100 on prompt tokens"]
    OUT["pixel_values, input_ids,\nattention_mask, labels"]
  end

  LOADER --> TR_IMG & BUILD --> TOK --> OUT
```

---

## 5. VLA robot data → VLM samples

```mermaid
flowchart LR
  subgraph VLA["SmolVLA robot dataset\n(540 community + sim + real-world datasets)"]
    EPI["Episode\n(frames, language annotation,\naction trajectory)"]
  end

  subgraph CONVERT["vla_sample_to_vlm()"]
    MODE{"robotics_vlm_mode"}
    PRETRAIN_M["pretrain mode\nimage + caption\n(no Q&A format)"]
    FINETUNE_M["finetune mode\nQ: 'What is the robot doing?'\nA: annotation text"]
    INSTR_M["instruction mode\nQ: 'How should the robot move?'\nA: trajectory description"]
  end

  EPI --> MODE
  MODE --> PRETRAIN_M & FINETUNE_M & INSTR_M

  MIX["mixed_registry\n(VLM + converted VLA samples)"] 
  PRETRAIN_M & FINETUNE_M & INSTR_M --> MIX
  MIX --> LOADER["VLMTrainer DataLoader"]
```

---

## 6. Three-phase VLA training pipeline

```mermaid
flowchart TB
  subgraph PRETRAIN_P["Phase 1 — VLA Pretraining"]
    P1["Egocentric video datasets\n(Ego4D, VITRA, EgoVLA)\nlearn from human demonstrations"]
    P2["Synthetic robot corpora\n(SynGrasp-1B, RoboCasa)\nlearn diverse manipulation"]
    P3["Community robot datasets\n(OpenX-Embodiment, Bridge, etc.)"]
  end

  subgraph MIDTRAIN_P["Phase 2 — Mid-train alignment\n(no action labels needed)"]
    M1["Embodied VLM datasets\n(RefSpatial, RoboPoint, Robo2VLM)\nlearn spatial language grounding"]
    M2["Navigation datasets\n(VLN-R2R, EmbSpatial-Bench)\nlearn scene understanding"]
  end

  subgraph POSTTRAIN_P["Phase 3 — Post-train preference\n(local robot collection)"]
    A1["FlowPRO rollback pairs\n(successful vs failed trajectories)"]
    A2["APO intervention labels\n(human corrects the robot mid-task)"]
    A3["LfH hindsight relabeling\n(reinterpret failed attempts as new goals)"]
    A4["DAgger corrections\n(human takes over when policy is uncertain)"]
  end

  PRETRAIN_P --> MIDTRAIN_P --> POSTTRAIN_P
  POSTTRAIN_P --> POLICY["Final VLA policy\n(text + action + world-model outputs)"]
```

---

## 7. Loss computation

```mermaid
flowchart LR
  LOGITS["logits\n[B, S, V]"]
  LABELS["labels\n[B, S]\n(-100 on prompt tokens)"]
  PRED["pixel_values via projector\nvisual tokens merged at &lt;image&gt; position"]
  CE["cross-entropy loss\nonly on answer tokens\n(labels ≠ -100)"]

  LOGITS & LABELS --> CE --> TOTAL["total loss\n(backprop through LoRA + projector only)"]
  PRED --> CE
```

---

## 8. Package structure (class diagram)

```mermaid
classDiagram
  class VLMRunConfig {
    +VLMModelConfig model
    +VLMDataConfig data
  }
  class HaleVLM {
    +encode_images(pixel_values)
    +merge_image_embeddings(...)
    +forward(input_ids, pixel_values, labels)
    +from_config(cfg)$
  }
  class VisionTower
  class VisionProjector
  class LLMBackbone
  class VISION_CONNECTORS {
    <<Registry>>
    qformer
    gated_cross_attention
  }
  class QFormer
  class GatedCrossAttentionProjector
  class VLMDataAdapter {
    <<abstract>>
    +open_stream()
    +normalize(row)
    +iter_samples()
  }
  class VLADataAdapter {
    <<abstract>>
  }
  class DATASETS {
    <<Registry>>
  }
  class PHASE_DATASETS {
    <<Registry>>
  }
  class VLA_DATASETS {
    <<Registry>>
  }
  class VLA_PHASE_DATASETS {
    <<Registry>>
  }
  class VLMTrainer
  class VLMMetricBase {
    <<abstract>>
    +reset()
    +update()
    +compute()
  }

  VLMRunConfig --> HaleVLM : from_config
  HaleVLM *-- VisionTower
  HaleVLM *-- VisionProjector
  HaleVLM *-- LLMBackbone
  VisionProjector ..> VISION_CONNECTORS : build_vision_connector
  VISION_CONNECTORS o-- QFormer
  VISION_CONNECTORS o-- GatedCrossAttentionProjector
  DATASETS o-- VLMDataAdapter
  PHASE_DATASETS o-- VLMDataAdapter
  VLA_DATASETS o-- VLADataAdapter
  VLA_PHASE_DATASETS o-- VLADataAdapter
  VLMTrainer ..> HaleVLM : optimises trainable parameters
  VLMTrainer ..> VLMMetricBase : evaluation
```

Registries are `hale_core` `NamedRegistry` instances (duplicate keys raise, unknown keys list the
registered names). `get_dataset` and `get_vla_dataset` fall back from the SmolVLM / SmolVLA
registries to the phase registries, so a config can name a dataset without knowing which holds it.

## 9. Training run (sequence diagram)

```mermaid
sequenceDiagram
  participant CLI as hale-vlm-train
  participant CL as load_vlm_config
  participant B as register_vlm_plugins
  participant L as train_vlm (hale_core train)
  participant DM as MultimodalDataModule
  participant S as SequentialMultiDatasetStream
  participant M as HaleVLM
  participant T as VLMTrainer

  CLI->>CL: path, --cfg-options
  CL-->>CLI: VLMRunConfig (inherits + overrides, validated)
  CLI->>L: cfg
  L->>B: import registering modules
  L->>M: build_vlm(cfg)
  L->>DM: setup(cfg, tokenizer)
  L->>T: fit()
  loop every step
    T->>DM: next batch
    DM->>S: iterate datasets (prefetch next)
    S-->>DM: VLMSample
    DM-->>T: pixel_values, input_ids, labels
    T->>M: forward -> loss
    T->>T: backward on trainable parameters only
  end
```

## 10. Error hierarchy

```mermaid
classDiagram
  HaleCoreError <|-- HaleVLMError
  HaleVLMError <|-- VLMConfigError
  HaleVLMError <|-- VLMInputError
  HaleVLMError <|-- VLMDatasetError
  VLMDatasetError <|-- DatasetUnavailableError
  HaleVLMError <|-- UnsupportedOperationError
  ConfigError <|-- VLMConfigError
  DataError <|-- VLMDatasetError
```

Each VLM error also derives from the built-in the call site raised before (`ValueError`,
`NotImplementedError`), so existing `except` clauses keep working.

## 11. Sources and versions

Resolved in `uv.lock` at the time of this revision: transformers 5.16.1, torch 2.13.0, peft 0.20.0,
pydantic 2.13.5, datasets 5.0.1, torchvision 0.28.0, accelerate 1.14.0, loguru 0.7.3. Model
families follow their Hugging Face model cards (Qwen3, DeepSeek-R1-Distill-Qwen, SigLIP); the
Q-Former follows BLIP-2 (Li et al., 2023), the gated cross-attention connector Flamingo
(Alayrac et al., 2022), the dataset catalogs SmolVLM (arXiv:2504.05299) and SmolVLA
(arXiv:2506.01844).
