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
