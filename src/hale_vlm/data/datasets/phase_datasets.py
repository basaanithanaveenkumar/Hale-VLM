"""Phase-tagged VLM dataset registrations for Hale-VLM (pretrain/mid-train/post-train).

Registered in PHASE_DATASETS (see data/phase_registry.py), not DATASETS.
"""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage

# ---------------------------------------------------------------------------
# Phase 1 — PRETRAIN
# ---------------------------------------------------------------------------


@register_phase_dataset("laion-aesthetics-v2-5plus")
class LAIONAestheticsAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="laion-aesthetics-v2-5plus",
        hf_path="laion/laion2B-en-aesthetic",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="LAION-2B-en aesthetic subset (≥5.0); ~600M image-text pairs.",
        paper_reference="Schuhmann et al. (2022) LAION-5B",
        image_fields=("image", "url"),
        text_fields=("caption", "text"),
        conversation_field=None,
    )


@register_phase_dataset("cc3m")
class CC3MAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="cc3m",
        hf_path="pixparse/cc3m-wds",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="Conceptual Captions 3M — 3.3M image-alt-text pairs.",
        paper_reference="Sharma et al. (2018) CC3M",
        image_fields=("jpg", "image"),
        text_fields=("txt", "caption"),
        conversation_field=None,
    )


@register_phase_dataset("cc12m")
class CC12MAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="cc12m",
        hf_path="pixparse/cc12m-wds",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="Conceptual Captions 12M — ~12M image-text pairs.",
        paper_reference="Changpinyo et al. (2021) CC12M",
        image_fields=("jpg", "image"),
        text_fields=("txt", "caption"),
        conversation_field=None,
    )


@register_phase_dataset("datacomp-1b")
class DataComp1BAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="datacomp-1b",
        hf_path="mlfoundations/datacomp_1b",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="DataComp-1B CommonPool — 1.28B CLIP-filtered image-text pairs.",
        paper_reference="Gadre et al. (2023) DataComp",
        image_fields=("image", "url"),
        text_fields=("caption", "text"),
        conversation_field=None,
        trust_remote_code=True,
    )


@register_phase_dataset("wit")
class WITAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="wit",
        hf_path="google/wit",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="Wikipedia-based Image Text (WIT) — 37.6M pairs.",
        paper_reference="Srinivasan et al. (2021) WIT",
        image_fields=("image_url", "image"),
        text_fields=("caption_reference_description", "caption_attribution_description"),
        conversation_field=None,
    )


@register_phase_dataset("redcaps")
class RedCapsAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="redcaps",
        hf_path="red_caps",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="RedCaps — 12M human-written image-text pairs from Reddit.",
        paper_reference="Desai et al. (2021) RedCaps",
        image_fields=("image", "image_url"),
        text_fields=("caption", "raw_caption"),
        conversation_field=None,
    )


# ---------------------------------------------------------------------------
# Phase 2 — MID_TRAIN
# ---------------------------------------------------------------------------


@register_phase_dataset("llava-pretrain-558k")
class LLaVAPretrain558KAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="llava-pretrain-558k",
        hf_path="liuhaotian/LLaVA-Pretrain",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="LLaVA pretraining 558K BLIP-generated captions from CC3M.",
        paper_reference="Liu et al. (2023) LLaVA-1.5",
        image_fields=("image",),
        text_fields=("caption", "text", "conversations"),
        conversation_field="conversations",
    )


@register_phase_dataset("sharegpt4v-pt")
class ShareGPT4VPTAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="sharegpt4v-pt",
        hf_path="Lin-Chen/ShareGPT4V",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="ShareGPT4V 1.2M GPT-4V captions.",
        paper_reference="Chen et al. (2023) ShareGPT4V",
        image_fields=("image", "images"),
        text_fields=("caption", "conversations"),
        conversation_field="conversations",
    )


@register_phase_dataset("blip-laion-cc-sbu-558k")
class BLIPPretrainAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="blip-laion-cc-sbu-558k",
        hf_path="BleachNick/BLIP_laion_cc_sbu_558k",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="BLIP pretraining LAION+CC3M+SBU mixture.",
        paper_reference="Li et al. (2022) BLIP",
        image_fields=("image",),
        text_fields=("caption", "text"),
        conversation_field=None,
    )


@register_phase_dataset("recap-datacomp-1b")
class RecapDataComp1BAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="recap-datacomp-1b",
        hf_path="BAAI/Recap-DataComp-1B",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="DataComp-1B re-captioned with LLaVA-1.5.",
        paper_reference="Li et al. (2024) Recap-DataComp",
        image_fields=("image", "url"),
        text_fields=("recaption", "caption", "text"),
        conversation_field=None,
        trust_remote_code=True,
    )


@register_phase_dataset("allava-vflan")
class AllaVAVFLANAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="allava-vflan",
        hf_path="FreedomIntelligence/ALLaVA-4V",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="ALLaVA VFLAN ~1.5M reason-annotated visual QA.",
        paper_reference="Chen et al. (2024) ALLaVA",
        image_fields=("image", "images"),
        text_fields=("conversations", "caption"),
        conversation_field="conversations",
    )


# ---------------------------------------------------------------------------
# Phase 3 — POST_TRAIN
# ---------------------------------------------------------------------------


@register_phase_dataset("llava-instruct-665k")
class LLaVAInstruct665KAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="llava-instruct-665k",
        hf_path="liuhaotian/LLaVA-Instruct-150K",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="LLaVA-1.5 SFT 665K instruction-following conversations.",
        paper_reference="Liu et al. (2023) LLaVA-1.5",
        image_fields=("image",),
        text_fields=("conversations",),
        conversation_field="conversations",
    )


@register_phase_dataset("textvqa")
class TextVQAAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="textvqa",
        hf_path="Howard-hou/TextVQA_0.5.1",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="TextVQA — 45K VQA questions on text in natural images.",
        paper_reference="Singh et al. (2019) TextVQA",
        image_fields=("image",),
        text_fields=("question", "answers"),
        conversation_field=None,
    )


@register_phase_dataset("scienceqa")
class ScienceQAAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="scienceqa",
        hf_path="derek-thomas/ScienceQA",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="ScienceQA — 21K multimodal science MCQ.",
        paper_reference="Lu et al. (2022) ScienceQA",
        image_fields=("image",),
        text_fields=("question", "choices", "solution"),
        conversation_field=None,
    )


@register_phase_dataset("chartqa")
class ChartQAAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="chartqa",
        hf_path="HuggingFaceM4/ChartQA",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="ChartQA — 32K QA on charts requiring numerical reasoning.",
        paper_reference="Masry et al. (2022) ChartQA",
        image_fields=("image",),
        text_fields=("question", "answer"),
        conversation_field=None,
    )


@register_phase_dataset("infographics-vqa")
class InfographicsVQAAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="infographics-vqa",
        hf_path="lmms-lab/InfoVQA",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="InfographicsVQA — 30K QA on infographic images.",
        paper_reference="Mathew et al. (2022) InfographicsVQA",
        image_fields=("image",),
        text_fields=("question", "answers"),
        conversation_field=None,
    )


@register_phase_dataset("seed-bench")
class SEEDBenchAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="seed-bench",
        hf_path="AILab-CVC/SEED-Bench",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="SEED-Bench — 19K MCQ across 12 evaluation dimensions.",
        paper_reference="Li et al. (2023) SEED-Bench",
        image_fields=("image",),
        text_fields=("question", "choice_a", "choice_b", "choice_c", "choice_d"),
        conversation_field=None,
        trust_remote_code=True,
    )


@register_phase_dataset("llava-plus")
class LLaVAPlusAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="llava-plus",
        hf_path="zhaohengyuan1/llava_plus_stage2",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="LLaVA-Plus skill-augmented instruction-following data.",
        paper_reference="Liu et al. (2023) LLaVA-Plus",
        image_fields=("image", "images"),
        text_fields=("conversations",),
        conversation_field="conversations",
    )
