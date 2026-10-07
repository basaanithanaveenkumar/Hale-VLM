"""VLA phase-tagged pretrain dataset registrations for Hale-VLM.

All adapters here are registered in VLA_PHASE_DATASETS (see vla/phase_registry.py),
NOT in the SmolVLA VLA_DATASETS registry, so existing tests are unaffected.

Pretrain corpus (12 datasets) — the standard VLA pretraining stack plus
aggregated packs and human-video-based sources:

Core / flagship cross-embodiment:
  open-x-embodiment, bridge-v2, fractal-rt1, bc-z,
  droid-v1, libero-pretrain

Aggregated / preprocessed packs:
  openEAI-dataset, lerobot-community-v3, robogene

Human-video and tactile pretraining:
  being-h0, agibot-world, h-tac-ttp
"""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase

# ---------------------------------------------------------------------------
# Phase 1 — PRETRAIN: core / flagship cross-embodiment corpora
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("open-x-embodiment")
class OpenXEmbodimentAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="open-x-embodiment",
        hf_path="jxu124/OpenX-Embodiment",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Open X-Embodiment — 22 robot types, ~2M demonstrations across "
            "kitchen, tabletop, and outdoor environments. "
            "Standard cross-embodiment pretraining corpus for OpenVLA, Octo, π₀."
        ),
        paper_reference="Open X-Embodiment Collaboration (2023)",
        episodes=2_000_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("bridge-v2")
class BridgeV2Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="bridge-v2",
        hf_path="lerobot/bridge_data_v2",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "BridgeData V2 — 2.26M steps on WidowX arm. "
            "Standard cross-embodiment eval; diverse household manipulation. "
            "lerobot/bridge_data_v2 (canonical LeRobot location)."
        ),
        paper_reference="Walke et al. (2023) Bridge Data V2",
        episodes=60_000,
    )


@register_vla_phase_dataset("fractal-rt1")
class FractalRT1Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="fractal-rt1",
        hf_path="lerobot/fractal",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "FractalData / RT-1 — 3.79M steps, Google RT-1 training corpus. "
            "Part of Open X-Embodiment; lerobot/fractal canonical location."
        ),
        paper_reference="Brohan et al. (2022) RT-1",
        episodes=130_000,
    )


@register_vla_phase_dataset("bc-z")
class BCZAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="bc-z",
        hf_path="lerobot/bc_z",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "BC-Z — 25K robot episodes across 100 tasks on Google's robot. "
            "Collected via human demonstration and teleoperation."
        ),
        paper_reference="Jang et al. (2022) BC-Z",
        episodes=25_000,
    )


@register_vla_phase_dataset("droid-v1")
class DroidV1Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="droid-v1",
        hf_path="lerobot/droid_1.0.1",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "DROID 1.0.1 — 76K trajectories, 564 scenes, 86 tasks, 50 operators. "
            "In-the-wild Franka Panda across many labs. High scene diversity; "
            "key pretrain complement to OXE (Khazatsky et al., 2024)."
        ),
        paper_reference="Khazatsky et al. (2024) DROID",
        episodes=76_000,
    )


@register_vla_phase_dataset("libero-pretrain")
class LiberoPretrainAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="libero-pretrain",
        hf_path="HuggingFaceVLA/libero",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "LIBERO (HuggingFaceVLA edition) — 130+ tasks, 5K+ episodes on Franka. "
            "Officially integrated into LeRobot v0.4.0. Used for instruction-following "
            "and multi-task generalisation pretraining."
        ),
        paper_reference="Liu et al. (2023) LIBERO",
        episodes=5_000,
    )


# ---------------------------------------------------------------------------
# Phase 1 — PRETRAIN: aggregated / preprocessed packs
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("openEAI-dataset")
class OpenEAIDatasetAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="openEAI-dataset",
        hf_path="OpenEAI/OpenEAI-Dataset",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "OpenEAI-Dataset — ~3.12 TB unified HDF5 corpus aggregating "
            "OXE + UMI Community + DROID + BC-Z. "
            "Single-format pretrain pack for cross-embodiment robot learning."
        ),
        paper_reference="OpenEAI (2024)",
        episodes=3_000_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("lerobot-community-v3")
class LeRobotCommunityV3Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="lerobot-community-v3",
        hf_path="lerobot/community_dataset_v3",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "LeRobot Community Dataset v3 — 791 datasets across 46 robot types, "
            "consolidated from 851 community sources. "
            "LeRobot Datasets v3.0 with chunked episodes for OXE-scale (>400 GB)."
        ),
        paper_reference="Cadène et al. (2024) LeRobot",
        episodes=5_000_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("robogene")
class RoboGeneAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="robogene",
        hf_path="X-Humanoid/RoboGene",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "RoboGene — VLA-specific pretraining dataset via diversity-driven "
            "agentic generation. Addresses limited scene variety and insufficient "
            "physical grounding in existing corpora. LeRobot-compatible."
        ),
        paper_reference="X-Humanoid (2024) RoboGene",
        episodes=500_000,
        trust_remote_code=True,
    )


# ---------------------------------------------------------------------------
# Phase 1 — PRETRAIN: human-video and tactile pretraining
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("being-h0")
class BeingH0Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="being-h0",
        hf_path="BeingBeyond/Being-H0",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Being-H0 post-training dataset — pretrained from large-scale human "
            "videos via explicit hand motion modelling. "
            "Bridges embodiment gap using egocentric 2D/3D cues."
        ),
        paper_reference="BeingBeyond (2024) Being-H0",
        episodes=100_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("agibot-world")
class AgibotWorldAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="agibot-world",
        hf_path="lerobot/xvla-agibot-world",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "AgiBot World — bimanual real-world manipulation used in X-VLA pretraining. "
            "High-quality dexterous tasks with rich scene diversity."
        ),
        paper_reference="AgiBot (2024) AgiBot World",
        episodes=200_000,
    )


@register_vla_phase_dataset("h-tac-ttp")
class HTacTTPAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="h-tac-ttp",
        hf_path="BeingBeyond/TTP",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "H-Tac TTP (Tactile Transformer Pretraining) — tactile sensor "
            "pretraining for dexterous manipulation. "
            "Captures contact dynamics unavailable in vision-only corpora."
        ),
        paper_reference="BeingBeyond (2024) H-Tac",
        episodes=50_000,
        trust_remote_code=True,
    )


__all__ = [
    "AgibotWorldAdapter",
    "BCZAdapter",
    "BeingH0Adapter",
    "BridgeV2Adapter",
    "DroidV1Adapter",
    "FractalRT1Adapter",
    "HTacTTPAdapter",
    "LeRobotCommunityV3Adapter",
    "LiberoPretrainAdapter",
    "OpenEAIDatasetAdapter",
    "OpenXEmbodimentAdapter",
    "RoboGeneAdapter",
]
