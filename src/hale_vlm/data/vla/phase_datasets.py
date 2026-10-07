"""VLA phase-tagged dataset registrations for Hale-VLM (all three phases).

All adapters here are registered in VLA_PHASE_DATASETS (see vla/phase_registry.py),
NOT in the SmolVLA VLA_DATASETS registry, so existing tests are unaffected.

Phase 1 — PRETRAIN (18 datasets):
  Core cross-embodiment:
    open-x-embodiment, bridge-v2, fractal-rt1, bc-z,
    droid-v1, libero-pretrain
  Aggregated packs:
    openEAI-dataset, lerobot-community-v3, robogene
  Human-video / tactile:
    being-h0, agibot-world, h-tac-ttp
  Synthetic / simulation:
    syngrasp-1b, robocasa
  Egocentric human video (cross-embodiment bridge):
    ego4d, vitra, egovla

Phase 2 — MID_TRAIN (5 datasets, embodied VLM — no action labels):
  refspatial, embspatial-bench, robo2vlm, robopoint, vln-r2r

Phase 3 — POST_TRAIN (4 local-collection datasets):
  flowpro-pairs, apo-interventions, hindsight-relabeled, dagger-corrections
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


# ---------------------------------------------------------------------------
# Phase 1 — PRETRAIN: synthetic / simulation corpora
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("syngrasp-1b")
class SynGrasp1BAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="syngrasp-1b",
        hf_path="GraspVLA/SynGrasp-1B",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "SynGrasp-1B — 1B procedurally-generated grasp scenes with randomised "
            "objects, lighting, and camera poses (GraspVLA). "
            "Provides robust geometric pretraining at scale."
        ),
        paper_reference="GraspVLA (2025) SynGrasp-1B",
        episodes=1_000_000_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("robocasa")
class RoboCasaAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="robocasa",
        hf_path="lerobot/robocasa",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "RoboCasa — scalable household manipulation rollouts across diverse "
            "kitchen and living room layouts. "
            "Enables generalist policy pretraining at simulation scale."
        ),
        paper_reference="Nasiriany et al. (2024) RoboCasa",
        episodes=100_000,
    )


# ---------------------------------------------------------------------------
# Phase 1 — PRETRAIN: egocentric human video (cross-embodiment bridge)
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("ego4d")
class Ego4DAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="ego4d",
        hf_path="facebook/ego4d",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.HUMAN,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Ego4D — 3,600 hours of first-person video from 931 participants across "
            "74 worldwide scenarios. Low-cost VLA pretraining source; "
            "human-to-robot transfer via diverse egocentric observations."
        ),
        paper_reference="Grauman et al. (2022) Ego4D",
        episodes=9_600,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("vitra")
class VITRAAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="vitra",
        hf_path="VITRA-Dataset/VITRA",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.HUMAN,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "VITRA (ICRA 2026) — converts in-the-wild human hand videos into "
            "(image, instruction, action) tuples for VLA pretraining. "
            "Explicitly bridges the embodiment gap via egocentric hand motion parsing."
        ),
        paper_reference="VITRA (2026) ICRA",
        episodes=500_000,
    )


@register_vla_phase_dataset("egovla")
class EgoVLAAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="egovla",
        hf_path="EgoVLA/EgoVLA",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.HUMAN,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "EgoVLA — large-scale egocentric human video corpus for VLA pretraining. "
            "Overcomes robot data scarcity via human-to-robot cross-embodiment transfer."
        ),
        paper_reference="EgoVLA (2025)",
        episodes=1_000_000,
    )


# ---------------------------------------------------------------------------
# Phase 2 — MID_TRAIN: embodied-oriented VLM data (no action labels)
# Key reference: EmbodiedMidtrain (2026) proximity-based data engine
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("refspatial")
class RefSpatialAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="refspatial",
        hf_path="RefSpatial/RefSpatial",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "RefSpatial — spatial referring and reasoning dataset for embodied agents. "
            "No action labels; trains VLM spatial grounding needed for robot control."
        ),
        paper_reference="RefSpatial (2025)",
        episodes=100_000,
    )


@register_vla_phase_dataset("embspatial-bench")
class EmbSpatialBenchAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="embspatial-bench",
        hf_path="EmbSpatial/EmbSpatial-Bench",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "EmbSpatial-Bench — embodied spatial understanding VQA. "
            "Tests relative positions, distances, directions in 3D scene context. "
            "Used as mid-train alignment data (EmbodiedMidtrain 2026)."
        ),
        paper_reference="EmbSpatial (2024)",
        episodes=10_000,
    )


@register_vla_phase_dataset("robo2vlm")
class Robo2VLMAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="robo2vlm",
        hf_path="Robo2VLM/Robo2VLM",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "Robo2VLM — robotic VQA generated from robot observation trajectories. "
            "No action labels; aligns VLM priors to robot-camera viewpoints and "
            "manipulation contexts (500K Q&A pairs)."
        ),
        paper_reference="Robo2VLM (2025)",
        episodes=500_000,
    )


@register_vla_phase_dataset("robopoint")
class RoboPointAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="robopoint",
        hf_path="wentao-yuan/robopoint-data",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "RoboPoint — spatial affordance prediction: given image + instruction, "
            "predict the target 2D manipulation point. "
            "Trains spatial reasoning without requiring low-level action labels."
        ),
        paper_reference="Yuan et al. (2024) RoboPoint",
        episodes=600_000,
    )


@register_vla_phase_dataset("vln-r2r")
class VLNR2RAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="vln-r2r",
        hf_path="prs-eth/room_across_the_room",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "R2R (Room-to-Room) VLN — vision-and-language navigation trajectories "
            "in photorealistic Matterport3D environments. "
            "Trajectory-centric supervision for spatial grounding and path following."
        ),
        paper_reference="Anderson et al. (2018) R2R",
        episodes=22_000,
    )


# ---------------------------------------------------------------------------
# Phase 3 — POST_TRAIN: preference / DPO / DAgger / offline-RL datasets
# These are locally-collected; hf_path is None and episodes is None.
# The registry skips them when no local_path is supplied.
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("flowpro-pairs")
class FlowPROPairsAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="flowpro-pairs",
        hf_path=None,
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "FlowPRO rollback preference pairs — a single operator intervention "
            "yields a (winner, loser) trajectory pair via operator-chosen rollback "
            "horizon. No separate positive/negative recordings needed. "
            "Requires local robot deployment to collect."
        ),
        paper_reference="FlowPRO (2025)",
        episodes=None,
        requires_local_collection=True,
    )


@register_vla_phase_dataset("apo-interventions")
class APOInterventionsAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="apo-interventions",
        hf_path=None,
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "APO (Action Preference Optimisation) — human-in-the-loop intervention "
            "preference data with adaptive reweighting. "
            "Learns from sub-optimal correction trajectories. Requires local collection."
        ),
        paper_reference="APO (2025)",
        episodes=None,
        requires_local_collection=True,
    )


@register_vla_phase_dataset("hindsight-relabeled")
class HindsightRelabeledAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="hindsight-relabeled",
        hf_path=None,
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "LfH (Learning from Hindsight) hindsight-relabeled rollouts — "
            "failed trajectories are scored against tasks actually achieved and "
            "reused as positives. Useful when early policies rarely succeed."
        ),
        paper_reference="LfH (2024)",
        episodes=None,
        requires_local_collection=True,
    )


@register_vla_phase_dataset("dagger-corrections")
class DAggerCorrectionsAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="dagger-corrections",
        hf_path=None,
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "DAgger-style interactive correction trajectories — human operator "
            "takes over at failure modes, producing corrective demos that cover "
            "the state distribution induced by the current policy."
        ),
        paper_reference="Ross et al. (2011) DAgger",
        episodes=None,
        requires_local_collection=True,
    )


__all__ = [
    # Phase 1 — PRETRAIN: core cross-embodiment
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
    # Phase 1 — PRETRAIN: synthetic / simulation
    "RoboCasaAdapter",
    "SynGrasp1BAdapter",
    # Phase 1 — PRETRAIN: egocentric human video
    "Ego4DAdapter",
    "EgoVLAAdapter",
    "VITRAAdapter",
    # Phase 2 — MID_TRAIN: embodied VLM (no action labels)
    "EmbSpatialBenchAdapter",
    "RefSpatialAdapter",
    "Robo2VLMAdapter",
    "RoboPointAdapter",
    "VLNR2RAdapter",
    # Phase 3 — POST_TRAIN: preference / DAgger / offline-RL
    "APOInterventionsAdapter",
    "DAggerCorrectionsAdapter",
    "FlowPROPairsAdapter",
    "HindsightRelabeledAdapter",
]
