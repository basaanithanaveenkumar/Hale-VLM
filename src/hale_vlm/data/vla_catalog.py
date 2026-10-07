"""SmolVLA paper dataset catalog, stage presets, and training-phase presets."""

from __future__ import annotations

from hale_vlm.data.types import RobotEmbodiment, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.community_paths import SMOLVLA_COMMUNITY_HF_PATHS, hf_path_to_registry_name

# Simulation benchmarks (§4.1)
SMOLVLA_SIMULATION_DATASETS: tuple[str, ...] = (
    "libero",
    "metaworld-mt50",
)

# Real-world author datasets (§4.1, Figure 4)
SMOLVLA_REAL_WORLD_DATASETS: tuple[str, ...] = (
    "svla-so100-pickplace",
    "svla-so100-stacking",
    "svla-so100-sorting",
    "svla-so101-pickplace",
)

# Community SO-100 pretraining (Appendix A.1)
SMOLVLA_COMMUNITY_DATASETS: tuple[str, ...] = tuple(
    hf_path_to_registry_name(path) for path in SMOLVLA_COMMUNITY_HF_PATHS
)

SMOLVLA_ALL_DATASETS: tuple[str, ...] = (
    *SMOLVLA_COMMUNITY_DATASETS,
    *SMOLVLA_SIMULATION_DATASETS,
    *SMOLVLA_REAL_WORLD_DATASETS,
)

VLA_STAGE_PRESETS: dict[VLAStage, tuple[str, ...]] = {
    VLAStage.COMMUNITY: SMOLVLA_COMMUNITY_DATASETS,
    VLAStage.SIMULATION: SMOLVLA_SIMULATION_DATASETS,
    VLAStage.REAL_WORLD: SMOLVLA_REAL_WORLD_DATASETS,
}

SIMULATION_DATASET_NOTES: dict[str, str] = {
    "libero": "LIBERO — 40 tasks, 1,693 episodes (Liu et al., 2023a)",
    "metaworld-mt50": "Meta-World MT50 — 50 tasks, 2,500 episodes (Yu et al., 2020)",
}

REAL_WORLD_DATASET_NOTES: dict[str, str] = {
    "svla-so100-pickplace": "Pick cube, place in box — 50 demos on SO-100",
    "svla-so100-stacking": "Stack red cube on blue cube — 50 demos on SO-100",
    "svla-so100-sorting": "Sort colored cubes into boxes — 50 demos on SO-100",
    "svla-so101-pickplace": "Pick pink Lego, place in box — 50 demos on SO-101",
}

COMMUNITY_STATS = {
    "paper_reported_datasets": 481,
    "appendix_hf_paths": len(SMOLVLA_COMMUNITY_HF_PATHS),
    "paper_reported_episodes": 22_900,
    "paper_reported_frames": 10_600_000,
}

EMBODIMENT_BY_STAGE: dict[VLAStage, RobotEmbodiment] = {
    VLAStage.COMMUNITY: RobotEmbodiment.SO100,
    VLAStage.SIMULATION: RobotEmbodiment.MIXED,
    VLAStage.REAL_WORLD: RobotEmbodiment.SO100,
}

# ---------------------------------------------------------------------------
# VLA training-phase dataset groups (separate from SmolVLA stage taxonomy)
# These live in a separate VLA_PHASE_DATASETS registry.
# ---------------------------------------------------------------------------

# Phase 1 — PRETRAIN: core + aggregated + human-video sources
PRETRAIN_VLA_DATASETS: tuple[str, ...] = (
    # Core / flagship cross-embodiment corpora
    "open-x-embodiment",      # OXE: 22 robot types, ~2M demos
    "bridge-v2",              # BridgeData V2: 2.26M steps, WidowX
    "fractal-rt1",            # FractalData/RT-1: 3.79M steps (lerobot/fractal)
    "bc-z",                   # BC-Z: 25K eps, 100 tasks
    "droid-v1",               # DROID 1.0.1: 76K trajectories, in-the-wild Franka
    "libero-pretrain",        # LIBERO HuggingFaceVLA: 130+ tasks, 5K+ eps
    # Aggregated / preprocessed packs
    "openEAI-dataset",        # OpenEAI: OXE + UMI + DROID + BC-Z unified HDF5
    "lerobot-community-v3",   # LeRobot Community v3: 791 datasets, 46 robots
    "robogene",               # RoboGene: diversity-driven agentic generation
    # Human-video and tactile pretraining
    "being-h0",               # Being-H0: large-scale human video pretraining
    "agibot-world",           # AgiBot World: bimanual real-world manip (X-VLA)
    "h-tac-ttp",              # H-Tac TTP: tactile pretraining
)

VLA_PHASE_NOTES: dict[VLATrainingPhase, str] = {
    VLATrainingPhase.PRETRAIN: (
        "Large-scale cross-embodiment robot teleoperation + aggregated packs "
        "+ human-video pretraining. Standard OXE + DROID + BridgeV2 baseline "
        "plus RoboGene and LeRobot Community v3 for breadth."
    ),
    VLATrainingPhase.MID_TRAIN: (
        "Domain-specific manipulation data for the target embodiment family. "
        "Adapts pretrained features to workspace and object distribution."
    ),
    VLATrainingPhase.POST_TRAIN: (
        "Task-specific fine-tuning on a small set of target tasks. "
        "Maximises success rate on the deployment benchmark."
    ),
}
