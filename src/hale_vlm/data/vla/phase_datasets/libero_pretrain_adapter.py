"""LiberoPretrainAdapter: LiberoPretrainAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


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
