"""EgoVLAAdapter: EgoVLAAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


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
