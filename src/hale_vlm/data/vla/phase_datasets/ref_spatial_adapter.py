"""RefSpatialAdapter: RefSpatialAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


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
