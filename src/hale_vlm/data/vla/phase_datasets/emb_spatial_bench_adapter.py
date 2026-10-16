"""EmbSpatialBenchAdapter: EmbSpatialBenchAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


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
