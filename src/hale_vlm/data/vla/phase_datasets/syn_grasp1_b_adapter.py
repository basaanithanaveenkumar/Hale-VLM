"""SynGrasp1BAdapter: SynGrasp1BAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


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
