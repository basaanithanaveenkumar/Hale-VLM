"""RoboCasaAdapter: RoboCasaAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


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
