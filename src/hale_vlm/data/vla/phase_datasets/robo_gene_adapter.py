"""RoboGeneAdapter: RoboGeneAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


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
