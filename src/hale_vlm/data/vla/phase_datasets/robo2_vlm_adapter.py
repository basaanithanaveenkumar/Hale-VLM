"""Registry adapter for the ``robo2vlm`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("robo2vlm")
class Robo2VLMAdapter(VLADataAdapter):
    """Registry adapter for ``robo2vlm``.

    Source: Hub dataset ``Robo2VLM/Robo2VLM``.

    Robo2VLM — robotic VQA generated from robot observation trajectories. No action labels;
    aligns VLM priors to robot-camera viewpoints and manipulation contexts (500K Q&A pairs).

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

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
