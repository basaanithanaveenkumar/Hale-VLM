"""Ego4DAdapter: Ego4DAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


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
