"""OpenXEmbodimentAdapter: OpenXEmbodimentAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("open-x-embodiment")
class OpenXEmbodimentAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="open-x-embodiment",
        hf_path="jxu124/OpenX-Embodiment",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Open X-Embodiment — 22 robot types, ~2M demonstrations across "
            "kitchen, tabletop, and outdoor environments. "
            "Standard cross-embodiment pretraining corpus for OpenVLA, Octo, π₀."
        ),
        paper_reference="Open X-Embodiment Collaboration (2023)",
        episodes=2_000_000,
        trust_remote_code=True,
    )
