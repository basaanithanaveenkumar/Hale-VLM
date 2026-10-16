"""AgibotWorldAdapter: AgibotWorldAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("agibot-world")
class AgibotWorldAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="agibot-world",
        hf_path="lerobot/xvla-agibot-world",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "AgiBot World — bimanual real-world manipulation used in X-VLA pretraining. "
            "High-quality dexterous tasks with rich scene diversity."
        ),
        paper_reference="AgiBot (2024) AgiBot World",
        episodes=200_000,
    )
