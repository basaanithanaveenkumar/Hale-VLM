"""HTacTTPAdapter: HTacTTPAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("h-tac-ttp")
class HTacTTPAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="h-tac-ttp",
        hf_path="BeingBeyond/TTP",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "H-Tac TTP (Tactile Transformer Pretraining) — tactile sensor "
            "pretraining for dexterous manipulation. "
            "Captures contact dynamics unavailable in vision-only corpora."
        ),
        paper_reference="BeingBeyond (2024) H-Tac",
        episodes=50_000,
        trust_remote_code=True,
    )
