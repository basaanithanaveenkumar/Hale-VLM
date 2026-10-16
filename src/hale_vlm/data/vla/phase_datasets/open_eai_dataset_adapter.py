"""OpenEAIDatasetAdapter: OpenEAIDatasetAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("openEAI-dataset")
class OpenEAIDatasetAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="openEAI-dataset",
        hf_path="OpenEAI/OpenEAI-Dataset",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "OpenEAI-Dataset — ~3.12 TB unified HDF5 corpus aggregating "
            "OXE + UMI Community + DROID + BC-Z. "
            "Single-format pretrain pack for cross-embodiment robot learning."
        ),
        paper_reference="OpenEAI (2024)",
        episodes=3_000_000,
        trust_remote_code=True,
    )
