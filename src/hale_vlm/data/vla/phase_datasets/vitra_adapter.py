"""Registry adapter for the ``vitra`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("vitra")
class VITRAAdapter(VLADataAdapter):
    """Registry adapter for ``vitra``.

    Source: Hub dataset ``VITRA-Dataset/VITRA``.

    VITRA (ICRA 2026) — converts in-the-wild human hand videos into (image, instruction, action)
    tuples for VLA pretraining. Explicitly bridges the embodiment gap via egocentric hand motion
    parsing.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = VLADatasetSpec(
        name="vitra",
        hf_path="VITRA-Dataset/VITRA",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.HUMAN,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "VITRA (ICRA 2026) — converts in-the-wild human hand videos into "
            "(image, instruction, action) tuples for VLA pretraining. "
            "Explicitly bridges the embodiment gap via egocentric hand motion parsing."
        ),
        paper_reference="VITRA (2026) ICRA",
        episodes=500_000,
    )
