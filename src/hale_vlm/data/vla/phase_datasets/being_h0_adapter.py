"""Registry adapter for the ``being-h0`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("being-h0")
class BeingH0Adapter(VLADataAdapter):
    """Registry adapter for ``being-h0``.

    Source: Hub dataset ``BeingBeyond/Being-H0``.

    Being-H0 post-training dataset — pretrained from large-scale human videos via explicit hand
    motion modelling. Bridges embodiment gap using egocentric 2D/3D cues.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = VLADatasetSpec(
        name="being-h0",
        hf_path="BeingBeyond/Being-H0",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Being-H0 post-training dataset — pretrained from large-scale human "
            "videos via explicit hand motion modelling. "
            "Bridges embodiment gap using egocentric 2D/3D cues."
        ),
        paper_reference="BeingBeyond (2024) Being-H0",
        episodes=100_000,
        trust_remote_code=True,
    )
