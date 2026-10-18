"""Registry adapter for the ``bc-z`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("bc-z")
class BCZAdapter(VLADataAdapter):
    """Registry adapter for ``bc-z``.

    Source: Hub dataset ``lerobot/bc_z``.

    BC-Z — 25K robot episodes across 100 tasks on Google's robot. Collected via human
    demonstration and teleoperation.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = VLADatasetSpec(
        name="bc-z",
        hf_path="lerobot/bc_z",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "BC-Z — 25K robot episodes across 100 tasks on Google's robot. "
            "Collected via human demonstration and teleoperation."
        ),
        paper_reference="Jang et al. (2022) BC-Z",
        episodes=25_000,
    )
