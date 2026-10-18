"""Registry adapter for the ``robopoint`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("robopoint")
class RoboPointAdapter(VLADataAdapter):
    """Registry adapter for ``robopoint``.

    Source: Hub dataset ``wentao-yuan/robopoint-data``.

    RoboPoint — spatial affordance prediction: given image + instruction, predict the target 2D
    manipulation point. Trains spatial reasoning without requiring low-level action labels.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = VLADatasetSpec(
        name="robopoint",
        hf_path="wentao-yuan/robopoint-data",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "RoboPoint — spatial affordance prediction: given image + instruction, "
            "predict the target 2D manipulation point. "
            "Trains spatial reasoning without requiring low-level action labels."
        ),
        paper_reference="Yuan et al. (2024) RoboPoint",
        episodes=600_000,
    )
