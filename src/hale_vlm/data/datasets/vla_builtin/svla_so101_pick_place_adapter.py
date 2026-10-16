"""SvlaSO101PickPlaceAdapter: SvlaSO101PickPlaceAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage
from hale_vlm.data.vla_catalog import (
    REAL_WORLD_DATASET_NOTES,
)
from hale_vlm.data.vla_registry import register_vla_dataset


@register_vla_dataset("svla-so101-pickplace")
class SvlaSO101PickPlaceAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="svla-so101-pickplace",
        hf_path="lerobot/svla_so101_pickplace",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.SO101,
        description=REAL_WORLD_DATASET_NOTES["svla-so101-pickplace"],
        episodes=50,
    )
