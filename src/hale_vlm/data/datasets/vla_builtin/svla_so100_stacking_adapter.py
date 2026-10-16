"""SvlaSO100StackingAdapter: SvlaSO100StackingAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage
from hale_vlm.data.vla_catalog import (
    REAL_WORLD_DATASET_NOTES,
)
from hale_vlm.data.vla_registry import register_vla_dataset


@register_vla_dataset("svla-so100-stacking")
class SvlaSO100StackingAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="svla-so100-stacking",
        hf_path="lerobot/svla_so100_stacking",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.SO100,
        description=REAL_WORLD_DATASET_NOTES["svla-so100-stacking"],
        episodes=50,
    )
