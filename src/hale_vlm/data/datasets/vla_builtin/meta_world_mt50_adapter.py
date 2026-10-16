"""MetaWorldMT50Adapter: MetaWorldMT50Adapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage
from hale_vlm.data.vla_catalog import (
    SIMULATION_DATASET_NOTES,
)
from hale_vlm.data.vla_registry import register_vla_dataset


@register_vla_dataset("metaworld-mt50")
class MetaWorldMT50Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="metaworld-mt50",
        hf_path="lerobot/metaworld_mt50",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.SAWYER,
        description=SIMULATION_DATASET_NOTES["metaworld-mt50"],
        episodes=2500,
    )
