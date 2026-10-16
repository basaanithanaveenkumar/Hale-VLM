"""LiberoAdapter: LiberoAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage
from hale_vlm.data.vla_catalog import (
    SIMULATION_DATASET_NOTES,
)
from hale_vlm.data.vla_registry import register_vla_dataset


@register_vla_dataset("libero")
class LiberoAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="libero",
        hf_path="physical-intelligence/libero",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        description=SIMULATION_DATASET_NOTES["libero"],
        episodes=1693,
    )
