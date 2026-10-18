"""Registry adapter for the ``metaworld-mt50`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage
from hale_vlm.data.vla_catalog import (
    SIMULATION_DATASET_NOTES,
)
from hale_vlm.data.vla_registry import register_vla_dataset


@register_vla_dataset("metaworld-mt50")
class MetaWorldMT50Adapter(VLADataAdapter):
    """Registry adapter for ``metaworld-mt50``.

    Source: Hub dataset ``lerobot/metaworld_mt50``.


    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = VLADatasetSpec(
        name="metaworld-mt50",
        hf_path="lerobot/metaworld_mt50",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.SAWYER,
        description=SIMULATION_DATASET_NOTES["metaworld-mt50"],
        episodes=2500,
    )
