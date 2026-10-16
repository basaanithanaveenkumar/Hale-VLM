"""BridgeV2Adapter: BridgeV2Adapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("bridge-v2")
class BridgeV2Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="bridge-v2",
        hf_path="lerobot/bridge_data_v2",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "BridgeData V2 — 2.26M steps on WidowX arm. "
            "Standard cross-embodiment eval; diverse household manipulation. "
            "lerobot/bridge_data_v2 (canonical LeRobot location)."
        ),
        paper_reference="Walke et al. (2023) Bridge Data V2",
        episodes=60_000,
    )
