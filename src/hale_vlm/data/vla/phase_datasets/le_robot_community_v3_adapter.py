"""LeRobotCommunityV3Adapter: LeRobotCommunityV3Adapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("lerobot-community-v3")
class LeRobotCommunityV3Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="lerobot-community-v3",
        hf_path="lerobot/community_dataset_v3",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "LeRobot Community Dataset v3 — 791 datasets across 46 robot types, "
            "consolidated from 851 community sources. "
            "LeRobot Datasets v3.0 with chunked episodes for OXE-scale (>400 GB)."
        ),
        paper_reference="Cadène et al. (2024) LeRobot",
        episodes=5_000_000,
        trust_remote_code=True,
    )
