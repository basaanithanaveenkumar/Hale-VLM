"""HindsightRelabeledAdapter: HindsightRelabeledAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("hindsight-relabeled")
class HindsightRelabeledAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="hindsight-relabeled",
        hf_path=None,
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "LfH (Learning from Hindsight) hindsight-relabeled rollouts — "
            "failed trajectories are scored against tasks actually achieved and "
            "reused as positives. Useful when early policies rarely succeed."
        ),
        paper_reference="LfH (2024)",
        episodes=None,
        requires_local_collection=True,
    )
