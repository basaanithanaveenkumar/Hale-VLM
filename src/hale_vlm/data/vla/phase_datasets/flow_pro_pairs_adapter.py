"""FlowPROPairsAdapter: FlowPROPairsAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("flowpro-pairs")
class FlowPROPairsAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="flowpro-pairs",
        hf_path=None,
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "FlowPRO rollback preference pairs — a single operator intervention "
            "yields a (winner, loser) trajectory pair via operator-chosen rollback "
            "horizon. No separate positive/negative recordings needed. "
            "Requires local robot deployment to collect."
        ),
        paper_reference="FlowPRO (2025)",
        episodes=None,
        requires_local_collection=True,
    )
