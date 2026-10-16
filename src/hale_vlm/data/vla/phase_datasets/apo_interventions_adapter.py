"""APOInterventionsAdapter: APOInterventionsAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("apo-interventions")
class APOInterventionsAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="apo-interventions",
        hf_path=None,
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "APO (Action Preference Optimisation) — human-in-the-loop intervention "
            "preference data with adaptive reweighting. "
            "Learns from sub-optimal correction trajectories. Requires local collection."
        ),
        paper_reference="APO (2025)",
        episodes=None,
        requires_local_collection=True,
    )
