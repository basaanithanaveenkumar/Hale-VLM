"""Registry adapter for the ``dagger-corrections`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("dagger-corrections")
class DAggerCorrectionsAdapter(VLADataAdapter):
    """Registry adapter for ``dagger-corrections``.

    Source: locally collected corpus, no Hub path.

    DAgger-style interactive correction trajectories — human operator takes over at failure
    modes, producing corrective demos that cover the state distribution induced by the current
    policy.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = VLADatasetSpec(
        name="dagger-corrections",
        hf_path=None,
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "DAgger-style interactive correction trajectories — human operator "
            "takes over at failure modes, producing corrective demos that cover "
            "the state distribution induced by the current policy."
        ),
        paper_reference="Ross et al. (2011) DAgger",
        episodes=None,
        requires_local_collection=True,
    )
