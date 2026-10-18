"""Registry adapter for the ``vln-r2r`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("vln-r2r")
class VLNR2RAdapter(VLADataAdapter):
    """Registry adapter for ``vln-r2r``.

    Source: Hub dataset ``prs-eth/room_across_the_room``.

    R2R (Room-to-Room) VLN — vision-and-language navigation trajectories in photorealistic
    Matterport3D environments. Trajectory-centric supervision for spatial grounding and path
    following.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = VLADatasetSpec(
        name="vln-r2r",
        hf_path="prs-eth/room_across_the_room",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "R2R (Room-to-Room) VLN — vision-and-language navigation trajectories "
            "in photorealistic Matterport3D environments. "
            "Trajectory-centric supervision for spatial grounding and path following."
        ),
        paper_reference="Anderson et al. (2018) R2R",
        episodes=22_000,
    )
