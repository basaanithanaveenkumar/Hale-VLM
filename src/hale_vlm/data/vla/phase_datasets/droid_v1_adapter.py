"""Registry adapter for the ``droid-v1`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("droid-v1")
class DroidV1Adapter(VLADataAdapter):
    """Registry adapter for ``droid-v1``.

    Source: Hub dataset ``lerobot/droid_1.0.1``.

    DROID 1.0.1 — 76K trajectories, 564 scenes, 86 tasks, 50 operators. In-the-wild Franka Panda
    across many labs. High scene diversity; key pretrain complement to OXE (Khazatsky et al.,
    2024).

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = VLADatasetSpec(
        name="droid-v1",
        hf_path="lerobot/droid_1.0.1",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "DROID 1.0.1 — 76K trajectories, 564 scenes, 86 tasks, 50 operators. "
            "In-the-wild Franka Panda across many labs. High scene diversity; "
            "key pretrain complement to OXE (Khazatsky et al., 2024)."
        ),
        paper_reference="Khazatsky et al. (2024) DROID",
        episodes=76_000,
    )
