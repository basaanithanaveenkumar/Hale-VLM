"""FractalRT1Adapter: FractalRT1Adapter."""

from __future__ import annotations

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset


@register_vla_phase_dataset("fractal-rt1")
class FractalRT1Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="fractal-rt1",
        hf_path="lerobot/fractal",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "FractalData / RT-1 — 3.79M steps, Google RT-1 training corpus. "
            "Part of Open X-Embodiment; lerobot/fractal canonical location."
        ),
        paper_reference="Brohan et al. (2022) RT-1",
        episodes=130_000,
    )
