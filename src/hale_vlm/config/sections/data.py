"""VLM-specific data configuration."""

from __future__ import annotations

from typing import Literal

from hale_core.config.sections.data import DataConfig
from pydantic import Field, model_validator

from hale_vlm.data.types import RoboticsVLMMode, TrainingPhase
from hale_vlm.exceptions import VLMConfigError


class VLMDataConfig(DataConfig):
    source: Literal[
        "huggingface",
        "overfit",
        "registry",
        "phase_registry",
        "vla_registry",
        "mixed_registry",
    ] = "huggingface"
    registry_stage: Literal["all", "vision", "video", "context"] = "all"
    registry_datasets: list[str] = Field(default_factory=list)
    registry_phase: TrainingPhase | None = None
    """Training phase whose datasets ``source: phase_registry`` streams (pretrain/mid/post)."""
    # SmolVLA robotics registry (can also feed VLM training via robotics_vlm_mode)
    vla_registry_stage: Literal["all", "community", "simulation", "real_world"] = "all"
    vla_registry_datasets: list[str] = Field(default_factory=list)
    robotics_vlm_mode: RoboticsVLMMode = RoboticsVLMMode.OFF
    max_samples_per_dataset: int | None = 256
    streaming: bool = True
    prefetch_workers: int = 2
    max_video_frames: int = 8

    @model_validator(mode="after")
    def _require_phase_for_phase_registry(self) -> VLMDataConfig:
        if self.source == "phase_registry" and self.registry_phase is None:
            raise VLMConfigError(
                "data.source='phase_registry' requires data.registry_phase "
                "(one of: pretrain, mid_train, post_train)"
            )
        return self

    def model_post_init(self, __context) -> None:
        if not self.registry_datasets:
            from hale_vlm.data.catalog import SMOLVLM_ALL_DATASETS

            self.registry_datasets = list(SMOLVLM_ALL_DATASETS)
        if not self.vla_registry_datasets:
            from hale_vlm.data.vla_catalog import SMOLVLA_ALL_DATASETS

            self.vla_registry_datasets = list(SMOLVLA_ALL_DATASETS)

    def resolved_registry_datasets(self) -> list[str]:
        """Dataset names to stream for the VLM registry sources.

        Returns:
            The phase's registered datasets for ``source: phase_registry``, the stage preset
            when ``registry_stage`` is not ``all``, otherwise ``registry_datasets``.
        """
        if self.source == "phase_registry":
            from hale_vlm.data.phase_registry import list_phase_datasets

            return list_phase_datasets(phase=self.registry_phase)
        from hale_vlm.data.catalog import (
            SMOLVLM_ALL_DATASETS,
            SMOLVLM_CONTEXT_DATASETS,
            SMOLVLM_VIDEO_DATASETS,
            SMOLVLM_VISION_DATASETS,
        )

        presets = {
            "vision": SMOLVLM_VISION_DATASETS,
            "video": SMOLVLM_VIDEO_DATASETS,
            "context": SMOLVLM_CONTEXT_DATASETS,
            "all": SMOLVLM_ALL_DATASETS,
        }
        if self.registry_stage != "all":
            return list(presets[self.registry_stage])
        return self.registry_datasets

    def resolved_vla_registry_datasets(self) -> list[str]:
        from hale_vlm.data.vla_catalog import (
            SMOLVLA_ALL_DATASETS,
            SMOLVLA_COMMUNITY_DATASETS,
            SMOLVLA_REAL_WORLD_DATASETS,
            SMOLVLA_SIMULATION_DATASETS,
        )

        presets = {
            "community": SMOLVLA_COMMUNITY_DATASETS,
            "simulation": SMOLVLA_SIMULATION_DATASETS,
            "real_world": SMOLVLA_REAL_WORLD_DATASETS,
            "all": SMOLVLA_ALL_DATASETS,
        }
        if self.vla_registry_stage != "all":
            return list(presets[self.vla_registry_stage])
        return self.vla_registry_datasets

    def robotics_in_vlm_enabled(self) -> bool:
        return self.robotics_vlm_mode != RoboticsVLMMode.OFF

    def uses_registry_stream(self) -> bool:
        return (
            self.source in {"registry", "phase_registry", "mixed_registry"}
            or self.robotics_in_vlm_enabled()
        )
