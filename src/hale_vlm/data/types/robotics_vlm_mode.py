"""RoboticsVLMMode: How robotics registry samples are exposed to VLM training."""

from __future__ import annotations

from enum import StrEnum


class RoboticsVLMMode(StrEnum):
    """How robotics registry samples are exposed to VLM training."""

    OFF = "off"
    PRETRAINING = "pretraining"
    FINETUNING = "finetuning"
    INSTRUCTION_TUNING = "instruction_tuning"
