"""TrainingPhase: High-level training phase: pretrain → mid-train → post-train."""

from __future__ import annotations

from enum import StrEnum


class TrainingPhase(StrEnum):
    """High-level training phase: pretrain → mid-train → post-train."""

    PRETRAIN = "pretrain"
    MID_TRAIN = "mid_train"
    POST_TRAIN = "post_train"
