"""VLATrainingPhase: High-level training phase for VLA robot policies."""

from __future__ import annotations

from enum import StrEnum


class VLATrainingPhase(StrEnum):
    """High-level training phase for VLA robot policies."""

    PRETRAIN = "pretrain"
    MID_TRAIN = "mid_train"
    POST_TRAIN = "post_train"
