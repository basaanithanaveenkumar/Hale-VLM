"""Modality: Modality."""

from __future__ import annotations

from enum import StrEnum


class Modality(StrEnum):
    """Primary input modality of a dataset (text, image-text, video, ...)."""

    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    MULTI_IMAGE = "multi_image"
