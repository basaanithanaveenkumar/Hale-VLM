"""A registered dataset exists but cannot be streamed."""

from __future__ import annotations

from hale_vlm.exceptions.vlm_dataset_error import VLMDatasetError


class DatasetUnavailableError(VLMDatasetError):
    """The adapter is disabled in its registry, or needs data collected locally (no Hub path)."""
