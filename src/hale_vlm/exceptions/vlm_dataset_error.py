"""A VLM/VLA dataset cannot be resolved or read."""

from __future__ import annotations

from hale_core.exceptions import DataError

from hale_vlm.exceptions.hale_vlm_error import HaleVLMError


class VLMDatasetError(HaleVLMError, DataError):
    """Dataset-level failure in the VLM or VLA registries; also a :class:`ValueError`."""
