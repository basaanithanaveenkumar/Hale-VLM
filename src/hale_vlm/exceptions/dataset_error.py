"""A requested dataset cannot be used (disabled, unknown, or an empty selection)."""

from __future__ import annotations

from hale_vlm.exceptions.hale_vlm_error import HaleVLMError


class DatasetError(HaleVLMError, ValueError):
    """Raised when a dataset selection is invalid.

    Also a :class:`ValueError`, which these call sites raised before the hierarchy existed.
    """
