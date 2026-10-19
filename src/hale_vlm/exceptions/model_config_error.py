"""Architecture settings that cannot build a working module."""

from __future__ import annotations

from hale_vlm.exceptions.hale_vlm_error import HaleVLMError


class ModelConfigError(HaleVLMError, ValueError):
    """Raised for impossible model settings (e.g. width not divisible by heads, unknown type).

    Also a :class:`ValueError`, which these call sites raised before the hierarchy existed.
    """
