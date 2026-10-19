"""An operation the selected component does not provide."""

from __future__ import annotations

from hale_vlm.exceptions.hale_vlm_error import HaleVLMError


class UnsupportedOperationError(HaleVLMError, NotImplementedError):
    """Raised when a component cannot perform a requested operation (e.g. ``generate``)."""
