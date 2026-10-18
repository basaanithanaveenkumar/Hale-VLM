"""An operation the selected backbone does not implement."""

from __future__ import annotations

from hale_vlm.exceptions.hale_vlm_error import HaleVLMError


class UnsupportedOperationError(HaleVLMError, NotImplementedError):
    """The chosen backbone/model lacks the requested capability (for example ``generate()``)."""
