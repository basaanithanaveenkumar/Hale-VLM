"""Invalid arguments to a backbone animation (film)."""

from __future__ import annotations

from hale_vlm.exceptions.hale_vlm_error import HaleVLMError


class FilmError(HaleVLMError, ValueError):
    """Raised when a film is built with inconsistent data (empty stages, bad mapping, …)."""
