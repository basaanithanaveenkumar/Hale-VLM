"""Root of the Hale-VLM exception hierarchy."""

from __future__ import annotations

from hale_core.exceptions import HaleCoreError


class HaleVLMError(HaleCoreError):
    """Base class of every expected, actionable error raised by :mod:`hale_vlm`.

    It derives from :class:`hale_core.exceptions.HaleCoreError`, so a launcher that catches the
    core base class handles Hale-VLM failures too.
    """
