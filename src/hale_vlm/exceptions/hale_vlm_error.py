"""Root of the Hale-VLM exception hierarchy."""

from __future__ import annotations

from hale_core.exceptions import HaleCoreError


class HaleVLMError(HaleCoreError):
    """Base class of every error Hale-VLM raises on purpose.

    Rooted at :class:`~hale_core.exceptions.HaleCoreError` so one ``except HaleCoreError`` covers
    the whole Hale* family while callers can still catch only VLM failures.
    """
