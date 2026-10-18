"""Invalid argument passed to a Hale-VLM component."""

from __future__ import annotations

from hale_vlm.exceptions.hale_vlm_error import HaleVLMError


class VLMInputError(HaleVLMError, ValueError):
    """A constructor or method argument violates the component's contract.

    Typical causes are a wrong shape, a non-divisible dimension or an empty collection.

    Also a :class:`ValueError`, the built-in these sites raised before the hierarchy existed.
    """
