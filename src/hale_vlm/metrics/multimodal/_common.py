"""Shared helpers and constants of :mod:`multimodal` (split from the original module)."""

from __future__ import annotations

from torch import Tensor


def _l2_normalise(x: Tensor) -> Tensor:
    """Return L2-normalised version of a 1D or 2D tensor."""
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-8))
