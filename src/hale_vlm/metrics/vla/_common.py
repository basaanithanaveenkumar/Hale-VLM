"""Shared helpers and constants of :mod:`vla` (split from the original module)."""

from __future__ import annotations

import string


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())
