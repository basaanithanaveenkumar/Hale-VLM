"""Colour palette shared by every backbone film (dark theme, matching the backbone notebooks)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilmPalette:
    """Named colours for one consistent visual grammar across films.

    The same role always gets the same colour in every film, so a reader who has watched one film
    can read the next: pixels/patches are blue, merged LLM tokens are gold, text tokens are green,
    special/marker tokens are purple, positional information is cyan, and DeepStack taps are red.

    Attributes:
        background: figure background.
        panel: background of boxes and panels.
        ink: primary text.
        muted: secondary text and inactive outlines.
        faint: grid lines and disabled elements.
        patch: vision patches (pre-merge).
        token: visual tokens after merging, in the LLM embedding space.
        text: ordinary text tokens.
        special: special / marker tokens (``<|vision_start|>``, ``<row_1_col_1>``, ...).
        position: positional ids and rotary frequencies.
        tap: DeepStack taps and residual injections.
        accent: highlights (the element currently being explained).
    """

    background: str = "#0f1117"
    panel: str = "#1b1f2a"
    ink: str = "#e8eaf0"
    muted: str = "#8a91a3"
    faint: str = "#2c3242"
    patch: str = "#4c8dff"
    token: str = "#f5b829"
    text: str = "#3ecf8e"
    special: str = "#b07cff"
    position: str = "#33c3d6"
    tap: str = "#ff5c6c"
    accent: str = "#ff9f43"


DEFAULT_PALETTE = FilmPalette()
