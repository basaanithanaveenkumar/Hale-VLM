"""A procedural RGB 'photo' so films and notebooks need no image files or downloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


class SyntheticScene:
    """Draws a simple outdoor scene (sky, ground, sun, house) at any resolution.

    The content is deliberately recognisable at a glance, so tiles, crops and resizes are easy to
    follow in an animation, and deterministic, so notebook outputs are reproducible.
    """

    @staticmethod
    def render(height: int, width: int) -> np.ndarray:
        """Return a ``(height, width, 3)`` float array in ``[0, 1]``.

        Args:
            height: rows of the output.
            width: columns of the output.
        """
        import numpy as np  # lazy: keeps the package import light

        ys, xs = np.mgrid[0:height, 0:width] / np.array([height, width])[:, None, None]
        sky = np.stack([0.25 + 0.3 * ys, 0.45 + 0.3 * ys, 0.85 - 0.2 * ys], -1)
        ground = np.stack([0.2 + 0.1 * xs, 0.5 - 0.1 * ys, 0.25 + 0 * xs], -1)
        img = np.where((ys > 0.62)[..., None], ground, sky)
        sun = (xs - 0.78) ** 2 * (width / height) ** 2 + (ys - 0.22) ** 2 < 0.012
        img[sun] = (1.0, 0.85, 0.3)
        house = (xs > 0.2) & (xs < 0.42) & (ys > 0.45) & (ys < 0.7)
        img[house] = (0.75, 0.35, 0.3)
        roof = (ys < 0.45) & (ys > 0.45 - 0.5 * (0.11 - abs(xs - 0.31))) & (abs(xs - 0.31) < 0.13)
        img[roof] = (0.45, 0.2, 0.2)
        return img.clip(0, 1)
