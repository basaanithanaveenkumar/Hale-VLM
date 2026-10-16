"""Collect matplotlib frames and write a compact animated GIF."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure
    from PIL import Image


class GifWriter:
    """Accumulates rendered frames, each with its own duration, and saves one optimised GIF.

    All frames are quantised onto **one shared palette** fitted on a strip of evenly spaced frames.
    Pixels that do not change between frames then stay byte-identical, so the GIF encoder stores
    only what moved; per-frame palettes make every frame a full image (about 4x larger files).

    Args:
        dpi: rasterisation resolution for each matplotlib figure.
        colors: palette size (64 is enough for flat diagram colours).

    Example:
        >>> writer = GifWriter()
        >>> writer.add(fig, duration_ms=120)   # doctest: +SKIP
        >>> writer.save("film.gif")            # doctest: +SKIP
    """

    def __init__(self, dpi: int = 80, colors: int = 64) -> None:
        self.dpi = dpi
        self.colors = colors
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def add(self, fig: Figure, duration_ms: int = 100) -> None:
        """Rasterise ``fig`` as the next frame.

        Args:
            fig: the finished frame.
            duration_ms: how long the frame is shown.
        """
        from PIL import Image  # lazy: only needed when a film is rendered

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=self.dpi, facecolor=fig.get_facecolor())
        buffer.seek(0)
        self.frames.append(Image.open(buffer).convert("RGB"))
        self.durations.append(int(duration_ms))

    def hold(self, extra_ms: int) -> None:
        """Lengthen the last frame, a pause on a key moment.

        Args:
            extra_ms: milliseconds added to the last frame's duration.
        """
        if self.durations:
            self.durations[-1] += int(extra_ms)

    def save(self, path: str | Path) -> Path:
        """Write the GIF (looping forever) and return its path.

        Args:
            path: output file; parent directories are created.

        Returns:
            The written path.

        Raises:
            ValueError: if no frame was added.
        """
        if not self.frames:
            raise ValueError("GifWriter.save called with no frames")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = self._quantised()
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=self.durations,
            loop=0,
            optimize=True,
            disposal=1,
        )
        logger.info(
            "wrote {} ({} frames, {:.1f}s, {:.0f} kB)",
            path,
            len(frames),
            self.seconds,
            path.stat().st_size / 1024,
        )
        return path

    def _quantised(self) -> list[Image.Image]:
        """All frames mapped onto one palette, without dithering (flat colours stay flat)."""
        from PIL import Image

        sample = self.frames[:: max(1, len(self.frames) // 12)]
        width, height = sample[0].size
        strip = Image.new("RGB", (width, height * len(sample)))
        for k, frame in enumerate(sample):
            strip.paste(frame, (0, k * height))
        palette = strip.quantize(colors=self.colors, method=Image.Quantize.MEDIANCUT)
        return [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in self.frames]

    @property
    def seconds(self) -> float:
        """Total running time of the film."""
        return sum(self.durations) / 1000

    def __len__(self) -> int:
        return len(self.frames)
