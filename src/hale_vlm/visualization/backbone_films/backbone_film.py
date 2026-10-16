"""Abstract base class of every backbone film (Template Method)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from hale_vlm.visualization.backbone_films.film_palette import DEFAULT_PALETTE, FilmPalette
from hale_vlm.visualization.backbone_films.gif_writer import GifWriter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas


class BackboneFilm(ABC):
    """A short animation that explains one mechanism of a VLM backbone.

    Subclasses implement :meth:`frames`, a generator of ``(canvas, duration_ms)`` pairs; this base
    class owns everything else (rasterising, palette quantisation, saving, notebook display), so a
    new film is only the drawing logic. That is the Template Method pattern: :meth:`render` is the
    fixed algorithm, :meth:`frames` the variable step.

    Args:
        palette: colours shared by all films.
        dpi: rasterisation resolution of each frame.
    """

    #: file stem used by :class:`BackboneFilmRenderer` (``<name>.gif``); set by subclasses.
    name: str = "film"

    def __init__(self, palette: FilmPalette = DEFAULT_PALETTE, dpi: int = 80) -> None:
        self.palette = palette
        self.dpi = dpi

    @abstractmethod
    def frames(self) -> Iterator[tuple[FilmCanvas, int]]:
        """Yield every frame of the film with its display duration in milliseconds."""

    def canvas(self, title: str = "", caption: str = "") -> FilmCanvas:
        """A new frame in this film's palette (convenience for subclasses)."""
        from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas

        return FilmCanvas(title=title, caption=caption, palette=self.palette)

    def render(self, path: str | Path) -> Path:
        """Draw all frames and write them to ``path`` as a looping GIF.

        Args:
            path: output ``.gif`` file.

        Returns:
            The written path.
        """
        writer = GifWriter(dpi=self.dpi)
        for canvas, duration_ms in self.frames():
            writer.add(canvas.figure, duration_ms)
        logger.debug("{}: {} frames", type(self).__name__, len(writer))
        return writer.save(path)

    def show(self, path: str | Path):
        """Render to ``path`` and return an ``IPython.display.Image`` that embeds the GIF.

        Embedding (rather than linking) keeps the animation visible on GitHub and in exported
        notebooks.

        Args:
            path: output ``.gif`` file.

        Returns:
            An ``IPython.display.Image`` for the notebook cell output.
        """
        from IPython.display import Image  # lazy: only needed inside Jupyter

        return Image(filename=str(self.render(path)), embed=True)
