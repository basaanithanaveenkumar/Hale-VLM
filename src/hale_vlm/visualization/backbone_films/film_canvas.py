"""One frame of a film: a 16 x 9 data-coordinate stage with flat drawing primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hale_vlm.visualization.backbone_films.film_palette import DEFAULT_PALETTE, FilmPalette

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

Colour = str | tuple[float, float, float]


class FilmCanvas:
    """A fresh figure per frame with a fixed ``16 x 9`` coordinate system (origin bottom-left).

    Films describe *where* things are in these abstract units, so every film shares the same
    layout grammar (title on top, caption at the bottom) and renders at any DPI. Each primitive
    returns nothing; the frame is handed to :class:`GifWriter` via :attr:`figure`.

    Args:
        title: headline drawn at the top of the frame.
        caption: one-line explanation drawn at the bottom (what is happening *now*).
        palette: colours; defaults to the shared dark palette.
        size: figure size in inches (16:9 by default).
    """

    WIDTH = 16.0
    HEIGHT = 9.0

    def __init__(
        self,
        title: str = "",
        caption: str = "",
        palette: FilmPalette = DEFAULT_PALETTE,
        size: tuple[float, float] = (9.6, 5.4),
    ) -> None:
        # A standalone Figure (not pyplot) renders headless without touching the global backend,
        # so films never hijack a notebook's inline plots or leak open pyplot figures.
        from matplotlib.figure import Figure

        self.palette = palette
        self.figure: Figure = Figure(figsize=size, facecolor=palette.background)
        self.axes: Axes = self.figure.add_axes((0, 0, 1, 1))
        self.axes.set_xlim(0, self.WIDTH)
        self.axes.set_ylim(0, self.HEIGHT)
        self.axes.set_facecolor(palette.background)
        self.axes.axis("off")
        if title:
            self.text(0.4, 8.55, title, size=15, weight="bold", ha="left")
        if caption:
            self.text(self.WIDTH / 2, 0.35, caption, size=10.5, colour=palette.muted)

    # ------------------------------------------------------------------ primitives
    def text(
        self,
        x: float,
        y: float,
        label: str,
        size: float = 10,
        colour: Colour | None = None,
        ha: str = "center",
        va: str = "center",
        weight: str = "normal",
        alpha: float = 1.0,
        family: str = "DejaVu Sans",
    ) -> None:
        """Draw ``label`` at ``(x, y)``."""
        self.axes.text(
            x,
            y,
            label,
            fontsize=size,
            color=colour or self.palette.ink,
            ha=ha,
            va=va,
            fontweight=weight,
            alpha=alpha,
            family=family,
        )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        face: Colour | None = None,
        edge: Colour | None = None,
        label: str = "",
        label_size: float = 10,
        label_colour: Colour | None = None,
        alpha: float = 1.0,
        linewidth: float = 1.2,
        rounded: bool = True,
    ) -> None:
        """A (rounded) rectangle with lower-left corner ``(x, y)`` and an optional centred label."""
        from matplotlib.patches import FancyBboxPatch, Rectangle

        face = face if face is not None else self.palette.panel
        edge = edge if edge is not None else self.palette.faint
        if rounded and min(w, h) > 0.12:
            pad = min(0.08, 0.2 * min(w, h))
            patch = FancyBboxPatch(
                (x + pad, y + pad),
                w - 2 * pad,
                h - 2 * pad,
                boxstyle=f"round,pad={pad}",
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                alpha=alpha,
            )
        else:
            patch = Rectangle(
                (x, y), w, h, facecolor=face, edgecolor=edge, linewidth=linewidth, alpha=alpha
            )
        self.axes.add_patch(patch)
        if label:
            self.text(
                x + w / 2,
                y + h / 2,
                label,
                size=label_size,
                colour=label_colour or self.palette.ink,
                alpha=min(1.0, alpha + 0.2),
            )

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        colour: Colour | None = None,
        width: float = 1.6,
        alpha: float = 1.0,
        curvature: float = 0.0,
    ) -> None:
        """An arrow from ``start`` to ``end``; ``curvature`` bends it (arc3 connection style)."""
        from matplotlib.patches import FancyArrowPatch

        self.axes.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=width,
                color=colour or self.palette.muted,
                alpha=alpha,
                connectionstyle=f"arc3,rad={curvature}",
            )
        )

    def line(
        self,
        xs: list[float],
        ys: list[float],
        colour: Colour | None = None,
        width: float = 1.5,
        alpha: float = 1.0,
        style: str = "-",
    ) -> None:
        """A polyline through ``xs``/``ys``."""
        self.axes.plot(
            xs,
            ys,
            color=colour or self.palette.muted,
            linewidth=width,
            alpha=alpha,
            linestyle=style,
            solid_capstyle="round",
        )

    def image(self, pixels, extent: tuple[float, float, float, float], alpha: float = 1.0) -> None:
        """Show an ``(H, W, 3)`` array in ``extent = (left, right, bottom, top)``."""
        self.axes.imshow(
            pixels, extent=extent, alpha=alpha, interpolation="nearest", zorder=1, aspect="auto"
        )

    def grid(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        rows: int,
        cols: int,
        colour: Colour | None = None,
        width: float = 0.6,
        alpha: float = 0.9,
    ) -> None:
        """Grid lines dividing the rectangle ``(x, y, w, h)`` into ``rows x cols`` cells."""
        colour = colour or self.palette.ink
        for r in range(rows + 1):
            self.line([x, x + w], [y + h * r / rows] * 2, colour, width, alpha)
        for c in range(cols + 1):
            self.line([x + w * c / cols] * 2, [y, y + h], colour, width, alpha)

    # ------------------------------------------------------------------ easing helpers
    @staticmethod
    def ease(t: float) -> float:
        """Smooth-step easing of ``t`` in ``[0, 1]`` (slow in, slow out)."""
        t = min(max(t, 0.0), 1.0)
        return t * t * (3 - 2 * t)

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation from ``a`` to ``b``."""
        return a + (b - a) * t
