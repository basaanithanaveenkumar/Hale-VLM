"""Film: the end-to-end backbone pipeline, stage by stage, with the tensor shape between stages."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator, Sequence

from hale_vlm.exceptions import VLMInputError
from hale_vlm.visualization.backbone_films.backbone_film import BackboneFilm
from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas
from hale_vlm.visualization.backbone_films.pipeline_stage import PipelineStage


class PipelineFilm(BackboneFilm):
    """Stages light up left to right while a data packet travels between them.

    Each arrow is labelled with the tensor shape leaving the previous stage, so the film doubles
    as a shape cheat-sheet for the notebook that follows. Stages wrap onto a second row (snake
    order) when there are more than ``per_row``.

    Args:
        title: film headline.
        stages: the pipeline, in data-flow order.
        per_row: stages per row before wrapping.
        steps_per_hop: animation frames for the packet to travel one arrow.
        **kwargs: forwarded to :class:`BackboneFilm` (palette, dpi).
    """

    name = "pipeline"

    def __init__(
        self,
        title: str,
        stages: Sequence[PipelineStage],
        per_row: int = 4,
        steps_per_hop: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not stages:
            raise VLMInputError("PipelineFilm needs at least one stage")
        self.title = title
        self.stages = list(stages)
        self.per_row = per_row
        self.steps_per_hop = steps_per_hop

    # ------------------------------------------------------------------ layout
    def _slot(self, index: int) -> tuple[float, float, float, float]:
        """Box ``(x, y, w, h)`` for stage ``index`` (snake order across rows)."""
        rows = (len(self.stages) + self.per_row - 1) // self.per_row
        row, col = divmod(index, self.per_row)
        if row % 2 == 1:  # snake: odd rows run right-to-left so arrows stay short
            col = self.per_row - 1 - col
        gap = 0.9
        width = (FilmCanvas.WIDTH - 0.8 - gap * (self.per_row - 1)) / self.per_row
        height = 1.9 if rows > 1 else 2.4
        top = 6.6 if rows > 1 else 5.6
        y = top - row * (height + 1.3) - height
        return 0.4 + col * (width + gap), y, width, height

    def _anchor(self, index: int, side: str) -> tuple[float, float]:
        """Arrow anchor on the ``side`` (``in``/``out``) of stage ``index``."""
        x, y, w, h = self._slot(index)
        row = index // self.per_row
        forward = row % 2 == 0
        next_row = (index + 1) // self.per_row
        if side == "out" and next_row != row:  # last box of a row: leave from the bottom
            return x + w / 2, y
        if (
            side == "in" and index % self.per_row == 0 and index > 0
        ):  # first box of a row: enter on top
            return x + w / 2, y + h
        if side == "out":
            return (x + w, y + h / 2) if forward else (x, y + h / 2)
        return (x, y + h / 2) if forward else (x + w, y + h / 2)

    # ------------------------------------------------------------------ drawing
    def _draw(self, lit: int, hop: int | None, progress: float) -> FilmCanvas:
        """One frame: stages ``<= lit`` are lit; the packet is ``progress`` along arrow ``hop``."""
        palette = self.palette
        stage = self.stages[min(lit, len(self.stages) - 1)]
        canvas = self.canvas(self.title, f"{stage.name}: {stage.detail}")
        for i, item in enumerate(self.stages):
            x, y, w, h = self._slot(i)
            active = i <= lit
            colour = getattr(palette, item.role)
            canvas.box(
                x,
                y,
                w,
                h,
                face=palette.panel,
                edge=colour if active else palette.faint,
                linewidth=2.2 if i == lit else 1.2,
                alpha=1.0 if active else 0.45,
            )
            canvas.text(
                x + w / 2,
                y + h - 0.42,
                item.name,
                size=11,
                weight="bold",
                colour=colour if active else palette.muted,
            )
            detail = textwrap.fill(item.detail, width=max(14, int(w * 6.2)))
            canvas.text(
                x + w / 2,
                y + h / 2 - 0.05,
                detail,
                size=8.2,
                colour=palette.ink if active else palette.faint,
            )
            canvas.text(
                x + w / 2,
                y + 0.3,
                f"→ {item.shape}",
                size=7.8,
                colour=palette.ink if active else palette.faint,
                family="DejaVu Sans Mono",
            )
            if i < len(self.stages) - 1:
                canvas.arrow(
                    self._anchor(i, "out"),
                    self._anchor(i + 1, "in"),
                    colour=colour if i < lit else palette.faint,
                    width=1.6,
                )
        if hop is not None:
            start, end = self._anchor(hop, "out"), self._anchor(hop + 1, "in")
            t = FilmCanvas.ease(progress)
            px, py = FilmCanvas.lerp(start[0], end[0], t), FilmCanvas.lerp(start[1], end[1], t)
            canvas.box(
                px - 0.16,
                py - 0.16,
                0.32,
                0.32,
                face=palette.accent,
                edge=palette.accent,
                rounded=False,
            )
        last = self.stages[-1]
        canvas.text(
            FilmCanvas.WIDTH - 0.4,
            1.0,
            f"output: {last.shape}",
            size=9,
            ha="right",
            colour=palette.muted if lit < len(self.stages) - 1 else palette.token,
            family="DejaVu Sans Mono",
        )
        return canvas

    def frames(self) -> Iterator[tuple[FilmCanvas, int]]:
        yield self._draw(0, None, 0.0), 900
        for hop in range(len(self.stages) - 1):
            for step in range(1, self.steps_per_hop + 1):
                yield self._draw(hop, hop, step / self.steps_per_hop), 70
            yield self._draw(hop + 1, None, 0.0), 1100
        yield self._draw(len(self.stages) - 1, None, 0.0), 2200
