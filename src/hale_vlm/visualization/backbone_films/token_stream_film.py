"""Film: the prompt the LLM actually sees, written token by token, with image runs expanded."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from hale_vlm.visualization.backbone_films.backbone_film import BackboneFilm
from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas
from hale_vlm.visualization.backbone_films.token_chip import TokenChip


class TokenStreamFilm(BackboneFilm):
    """Chips appear one after another while a counter tracks the true sequence length.

    A run of identical image placeholders is one chip (``<|image_pad|> ×256``) so long prompts
    fit on screen, but the counter adds the full run length: that number is exactly how many
    positions the image costs in the LLM context.

    Args:
        title: film headline.
        chips: the prompt in order.
        notes: optional ``{chip_index: caption}``; the caption is shown when that chip appears.
        **kwargs: forwarded to :class:`BackboneFilm`.
    """

    name = "token_stream"

    def __init__(
        self, title: str, chips: Sequence[TokenChip], notes: dict[int, str] | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if not chips:
            raise ValueError("TokenStreamFilm needs at least one chip")
        self.title = title
        self.chips = list(chips)
        self.notes = notes or {}
        self.layout = self._layout()

    def _layout(self) -> list[tuple[float, float, float]]:
        """``(x, y, width)`` per chip, wrapping at the right margin or on ``newline``."""
        x, y, row_h = 0.5, 7.4, 0.62
        placed = []
        for chip in self.chips:
            width = 0.2 + 0.105 * len(chip.caption)
            if x + width > FilmCanvas.WIDTH - 0.5:
                x, y = 0.5, y - row_h
            placed.append((x, y, width))
            x += width + 0.08
            if chip.newline:
                x, y = 0.5, y - row_h
        return placed

    def _draw(self, shown: int, caption: str) -> FilmCanvas:
        palette = self.palette
        canvas = self.canvas(self.title, caption)
        length = 0
        for i, (chip, (x, y, width)) in enumerate(zip(self.chips, self.layout, strict=True)):
            if i >= shown:
                break
            colour = getattr(palette, chip.role)
            newest = i == shown - 1
            canvas.box(
                x, y, width, 0.48, face=palette.panel, edge=colour, linewidth=2.0 if newest else 1.0
            )
            canvas.text(
                x + width / 2,
                y + 0.24,
                chip.caption,
                size=8,
                colour=colour,
                family="DejaVu Sans Mono",
            )
            length += chip.count
        counts: dict[str, int] = {}
        for chip in self.chips[:shown]:
            counts[chip.role] = counts.get(chip.role, 0) + chip.count
        canvas.text(0.5, 1.35, f"sequence length: {length}", size=12, ha="left", weight="bold")
        legend = [
            ("text", "text"),
            ("special", "markers"),
            ("token", "image tokens"),
            ("position", "position tags"),
        ]
        x = 5.6
        for role, label in legend:
            if counts.get(role):
                canvas.box(x, 1.17, 0.36, 0.36, face=getattr(palette, role), rounded=False)
                entry = f"{label}: {counts[role]}"
                canvas.text(x + 0.5, 1.35, entry, size=9.5, ha="left")
                x += 0.9 + 0.11 * len(entry)
        return canvas

    def frames(self) -> Iterator[tuple[FilmCanvas, int]]:
        caption = "The processor expands every image into placeholder tokens before the LLM runs"
        for shown in range(1, len(self.chips) + 1):
            caption = self.notes.get(shown - 1, caption)
            chip = self.chips[shown - 1]
            yield (
                self._draw(shown, caption),
                650 if chip.count > 1 or (shown - 1) in self.notes else 160,
            )
        yield self._draw(len(self.chips), caption), 2800
