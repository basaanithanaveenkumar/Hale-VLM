"""Film: Qwen3-VL's 3-D (time, height, width) positions and their interleaved frequency layout."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from hale_vlm.visualization.backbone_films.backbone_film import BackboneFilm
from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas


class MRoPEPositionFilm(BackboneFilm):
    """Each token gets a ``(t, h, w)`` position triple; an image is a 2-D block, text is a diagonal.

    Positions follow ``Qwen3VLModel.get_rope_index``: text tokens use ``t = h = w = p`` and
    advance ``p`` by one; an image placeholder grid of ``grid_h x grid_w`` merged tokens uses
    ``t = p``, ``h = p + row``, ``w = p + col`` and then advances ``p`` by ``max(grid_h, grid_w)``
    instead of ``grid_h * grid_w``. The closing frames show how the 64 rotary frequency pairs of a
    head are shared between the three axes: Qwen2-VL used contiguous chunks ``[T…][H…][W…]``,
    Qwen3-VL interleaves them ``T H W T H W …`` so every axis gets both fast and slow frequencies.

    Args:
        before: text tokens before the image (``<|vision_start|>`` included).
        grid: ``(grid_h, grid_w)`` merged image tokens.
        after: text tokens after the image (``<|vision_end|>`` included).
        mrope_section: frequency pairs per axis ``(t, h, w)``.
        **kwargs: forwarded to :class:`BackboneFilm`.
    """

    name = "qwen3_vl_mrope"

    def __init__(
        self,
        before: Sequence[str] = ("user", ":", "<vs>"),
        grid: tuple[int, int] = (3, 4),
        after: Sequence[str] = ("<ve>", "what", "is", "this", "?"),
        mrope_section: tuple[int, int, int] = (24, 20, 20),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.before = list(before)
        self.grid_h, self.grid_w = grid
        self.after = list(after)
        self.mrope_section = mrope_section
        self.tokens = self._positions()

    def _positions(self) -> list[tuple[str, str, int, int, int]]:
        """``(label, kind, t, h, w)`` for every token, in sequence order."""
        tokens, p = [], 0
        for label in self.before:
            tokens.append((label, "text", p, p, p))
            p += 1
        for row in range(self.grid_h):
            for col in range(self.grid_w):
                tokens.append((f"{row},{col}", "image", p, p + row, p + col))
        p += max(self.grid_h, self.grid_w)
        for label in self.after:
            tokens.append((label, "text", p, p, p))
            p += 1
        return tokens

    # ------------------------------------------------------------------ position table
    def _table(self, shown: int) -> FilmCanvas:
        palette = self.palette
        current = self.tokens[min(shown, len(self.tokens)) - 1]
        caption = (
            "image token: t fixed, h/w follow its row/column"
            if current[1] == "image"
            else "text token: t = h = w, advancing by one"
        )
        if shown == len(self.tokens):
            caption = (
                f"{self.grid_h * self.grid_w} image tokens use only "
                f"max({self.grid_h},{self.grid_w}) = {max(self.grid_h, self.grid_w)} positions; "
                "1-D RoPE would spend all of them"
            )
        canvas = self.canvas("Interleaved M-RoPE: 3-D positions", caption)
        n = len(self.tokens)
        cell = min(0.68, 13.4 / n)
        x0 = 2.1
        rows = [("t", 2, palette.tap), ("h", 3, palette.position), ("w", 4, palette.token)]
        canvas.text(1.6, 6.15, "token", size=9, ha="right", colour=palette.muted)
        for name, _, colour in rows:
            canvas.text(
                1.6,
                {"t": 5.35, "h": 4.6, "w": 3.85}[name],
                name,
                size=11,
                ha="right",
                colour=colour,
                weight="bold",
            )
        canvas.text(1.6, 2.75, "1-D", size=9, ha="right", colour=palette.muted)
        for i, (label, kind, t, h, w) in enumerate(self.tokens[:shown]):
            x = x0 + i * cell
            role = palette.token if kind == "image" else palette.text
            canvas.box(
                x,
                5.9,
                cell * 0.94,
                0.5,
                face=palette.panel,
                edge=role,
                linewidth=2.0 if i == shown - 1 else 0.8,
                rounded=False,
            )
            canvas.text(x + cell * 0.47, 6.15, label, size=6.5, colour=role)
            for value, y, (_, _, colour) in zip((t, h, w), (5.1, 4.35, 3.6), rows, strict=True):
                canvas.box(
                    x,
                    y,
                    cell * 0.94,
                    0.5,
                    face=palette.panel,
                    edge=colour,
                    linewidth=0.6,
                    rounded=False,
                )
                canvas.text(x + cell * 0.47, y + 0.25, str(value), size=8, colour=colour)
            canvas.text(x + cell * 0.47, 2.75, str(i), size=8, colour=palette.muted)
        # mini image grid (top right) highlighting the current image cell
        gx, gy, gs = 12.4, 6.9, 0.36
        canvas.text(
            gx + self.grid_w * gs / 2,
            gy + self.grid_h * gs + 0.25,
            "image grid (h, w)",
            size=8.5,
            colour=palette.muted,
        )
        image_index = 0
        for i, (_, kind, _, _, _) in enumerate(self.tokens):
            if kind != "image":
                continue
            row, col = divmod(image_index, self.grid_w)
            active = i < shown
            canvas.box(
                gx + col * gs,
                gy + (self.grid_h - 1 - row) * gs,
                gs,
                gs,
                face=palette.token if active else palette.panel,
                edge=palette.accent if i == shown - 1 else palette.faint,
                rounded=False,
                alpha=1.0 if active else 0.6,
            )
            image_index += 1
        return canvas

    # ------------------------------------------------------------------ frequency layout
    def _layout_bars(self, progress: float) -> FilmCanvas:
        palette = self.palette
        canvas = self.canvas(
            "Which rotary frequencies does each axis get?",
            "interleaving gives t, h and w a full range of fast and slow frequencies",
        )
        t, h, w = self.mrope_section
        pairs = t + h + w
        chunked = ["t"] * t + ["h"] * h + ["w"] * w
        interleaved = ["t"] * pairs
        for i in range(1, h * 3, 3):
            interleaved[i] = "h"
        for i in range(2, w * 3, 3):
            interleaved[i] = "w"
        colours = {"t": palette.tap, "h": palette.position, "w": palette.token}
        width = 14.4 / pairs
        for label, layout, y in (
            ("Qwen2-VL: chunked", chunked, 5.6),
            ("Qwen3-VL: interleaved", interleaved, 3.2),
        ):
            canvas.text(0.8, y + 1.0, label, size=11, ha="left", weight="bold")
            visible = int(round(pairs * progress))
            for i, axis in enumerate(layout[:visible]):
                canvas.box(
                    0.8 + i * width,
                    y,
                    width * 0.9,
                    0.7,
                    face=colours[axis],
                    rounded=False,
                    linewidth=0,
                )
        canvas.text(
            0.8,
            2.4,
            "fast (high frequency) →→→ slow (low frequency)",
            size=9.5,
            ha="left",
            colour=palette.muted,
        )
        x = 9.6
        for axis, colour in colours.items():
            canvas.box(x, 2.25, 0.32, 0.32, face=colour, rounded=False)
            canvas.text(x + 0.45, 2.41, axis, size=10, ha="left")
            x += 1.2
        return canvas

    def frames(self) -> Iterator[tuple[FilmCanvas, int]]:
        for shown in range(1, len(self.tokens) + 1):
            kind = self.tokens[shown - 1][1]
            first_of_kind = shown == 1 or self.tokens[shown - 2][1] != kind
            yield self._table(shown), 900 if first_of_kind else 260
        yield self._table(len(self.tokens)), 3000
        for step in range(1, 13):
            yield self._layout_bars(step / 12), 2800 if step == 12 else 90
