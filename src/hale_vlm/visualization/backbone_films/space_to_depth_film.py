"""Film: neighbouring patches folded into one wider token (Qwen3-VL merge, pixel shuffle)."""

from __future__ import annotations

import colorsys
from collections.abc import Iterator, Sequence

from hale_vlm.exceptions import VLMInputError
from hale_vlm.visualization.backbone_films.backbone_film import BackboneFilm
from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas


class SpaceToDepthFilm(BackboneFilm):
    """Patches fly from the encoder grid into the token they are concatenated into.

    The film is driven by ``mapping``, the *measured* gather pattern of the real operation:
    ``mapping[k]`` lists the flat patch indices (raster order) whose features are concatenated,
    in channel order, to form output token ``k``. The notebooks compute it by pushing an index
    tensor through the actual library function, so the animation cannot drift from the code.

    Left: the ``grid_h x grid_w`` patch grid, each patch coloured by its destination token (hue)
    and its slot inside that token (lightness). Right: the ``out_h x out_w`` token grid; each
    token is drawn as ``r*r`` stripes, one per concatenated patch, in channel order.

    Args:
        title: film headline.
        grid: ``(grid_h, grid_w)`` patches entering the operation.
        out_grid: ``(out_h, out_w)`` tokens leaving it.
        mapping: ``out_h*out_w`` lists of ``r*r`` patch indices, in channel order.
        in_label: label under the patch grid (e.g. ``"ViT patches (B, 64, 1152)"``).
        out_label: label under the token grid.
        animate_tokens: tokens animated one at a time; the remaining ones move together.
        **kwargs: forwarded to :class:`BackboneFilm`.
    """

    name = "space_to_depth"

    def __init__(
        self,
        title: str,
        grid: tuple[int, int],
        out_grid: tuple[int, int],
        mapping: Sequence[Sequence[int]],
        in_label: str,
        out_label: str,
        animate_tokens: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.grid_h, self.grid_w = grid
        self.out_h, self.out_w = out_grid
        self.mapping = [list(m) for m in mapping]
        if len(self.mapping) != self.out_h * self.out_w:
            raise VLMInputError(
                f"mapping has {len(self.mapping)} tokens, out_grid needs {self.out_h * self.out_w}"
            )
        self.group = len(self.mapping[0])
        covered = sorted(p for m in self.mapping for p in m)
        if covered != list(range(self.grid_h * self.grid_w)):
            raise VLMInputError("mapping must use every patch exactly once")
        self.in_label = in_label
        self.out_label = out_label
        self.animate_tokens = min(animate_tokens, len(self.mapping))
        # patch -> (token, slot): the inverse of mapping, used for colouring
        self.owner = {p: (k, j) for k, m in enumerate(self.mapping) for j, p in enumerate(m)}

    # ------------------------------------------------------------------ geometry
    def _patch_cell(self, patch: int) -> tuple[float, float, float]:
        """``(x, y, size)`` of a patch cell in the left grid."""
        size = min(6.2 / self.grid_w, 6.0 / self.grid_h)
        row, col = divmod(patch, self.grid_w)
        x0 = 0.6
        y0 = 1.6 + (6.0 - size * self.grid_h) / 2
        return x0 + col * size, y0 + (self.grid_h - 1 - row) * size, size

    def _token_cell(self, token: int) -> tuple[float, float, float]:
        """``(x, y, size)`` of a token cell in the right grid."""
        size = min(5.4 / self.out_w, 5.6 / self.out_h)
        row, col = divmod(token, self.out_w)
        x0 = 9.6 + (5.8 - size * self.out_w) / 2
        y0 = 1.8 + (5.6 - size * self.out_h) / 2
        return x0 + col * size, y0 + (self.out_h - 1 - row) * size, size

    def _stripe(self, token: int, slot: int) -> tuple[float, float, float, float]:
        """``(x, y, w, h)`` of stripe ``slot`` inside token ``token`` (top = first channels)."""
        x, y, size = self._token_cell(token)
        inner = size * 0.84
        pad = (size - inner) / 2
        h = inner / self.group
        return x + pad, y + pad + inner - (slot + 1) * h, inner, h

    def _colour(self, patch: int) -> tuple[float, float, float]:
        token, slot = self.owner[patch]
        hue = (0.58 + token * 0.618034) % 1.0  # golden-ratio hues: neighbours differ
        light = 0.38 + 0.36 * slot / max(1, self.group - 1)
        return colorsys.hls_to_rgb(hue, light, 0.7)

    # ------------------------------------------------------------------ drawing
    def _draw(
        self, placed: set[int], moving: dict[int, float], caption: str, focus: int | None = None
    ) -> FilmCanvas:
        palette = self.palette
        canvas = self.canvas(self.title, caption)
        in_flight = {p for k in moving for p in self.mapping[k]}
        for patch in range(self.grid_h * self.grid_w):
            x, y, s = self._patch_cell(patch)
            token, _ = self.owner[patch]
            gone = token in placed or patch in in_flight
            canvas.box(
                x,
                y,
                s,
                s,
                face=self._colour(patch),
                edge=palette.background,
                alpha=0.18 if gone else 1.0,
                rounded=False,
                linewidth=0.8,
            )
            if self.grid_w <= 8:
                canvas.text(
                    x + s / 2,
                    y + s / 2,
                    str(patch),
                    size=7,
                    colour=palette.background if not gone else palette.faint,
                )
        if focus is not None:
            for patch in self.mapping[focus]:
                x, y, s = self._patch_cell(patch)
                canvas.box(
                    x, y, s, s, face="none", edge=palette.accent, linewidth=2.2, rounded=False
                )
        for token in range(len(self.mapping)):
            x, y, s = self._token_cell(token)
            canvas.box(
                x,
                y,
                s,
                s,
                face=palette.panel,
                edge=palette.token if token in placed else palette.faint,
                linewidth=1.6 if token == focus else 1.0,
                rounded=False,
            )
            if token in placed:
                for slot, patch in enumerate(self.mapping[token]):
                    sx, sy, sw, sh = self._stripe(token, slot)
                    canvas.box(
                        sx,
                        sy,
                        sw,
                        sh,
                        face=self._colour(patch),
                        edge=palette.background,
                        rounded=False,
                        linewidth=0.4,
                    )
        for token, progress in moving.items():
            t = FilmCanvas.ease(progress)
            for slot, patch in enumerate(self.mapping[token]):
                px, py, ps = self._patch_cell(patch)
                sx, sy, sw, sh = self._stripe(token, slot)
                canvas.box(
                    FilmCanvas.lerp(px, sx, t),
                    FilmCanvas.lerp(py, sy, t),
                    FilmCanvas.lerp(ps, sw, t),
                    FilmCanvas.lerp(ps, sh, t),
                    face=self._colour(patch),
                    edge=palette.background,
                    rounded=False,
                    linewidth=0.4,
                )
        canvas.text(3.7, 1.15, self.in_label, size=9.5, colour=palette.patch)
        canvas.text(12.5, 1.15, self.out_label, size=9.5, colour=palette.token)
        canvas.arrow((7.3, 4.6), (9.3, 4.6), colour=palette.muted)
        canvas.text(8.3, 5.05, f"concat {self.group} patches", size=8.5, colour=palette.muted)
        canvas.text(
            8.3,
            4.15,
            f"{self.grid_h * self.grid_w} → {len(self.mapping)} tokens",
            size=8.5,
            colour=palette.muted,
        )
        return canvas

    def frames(self) -> Iterator[tuple[FilmCanvas, int]]:
        placed: set[int] = set()
        yield (
            self._draw(placed, {}, "Colour = destination token; shade = channel slot inside it"),
            1500,
        )
        for token in range(self.animate_tokens):
            members = ", ".join(str(p) for p in self.mapping[token])
            caption = f"token {token} = concat(patches {members})"
            yield self._draw(placed, {}, caption, focus=token), 500
            for step in range(1, 8):
                yield self._draw(placed, {token: step / 7}, caption, focus=token), 60
            placed.add(token)
            yield self._draw(placed, {}, caption, focus=token), 500
        rest = list(range(self.animate_tokens, len(self.mapping)))
        if rest:
            caption = "…every other group moves the same way, in parallel"
            for step in range(1, 11):
                yield self._draw(placed, {k: step / 10 for k in rest}, caption), 60
            placed.update(rest)
        yield (
            self._draw(
                placed,
                {},
                f"{self.group}× fewer tokens, {self.group}× wider features: "
                "the projector maps them to the LLM width",
            ),
            2600,
        )
