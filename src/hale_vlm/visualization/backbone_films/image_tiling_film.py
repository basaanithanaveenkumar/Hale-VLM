"""Film: SmolVLM image splitting into position-tagged tiles plus a global thumbnail."""

from __future__ import annotations

from collections.abc import Iterator

from hale_vlm.exceptions import VLMInputError
from hale_vlm.visualization.backbone_films.backbone_film import BackboneFilm
from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas
from hale_vlm.visualization.backbone_films.synthetic_scene import SyntheticScene


class ImageTilingFilm(BackboneFilm):
    """The image is resized to a multiple of the tile size, cut into tiles, plus a thumbnail.

    Mirrors ``SmolVLMImageProcessor``: the longest edge is first rescaled to ``size.longest_edge``,
    each side is then rounded *up* to a multiple of ``tile`` (``max_image_size``), the result is
    cut into ``rows x cols`` tiles (``<row_r_col_c>``), and the whole image squashed to one tile
    is appended as the global view (``<global-img>``). Every tile costs ``tokens_per_tile`` LLM
    tokens after pixel shuffle.

    Args:
        image_hw: original image ``(height, width)``.
        resized_hw: size after resizing, a multiple of ``tile`` on both sides.
        tile: tile side in pixels (512 for SmolVLM-256M/500M, 384 for SmolVLM-2.2B).
        tokens_per_tile: LLM tokens per tile after pixel shuffle (64 or 81).
        **kwargs: forwarded to :class:`BackboneFilm`.
    """

    name = "smolvlm_image_tiling"

    def __init__(
        self,
        image_hw: tuple[int, int] = (1100, 1600),
        resized_hw: tuple[int, int] = (1536, 2048),
        tile: int = 512,
        tokens_per_tile: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if resized_hw[0] % tile or resized_hw[1] % tile:
            raise VLMInputError(f"resized_hw {resized_hw} must be a multiple of tile {tile}")
        self.image_hw = image_hw
        self.resized_hw = resized_hw
        self.tile = tile
        self.rows, self.cols = resized_hw[0] // tile, resized_hw[1] // tile
        self.tokens_per_tile = tokens_per_tile
        self.scene = SyntheticScene.render(240, int(240 * resized_hw[1] / resized_hw[0]))

    def _image_rect(self) -> tuple[float, float, float, float]:
        """Where the resized image sits on the left of the frame: ``(x, y, w, h)``."""
        scale = min(6.6 / self.resized_hw[1], 5.6 / self.resized_hw[0])
        w, h = self.resized_hw[1] * scale, self.resized_hw[0] * scale
        return 0.6, 4.4 - h / 2, w, h

    def _tile_slot(self, index: int) -> tuple[float, float, float]:
        """Destination ``(x, y, size)`` of tile ``index`` on the right (global view is the last)."""
        count = self.rows * self.cols + 1
        per_row = min(count, 5)
        size = min(1.25, 6.4 / per_row - 0.2)
        row, col = divmod(index, per_row)
        return 8.6 + col * (size + 0.22), 6.6 - row * (size + 0.55), size

    def _crop(self, row: int, col: int):
        """Pixels of tile ``(row, col)`` from the synthetic scene."""
        h, w = self.scene.shape[:2]
        return self.scene[
            row * h // self.rows : (row + 1) * h // self.rows,
            col * w // self.cols : (col + 1) * w // self.cols,
        ]

    def _draw(self, phase: str, lifted: int, progress: float, caption: str) -> FilmCanvas:
        palette = self.palette
        canvas = self.canvas("SmolVLM: split the image into tiles + a global view", caption)
        x, y, w, h = self._image_rect()
        canvas.image(self.scene, (x, x + w, y, y + h), alpha=0.35 if lifted else 1.0)
        canvas.box(x, y, w, h, face="none", edge=palette.patch, rounded=False, linewidth=1.5)
        canvas.text(
            x + w / 2,
            y - 0.35,
            f"{self.image_hw[0]}×{self.image_hw[1]} → resized "
            f"{self.resized_hw[0]}×{self.resized_hw[1]}",
            size=9,
            colour=palette.patch,
        )
        if phase != "image":
            canvas.grid(x, y, w, h, self.rows, self.cols, colour=palette.accent, width=1.1)
        tile_w, tile_h = w / self.cols, h / self.rows
        for index in range(min(lifted + 1, self.rows * self.cols + 1)):
            moving = index == lifted
            if moving and progress <= 0:
                continue
            t = FilmCanvas.ease(progress) if moving else 1.0
            tx, ty, size = self._tile_slot(index)
            if index < self.rows * self.cols:
                row, col = divmod(index, self.cols)
                sx, sy = x + col * tile_w, y + (self.rows - 1 - row) * tile_h
                pixels, sw, sh = self._crop(row, col), tile_w, tile_h
                tag, colour = f"<row_{row + 1}_col_{col + 1}>", palette.position
            else:
                sx, sy, sw, sh, pixels = x, y, w, h, self.scene
                tag, colour = "<global-img>", palette.special
            cx, cy = FilmCanvas.lerp(sx, tx, t), FilmCanvas.lerp(sy, ty, t)
            cw, ch = FilmCanvas.lerp(sw, size, t), FilmCanvas.lerp(sh, size, t)
            canvas.image(pixels, (cx, cx + cw, cy, cy + ch))
            canvas.box(cx, cy, cw, ch, face="none", edge=colour, rounded=False, linewidth=1.4)
            if t >= 1.0:
                canvas.text(
                    tx + size / 2, ty - 0.2, tag, size=6.6, colour=colour, family="DejaVu Sans Mono"
                )
        total = self.rows * self.cols + 1
        done = min(lifted, total)
        canvas.text(
            8.6,
            1.25,
            f"{done} / {total} images × {self.tokens_per_tile} tokens = "
            f"{done * self.tokens_per_tile} image tokens",
            size=10.5,
            ha="left",
            colour=palette.token,
            weight="bold",
        )
        return canvas

    def frames(self) -> Iterator[tuple[FilmCanvas, int]]:
        yield (
            self._draw("image", 0, 0.0, "resize so both sides are multiples of the tile size"),
            1500,
        )
        yield (
            self._draw(
                "grid",
                0,
                0.0,
                f"cut into {self.rows}×{self.cols} tiles of {self.tile}×{self.tile} px",
            ),
            1400,
        )
        total = self.rows * self.cols + 1
        for index in range(total):
            last = index == total - 1
            caption = (
                "the whole image, squashed to one tile, is appended as the global view"
                if last
                else "each tile becomes its own encoder input, tagged with its grid position"
            )
            for step in range(1, 7):
                yield self._draw("grid", index, step / 6, caption), 60
            yield self._draw("grid", index + 1, 0.0, caption), 900 if last or index == 0 else 220
        yield (
            self._draw("grid", total, 0.0, "detail from the tiles, layout from the global view"),
            2800,
        )
