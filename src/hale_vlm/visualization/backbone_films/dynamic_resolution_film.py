"""Film: Qwen3-VL's native dynamic resolution (``smart_resize``) vs a fixed square resize."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from hale_vlm.visualization.backbone_films.backbone_film import BackboneFilm
from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas
from hale_vlm.visualization.backbone_films.synthetic_scene import SyntheticScene


class DynamicResolutionFilm(BackboneFilm):
    """Images of different shapes snap to a grid of ``factor``-pixel cells, one LLM token per cell.

    Left: what a fixed-resolution encoder does (squash every image to ``fixed x fixed``; aspect
    ratio lost, token count constant). Right: Qwen3-VL keeps the aspect ratio, rounds each side to
    a multiple of ``factor = patch_size * merge_size`` (32 px) and lets the token count follow the
    image size, clamped to ``[min_pixels, max_pixels]``.

    Args:
        sizes: input image sizes ``(height, width)`` to cycle through.
        factor: cell side in pixels (one LLM token after the 2x2 merge).
        min_pixels: lower bound on the resized area.
        max_pixels: upper bound on the resized area.
        fixed: side of the square a fixed-resolution encoder would use.
        **kwargs: forwarded to :class:`BackboneFilm`.
    """

    name = "qwen3_vl_dynamic_resolution"

    def __init__(
        self,
        sizes: Sequence[tuple[int, int]] = ((480, 640), (900, 300), (256, 1024)),
        factor: int = 32,
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 512 * 32 * 32,
        fixed: int = 448,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sizes = list(sizes)
        self.factor = factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.fixed = fixed

    @staticmethod
    def smart_resize(
        height: int, width: int, factor: int, min_pixels: int, max_pixels: int
    ) -> tuple[int, int]:
        """Qwen's resize rule: sides divisible by ``factor``, area in bounds, aspect kept.

        Mirrors ``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize`` (the
        notebook checks the two agree).

        Returns:
            ``(new_height, new_width)``.
        """
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt(height * width / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def _panel(
        self,
        canvas: FilmCanvas,
        centre_x: float,
        h_px: int,
        w_px: int,
        cells: tuple[int, int],
        progress: float,
        label: str,
        colour: str,
        source_hw: tuple[int, int],
    ) -> None:
        """Draw a ``source_hw`` image stretched to ``h_px x w_px`` with a ``cells`` grid.

        Rendering the scene at the *source* aspect and stretching it shows the distortion a fixed
        square resize introduces.
        """
        scale = min(6.2 / w_px, 5.2 / h_px)
        w, h = w_px * scale, h_px * scale
        x, y = centre_x - w / 2, 4.6 - h / 2
        scale_src = 120 / max(source_hw)
        pixels = SyntheticScene.render(
            max(8, int(source_hw[0] * scale_src)), max(8, int(source_hw[1] * scale_src))
        )
        canvas.image(pixels, (x, x + w, y, y + h))
        if progress > 0:
            rows, cols = cells
            canvas.grid(x, y, w, h, rows, cols, colour=colour, width=0.7, alpha=min(1.0, progress))
        canvas.box(x, y, w, h, face="none", edge=colour, rounded=False, linewidth=1.6)
        canvas.text(centre_x, 1.35, label, size=9.5, colour=colour)

    def frames(self) -> Iterator[tuple[FilmCanvas, int]]:
        palette = self.palette
        fixed_cells = self.fixed // self.factor
        for height, width in self.sizes:
            new_h, new_w = self.smart_resize(
                height, width, self.factor, self.min_pixels, self.max_pixels
            )
            rows, cols = new_h // self.factor, new_w // self.factor
            title = f"Input image {height}×{width}"
            for step in range(9):
                progress = step / 8
                canvas = self.canvas(
                    title,
                    f"one cell = {self.factor}×{self.factor} px = 4 ViT patches = 1 LLM token",
                )
                canvas.text(4.0, 8.0, "fixed square encoder", size=11, colour=palette.muted)
                canvas.text(
                    12.0, 8.0, "Qwen3-VL smart_resize", size=11, colour=palette.token, weight="bold"
                )
                self._panel(
                    canvas,
                    4.0,
                    self.fixed,
                    self.fixed,
                    (fixed_cells, fixed_cells),
                    progress,
                    f"{self.fixed}×{self.fixed}: aspect lost, {fixed_cells**2} tokens",
                    palette.muted,
                    (height, width),
                )
                shown = int(round(rows * cols * progress))
                self._panel(
                    canvas,
                    12.0,
                    new_h,
                    new_w,
                    (rows, cols),
                    progress,
                    f"{new_h}×{new_w} = {rows}×{cols} grid: {shown} tokens",
                    palette.token,
                    (height, width),
                )
                yield canvas, 1100 if step == 0 else (2400 if step == 8 else 90)
