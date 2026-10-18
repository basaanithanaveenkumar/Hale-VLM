"""Film: DeepStack, multi-level ViT features added into the first LLM layers at image positions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from hale_vlm.exceptions import VLMInputError
from hale_vlm.visualization.backbone_films.backbone_film import BackboneFilm
from hale_vlm.visualization.backbone_films.film_canvas import FilmCanvas


class DeepStackFilm(BackboneFilm):
    """The ViT climbs layer by layer; at each tap layer a side merger feeds one LLM layer.

    Matches ``Qwen3VLVisionModel.forward`` and ``Qwen3VLTextModel._deepstack_process``: the
    hidden state after ViT block ``taps[k]`` goes through its own patch merger
    (``deepstack_merger_list[k]``, LayerNorm *after* the 2x2 concat) and is **added** to the
    output of LLM decoder layer ``k``, only at the image-token positions. The final ViT state goes
    through the main merger and *replaces* the ``<|image_pad|>`` embeddings at the LLM input.

    Args:
        vit_depth: number of ViT blocks (27 for the SigLIP-2 so400m encoder of Qwen3-VL).
        taps: ViT block indices whose outputs feed DeepStack (``(8, 16, 24)``).
        llm_layers: LLM layers drawn (the first ``len(taps)`` receive injections).
        sequence: token kinds of a short prompt (``"t"`` text, ``"i"`` image).
        **kwargs: forwarded to :class:`BackboneFilm`.
    """

    name = "qwen3_vl_deepstack"

    def __init__(
        self,
        vit_depth: int = 27,
        taps: Sequence[int] = (8, 16, 24),
        llm_layers: int = 6,
        sequence: str = "ttiiiiiitt",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if any(t >= vit_depth for t in taps):
            raise VLMInputError(f"taps {taps} must be < vit_depth {vit_depth}")
        if len(taps) > llm_layers:
            raise VLMInputError("need at least one drawn LLM layer per tap")
        self.vit_depth = vit_depth
        self.taps = list(taps)
        self.llm_layers = llm_layers
        self.sequence = sequence

    # ------------------------------------------------------------------ geometry
    def _vit_y(self, layer: int) -> float:
        """Bottom y of ViT block ``layer`` (block 0 at the bottom)."""
        return 1.3 + layer * (6.4 / self.vit_depth)

    def _llm_box(self, layer: int) -> tuple[float, float, float, float]:
        h = 6.2 / self.llm_layers
        return 8.6, 1.3 + layer * h, 6.6, h * 0.82

    def _draw(self, climbed: int, injected: int, caption: str, final: bool = False) -> FilmCanvas:
        palette = self.palette
        canvas = self.canvas("DeepStack: ViT features injected into early LLM layers", caption)
        block_h = 6.4 / self.vit_depth
        canvas.text(1.9, 8.0, f"ViT ({self.vit_depth} blocks)", size=10.5, colour=palette.patch)
        for layer in range(self.vit_depth):
            active = layer < climbed
            tap = layer in self.taps
            canvas.box(
                0.9,
                self._vit_y(layer),
                2.0,
                block_h * 0.8,
                face=palette.patch if active else palette.panel,
                edge=palette.tap if tap else palette.faint,
                rounded=False,
                alpha=0.95 if active else 0.6,
                linewidth=1.6 if tap else 0.5,
            )
            if tap:
                canvas.text(
                    0.75,
                    self._vit_y(layer) + block_h * 0.4,
                    str(layer),
                    size=8,
                    ha="right",
                    colour=palette.tap,
                )
        canvas.text(11.9, 8.0, "LLM decoder layers", size=10.5, colour=palette.text)
        for layer in range(self.llm_layers):
            x, y, w, h = self._llm_box(layer)
            canvas.box(x, y, w, h, face=palette.panel, edge=palette.faint, rounded=False)
            canvas.text(
                x - 0.15, y + h / 2, f"L{layer}", size=8.5, ha="right", colour=palette.muted
            )
            cell = w / len(self.sequence)
            for k, kind in enumerate(self.sequence):
                hit = kind == "i" and layer < injected
                colour = palette.tap if hit else (palette.token if kind == "i" else palette.text)
                canvas.box(
                    x + k * cell + 0.04,
                    y + 0.08,
                    cell - 0.08,
                    h - 0.16,
                    face=colour,
                    alpha=0.95 if hit else 0.45,
                    rounded=False,
                    linewidth=0,
                )
        for k, tap in enumerate(self.taps):
            if climbed > tap:
                start = (2.9, self._vit_y(tap) + block_h * 0.4)
                x, y, w, h = self._llm_box(k)
                mid_x = 5.2
                canvas.box(
                    mid_x - 0.85,
                    start[1] - 0.22,
                    1.7,
                    0.44,
                    face=palette.panel,
                    edge=palette.tap,
                    label=f"merger {k}",
                    label_size=7.5,
                    label_colour=palette.tap,
                )
                canvas.arrow(start, (mid_x - 0.85, start[1]), colour=palette.tap, width=1.2)
                canvas.arrow(
                    (mid_x + 0.85, start[1]),
                    (x, y + h / 2),
                    colour=palette.tap,
                    width=1.4 if k < injected else 0.8,
                    alpha=1.0 if k < injected else 0.5,
                )
        if final:
            top_y = self._vit_y(self.vit_depth - 1) + block_h
            gutter = 3.45
            canvas.line(
                [2.9, gutter, gutter],
                [top_y - block_h / 2, top_y - block_h / 2, 0.97],
                colour=palette.token,
                width=1.6,
            )
            canvas.arrow((gutter, 0.97), (4.3, 0.97), colour=palette.token)
            canvas.box(
                4.3,
                0.75,
                1.8,
                0.44,
                face=palette.panel,
                edge=palette.token,
                label="main merger",
                label_size=7.5,
                label_colour=palette.token,
            )
            x, y, w, _ = self._llm_box(0)
            canvas.arrow((6.1, 0.97), (x + w / 2, y), colour=palette.token)
            canvas.text(
                6.3,
                0.72,
                "replaces <|image_pad|> embeddings at the LLM input",
                size=7.5,
                ha="left",
                colour=palette.token,
            )
        return canvas

    def frames(self) -> Iterator[tuple[FilmCanvas, int]]:
        """Yield ``(canvas, hold)``: each drawn frame and how many GIF ticks it stays."""
        yield self._draw(0, 0, "Image patches enter the ViT; prompt tokens wait at the LLM"), 1200
        for layer in range(1, self.vit_depth + 1):
            tapped = (layer - 1) in self.taps
            caption = (
                f"block {layer - 1} is a DeepStack tap: its features are 2×2-merged separately"
                if tapped
                else f"ViT block {layer - 1}"
            )
            yield self._draw(layer, 0, caption), 900 if tapped else 70
        yield (
            self._draw(self.vit_depth, 0, "final features → main merger → LLM input", final=True),
            1600,
        )
        for k in range(1, len(self.taps) + 1):
            yield (
                self._draw(
                    self.vit_depth,
                    k,
                    f"after LLM layer {k - 1}: hidden[image positions] += deepstack[{k - 1}]",
                    final=True,
                ),
                1300,
            )
        yield (
            self._draw(
                self.vit_depth,
                len(self.taps),
                "low-, mid- and high-level visual detail reaches the LLM without extra tokens",
                final=True,
            ),
            2800,
        )
