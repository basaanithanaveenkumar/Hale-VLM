"""One box of a :class:`PipelineFilm`: a backbone stage and the tensor shape it produces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStage:
    """A stage of a VLM backbone pipeline.

    Attributes:
        name: short stage name drawn in the box (``"ViT (SigLIP-2)"``).
        detail: one line under the name (what the stage does).
        shape: tensor shape leaving the stage (``"(N, 1152)"``); shown on the outgoing arrow.
        role: palette role of the box, one of ``patch``, ``token``, ``text``, ``special``,
            ``position``, ``tap``.
    """

    name: str
    detail: str
    shape: str
    role: str = "patch"
