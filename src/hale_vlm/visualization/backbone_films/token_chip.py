"""One chip of a :class:`TokenStreamFilm`: a token (or a run of identical tokens)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenChip:
    """A token, or ``count`` repetitions of it drawn as one chip (``<image> ×64``).

    Attributes:
        label: the token text as the tokenizer sees it.
        role: palette role (``text``, ``special``, ``token`` for image placeholders, ``position``).
        count: how many consecutive tokens the chip stands for.
        newline: start a new display line after this chip (layout only, not a token).
    """

    label: str
    role: str = "text"
    count: int = 1
    newline: bool = False

    @property
    def caption(self) -> str:
        """Text drawn on the chip."""
        return f"{self.label} ×{self.count}" if self.count > 1 else self.label
