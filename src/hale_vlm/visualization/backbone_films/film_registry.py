"""Registry of named backbone films (one entry per GIF the renderer writes)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_core.registry.base import NamedRegistry

T = TypeVar("T")

#: name -> zero-argument factory returning a configured :class:`BackboneFilm`.
BACKBONE_FILMS = NamedRegistry("backbone film")


def register_film(name: str) -> Callable[[Callable[[], T]], Callable[[], T]]:
    """Decorator registering a zero-argument film factory under ``name``.

    Factories (not classes) are registered because one film class is reused with different
    arguments, e.g. :class:`SpaceToDepthFilm` for Qwen3-VL's 2x2 merge and SmolVLM's pixel shuffle.

    Args:
        name: registry key; also the GIF file stem.

    Returns:
        The decorator, which returns the factory unchanged.
    """

    def decorator(factory: Callable[[], T]) -> Callable[[], T]:
        BACKBONE_FILMS.add(name, factory)
        return factory

    return decorator
