"""Registry of vision-to-LLM token connectors, keyed by ``projector_type``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_core.registry.base import NamedRegistry
from torch import nn

T = TypeVar("T", bound=type[nn.Module])

VISION_CONNECTORS = NamedRegistry("vision_connector", scope="hale_vlm")
"""``projector_type`` -> connector class exposing ``from_config(vision_dim, llm_dim, cfg)``."""


def register_vision_connector(name: str) -> Callable[[T], T]:
    """Class decorator registering a connector under the ``projector_type`` value ``name``.

    The class must provide ``from_config(vision_dim, llm_dim, cfg)`` returning an instance, so
    :func:`build_vision_connector` can create it without knowing its constructor.

    Args:
        name: the ``model.vision.projector_type`` value that selects this connector.

    Raises:
        RegistryError: if ``name`` is already registered.
    """

    def decorator(cls: T) -> T:
        VISION_CONNECTORS.add(name, cls)
        return cls

    return decorator
