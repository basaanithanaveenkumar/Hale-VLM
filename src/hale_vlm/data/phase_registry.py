"""Phase-aware VLM dataset registry for Hale-VLM (pretrain / mid-train / post-train).

Separate from the SmolVLM DATASETS registry so phase datasets do not change
the count asserted by existing SmolVLM tests.

Usage::

    from hale_vlm.data.phase_registry import (
        build_phase_dataset,
        list_phase_datasets,
        register_phase_dataset,
    )
    from hale_vlm.data.types import TrainingPhase

    names = list_phase_datasets(phase=TrainingPhase.PRETRAIN)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_core.registry.base import NamedRegistry

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.types import TrainingPhase
from hale_vlm.exceptions import DatasetUnavailableError

T = TypeVar("T", bound=VLMDataAdapter)

PHASE_DATASETS = NamedRegistry("vlm_phase_dataset")


def register_phase_dataset(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a phase-aware VLM adapter under ``name``.

    Args:
        name: registry key (also set as ``cls.registry_name``).

    Raises:
        RegistryError: if ``name`` is already registered.
    """

    def deco(cls: type[T]) -> type[T]:
        PHASE_DATASETS.add(name, cls)
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_phase_dataset(name: str) -> type[VLMDataAdapter]:
    """Return the adapter class registered as ``name``.

    Raises:
        RegistryError: for an unknown name (the message lists the registered ones).
    """
    return PHASE_DATASETS.get(name)


def build_phase_dataset(name: str, **kwargs) -> VLMDataAdapter:
    """Instantiate the adapter registered as ``name``.

    Args:
        name: registry key.
        **kwargs: forwarded to the adapter constructor.

    Raises:
        DatasetUnavailableError: if the adapter is disabled.
    """
    adapter = get_phase_dataset(name)(**kwargs)
    if not adapter.spec.enabled:
        raise DatasetUnavailableError(
            f"phase dataset {name!r} is disabled "
            f"(phase={adapter.spec.phase}): {adapter.spec.description}"
        )
    return adapter


def list_phase_datasets(
    *,
    phase: TrainingPhase | None = None,
    enabled_only: bool = True,
) -> list[str]:
    """Names of registered phase datasets, sorted.

    Args:
        phase: keep only datasets of this training phase.
        enabled_only: drop adapters whose spec is disabled.

    Returns:
        Registry keys; importing this lists triggers adapter registration.
    """
    import hale_vlm.data.datasets.phase_datasets  # noqa: F401

    names: list[str] = []
    for name, cls in sorted(PHASE_DATASETS.items.items()):
        spec = cls.spec
        if enabled_only and not spec.enabled:
            continue
        if phase is not None and spec.phase != phase:
            continue
        names.append(name)
    return names
