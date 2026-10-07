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

T = TypeVar("T", bound=VLMDataAdapter)

PHASE_DATASETS = NamedRegistry("vlm_phase_dataset")


def register_phase_dataset(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        PHASE_DATASETS.add(name, cls)
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_phase_dataset(name: str) -> type[VLMDataAdapter]:
    return PHASE_DATASETS.get(name)


def build_phase_dataset(name: str, **kwargs) -> VLMDataAdapter:
    adapter = get_phase_dataset(name)(**kwargs)
    if not adapter.spec.enabled:
        raise ValueError(
            f"phase dataset {name!r} is disabled "
            f"(phase={adapter.spec.phase}): {adapter.spec.description}"
        )
    return adapter


def list_phase_datasets(
    *,
    phase: TrainingPhase | None = None,
    enabled_only: bool = True,
) -> list[str]:
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
