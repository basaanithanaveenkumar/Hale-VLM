"""VLM dataset registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_core.registry.base import NamedRegistry

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.types import TrainingStage
from hale_vlm.exceptions import DatasetUnavailableError

T = TypeVar("T", bound=VLMDataAdapter)

DATASETS = NamedRegistry("vlm_dataset")


def register_dataset(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        DATASETS.add(name, cls)
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_dataset(name: str) -> type[VLMDataAdapter]:
    """Resolve a dataset adapter class by name.

    Falls back to the phase registry (pretrain / mid-train / post-train corpora), which is kept
    separate so the built-in SmolVLM registry count is unchanged.

    Raises:
        RegistryError: if ``name`` is in neither registry.
    """
    if name not in DATASETS:
        import hale_vlm.data.datasets.phase_datasets  # noqa: F401 — registers on import
        from hale_vlm.data.phase_registry import PHASE_DATASETS

        if name in PHASE_DATASETS:
            return PHASE_DATASETS.get(name)
    return DATASETS.get(name)


def build_dataset(name: str, **kwargs) -> VLMDataAdapter:
    adapter = get_dataset(name)(**kwargs)
    if not adapter.spec.enabled:
        raise DatasetUnavailableError(
            f"dataset {name!r} is disabled in the registry "
            f"(stage={adapter.spec.stage.value}): {adapter.spec.description}"
        )
    return adapter


def list_datasets(
    *,
    stage: TrainingStage | None = None,
    enabled_only: bool = True,
) -> list[str]:
    import hale_vlm.data.datasets.builtin  # noqa: F401 — ensure registration

    names: list[str] = []
    for name, cls in sorted(DATASETS.items.items()):
        spec = cls.spec
        if enabled_only and not spec.enabled:
            continue
        if stage is not None and spec.stage != stage:
            continue
        names.append(name)
    return names
