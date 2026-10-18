"""VLA / robotics dataset registry (SmolVLA paper)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_core.registry.base import NamedRegistry

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import VLAStage
from hale_vlm.exceptions import DatasetUnavailableError

T = TypeVar("T", bound=VLADataAdapter)

VLA_DATASETS = NamedRegistry("vla_dataset")


def register_vla_dataset(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a SmolVLA robotics adapter under ``name``.

    Args:
        name: registry key (also set as ``cls.registry_name``).

    Raises:
        RegistryError: if ``name`` is already registered.
    """

    def deco(cls: type[T]) -> type[T]:
        VLA_DATASETS.add(name, cls)
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_vla_dataset(name: str) -> type[VLADataAdapter]:
    """Resolve a VLA dataset adapter class by name, falling back to the VLA phase registry.

    Raises:
        RegistryError: if ``name`` is in neither registry.
    """
    if name not in VLA_DATASETS:
        import hale_vlm.data.vla.phase_datasets  # noqa: F401 — registers on import
        from hale_vlm.data.vla.phase_registry import VLA_PHASE_DATASETS

        if name in VLA_PHASE_DATASETS:
            return VLA_PHASE_DATASETS.get(name)
    return VLA_DATASETS.get(name)


def build_vla_dataset(name: str, **kwargs) -> VLADataAdapter:
    """Instantiate the robotics adapter registered as ``name`` (VLA phase datasets included).

    Args:
        name: registry key.
        **kwargs: forwarded to the adapter constructor.

    Raises:
        DatasetUnavailableError: if the adapter is disabled.
    """
    adapter = get_vla_dataset(name)(**kwargs)
    if not adapter.spec.enabled:
        raise DatasetUnavailableError(
            f"VLA dataset {name!r} is disabled in the registry "
            f"(stage={adapter.spec.stage.value}): {adapter.spec.description}"
        )
    return adapter


def list_vla_datasets(
    *,
    stage: VLAStage | None = None,
    enabled_only: bool = True,
) -> list[str]:
    """Names of registered SmolVLA datasets, sorted.

    Args:
        stage: keep only datasets of this stage (community / simulation / real world).
        enabled_only: drop adapters whose spec is disabled.
    """
    import hale_vlm.data.datasets.vla_builtin  # noqa: F401 — ensure registration

    names: list[str] = []
    for name, cls in sorted(VLA_DATASETS.items.items()):
        spec = cls.spec
        if enabled_only and not spec.enabled:
            continue
        if stage is not None and spec.stage != stage:
            continue
        names.append(name)
    return names
