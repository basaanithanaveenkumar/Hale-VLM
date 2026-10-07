"""VLA training-phase dataset registry for Hale-VLM.

Separate from the SmolVLA ``VLA_DATASETS`` registry so existing tests are
unaffected.  Phase datasets live in ``VLA_PHASE_DATASETS`` and are
auto-imported only when :func:`list_vla_phase_datasets` is called.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_core.registry.base import NamedRegistry

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import VLATrainingPhase

T = TypeVar("T", bound=VLADataAdapter)

VLA_PHASE_DATASETS = NamedRegistry("vla_phase_dataset")


def register_vla_phase_dataset(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator: register a VLA phase-dataset adapter class by name."""

    def deco(cls: type[T]) -> type[T]:
        VLA_PHASE_DATASETS.add(name, cls)
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def build_vla_phase_dataset(name: str, **kwargs) -> VLADataAdapter:
    """Instantiate a registered VLA phase dataset by name."""
    adapter_cls = VLA_PHASE_DATASETS.get(name)
    return adapter_cls(**kwargs)


def list_vla_phase_datasets(
    *,
    phase: VLATrainingPhase | str | None = None,
    enabled_only: bool = True,
) -> list[VLADataAdapter]:
    """Return adapter classes from the VLA phase registry.

    Triggers a one-time auto-import of ``hale_vlm.data.vla.phase_datasets``.

    Args:
        phase: If given, filter to datasets matching this training phase.
        enabled_only: Exclude adapters whose spec has ``enabled=False``.
    """
    # Auto-import registrations
    import importlib

    importlib.import_module("hale_vlm.data.vla.phase_datasets")

    adapters = []
    for _name, adapter_cls in VLA_PHASE_DATASETS.items():
        spec = getattr(adapter_cls, "spec", None)
        if enabled_only and spec is not None and not spec.enabled:
            continue
        if phase is not None:
            spec_phase = getattr(spec, "phase", None)
            if spec_phase is None or str(spec_phase) != str(phase):
                continue
        adapters.append(adapter_cls)
    return adapters


__all__ = [
    "VLA_PHASE_DATASETS",
    "build_vla_phase_dataset",
    "list_vla_phase_datasets",
    "register_vla_phase_dataset",
]
