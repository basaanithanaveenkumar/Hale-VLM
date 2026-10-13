"""Abstract base class for all Hale-VLM evaluation metrics.

Every metric follows the accumulator pattern:
  1. ``reset()``  — clear state at the start of each evaluation loop
  2. ``update()`` — called once per sample (or per batch)
  3. ``compute()``— return a flat dict[str, float] of scalar results

This mirrors the torchmetrics interface but is intentionally framework-free:
no dependency on torchmetrics, PyTorch Lightning, or any specific eval harness.
Metrics can be used standalone in scripts, inside VLMMetricsEvaluator, or in
distributed evaluation loops (callers are responsible for aggregation across ranks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VLMMetricBase(ABC):
    """Base for all Hale-VLM evaluation metrics.

    Subclasses must set ``name`` and implement ``reset``, ``update``, and
    ``compute``.  The ``name`` is used as a key prefix when metrics are combined
    in a registry, so keep it short and unique (e.g. ``"bleu"``, ``"vqa"``).
    """

    # Short identifier used for logging and registry lookups.
    name: str = ""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state.  Called at the beginning of each eval epoch."""
        ...

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Ingest one sample (or batch) worth of data.

        Keyword arguments vary by metric — see each subclass docstring for the
        expected keys.  Implementations should be as cheap as possible; heavy
        computation belongs in ``compute()``.
        """
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return a flat mapping of metric_name -> scalar value.

        May be called multiple times during an epoch (e.g. for logging at
        intermediate steps) without calling ``reset()``.  Results reflect
        everything accumulated since the last ``reset()``.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
