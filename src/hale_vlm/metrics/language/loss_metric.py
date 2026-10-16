"""LossMetric: Accumulates mean cross-entropy loss and derives perplexity."""

from __future__ import annotations

import math

from hale_vlm.metrics.base import VLMMetricBase


class LossMetric(VLMMetricBase):
    """Accumulates mean cross-entropy loss and derives perplexity.

    Interpretation
    --------------
    - **loss** (cross-entropy): lower is better.  Typical trained VLM: 1.5–3.0.
      Values > 6 suggest the model has not converged or the vocabulary mismatch.
    - **perplexity** (exp(loss)): the effective vocabulary size the model is
      "confused" over.  Perplexity of 10 means on average the model picks from
      10 equally likely next tokens.  Lower is better; strongly model-dependent.

    Usage::

        m = LossMetric()
        for batch in loader:
            loss = model_loss(batch)
            m.update(loss=float(loss))
        print(m.compute())   # {"loss": 2.34, "perplexity": 10.38}
    """

    name = "loss"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total: float = 0.0
        self._count: int = 0

    def update(self, *, loss: float) -> None:  # type: ignore[override]
        """Accumulate one loss value (usually the mean loss of a batch)."""
        self._total += loss
        self._count += 1

    def compute(self) -> dict[str, float]:
        mean = self._total / max(self._count, 1)
        try:
            ppl = math.exp(mean)
        except OverflowError:
            ppl = float("inf")
        return {"loss": mean, "perplexity": ppl}
