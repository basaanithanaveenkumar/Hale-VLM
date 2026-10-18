"""NumericalAccuracyMetric: Accuracy on numerical answers (counting, quantitative questions)."""

from __future__ import annotations

import re

from hale_vlm.metrics.base import VLMMetricBase


class NumericalAccuracyMetric(VLMMetricBase):
    """Accuracy on numerical answers (counting, quantitative questions).

    Extracts numbers from predictions and references, then measures:
    - ``num_exact``: numbers match exactly (as strings)
    - ``num_within_5pct``: numbers match within ±5% relative tolerance

    Interpretation
    --------------
    Counting questions are common in VQA ("how many dogs?").  Exact match is strict;
    ±5% tolerance handles rounding differences ("2.5" vs "2.48").
    """

    name = "numerical"

    def __init__(self, tolerance: float = 0.05) -> None:
        """
        Args:
            tolerance: Relative tolerance for within-tolerance match.  Default 5%.
        """
        self.tolerance = tolerance
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated state (start of each evaluation epoch)."""
        self._exact: int = 0
        self._within_tol: int = 0
        self._total: int = 0

    @staticmethod
    def _extract_number(text: str) -> float | None:
        """Extract the first number (int or float) from a string."""
        m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return float(m.group()) if m else None

    def update(self, *, prediction: str, answer: str) -> None:  # type: ignore[override]
        """Update with one (prediction, answer) pair."""
        pred_num = self._extract_number(prediction)
        ans_num = self._extract_number(answer)
        self._total += 1
        if pred_num is None or ans_num is None:
            return
        if abs(pred_num - ans_num) < 1e-9:
            self._exact += 1
            self._within_tol += 1
        elif ans_num != 0 and abs(pred_num - ans_num) / abs(ans_num) <= self.tolerance:
            self._within_tol += 1

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        n = max(self._total, 1)
        return {
            "num_exact": self._exact / n,
            f"num_within_{int(self.tolerance * 100)}pct": self._within_tol / n,
        }
