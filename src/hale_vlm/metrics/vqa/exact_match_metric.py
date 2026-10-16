"""ExactMatchMetric: Case-insensitive normalised exact string match."""

from __future__ import annotations

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _normalise_simple


class ExactMatchMetric(VLMMetricBase):
    """Case-insensitive normalised exact string match.

    The simplest possible accuracy metric: the prediction is correct if it
    exactly equals the reference after normalisation (lowercase, no punctuation,
    collapsed whitespace).

    Interpretation
    --------------
    EM = 1.0 means every prediction matched the reference exactly.
    EM is strict — a prediction of "dog" scores 0 against reference "a dog".
    Use SoftMatchMetric for multi-annotator robustness, OpenEndedF1Metric for
    partial credit.
    """

    name = "exact_match"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._correct: int = 0
        self._total: int = 0

    def update(self, *, prediction: str, answer: str) -> None:  # type: ignore[override]
        """Update with one (prediction, answer) pair."""
        self._correct += int(_normalise_simple(prediction) == _normalise_simple(answer))
        self._total += 1

    def compute(self) -> dict[str, float]:
        return {"exact_match": self._correct / max(self._total, 1)}
