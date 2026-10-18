"""YesNoAccuracyMetric: Binary accuracy on yes/no questions."""

from __future__ import annotations

from hale_vlm.metrics.base import VLMMetricBase


class YesNoAccuracyMetric(VLMMetricBase):
    """Binary accuracy on yes/no questions.

    Many VQA benchmarks (VQA v2, GQA) include yes/no questions as a separate
    category.  This metric normalises predictions to {yes, no, unknown} and
    measures binary classification accuracy.

    Interpretation
    --------------
    Random baseline is 50%.  Good models reach 85–90% on VQA v2 yes/no split.
    Also tracks class distribution to diagnose yes-bias (models that default to
    "yes" will have a high yes_count but poor no-accuracy).
    """

    name = "yes_no"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated state (start of each evaluation epoch)."""
        self._correct: int = 0
        self._total: int = 0
        self._yes_count: int = 0
        self._no_count: int = 0

    @staticmethod
    def _to_binary(text: str) -> str:
        """Normalise a string to 'yes', 'no', or 'unknown'."""
        t = text.lower().strip().rstrip(".")
        if t in {"yes", "yeah", "yep", "yup", "correct", "right", "true"}:
            return "yes"
        if t in {"no", "nope", "nah", "negative", "wrong", "false"}:
            return "no"
        # Single word starting with yes/no
        if t.startswith("yes"):
            return "yes"
        if t.startswith("no"):
            return "no"
        return "unknown"

    def update(self, *, prediction: str, answer: str) -> None:  # type: ignore[override]
        """Update with one (prediction, answer) pair. answer should be 'yes' or 'no'."""
        pred = self._to_binary(prediction)
        ans = self._to_binary(answer)
        self._total += 1
        if pred == ans:
            self._correct += 1
        if pred == "yes":
            self._yes_count += 1
        elif pred == "no":
            self._no_count += 1

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        return {
            "yes_no_acc": self._correct / max(self._total, 1),
            "yes_count": float(self._yes_count),
            "no_count": float(self._no_count),
            "unknown_count": float(self._total - self._yes_count - self._no_count),
        }
