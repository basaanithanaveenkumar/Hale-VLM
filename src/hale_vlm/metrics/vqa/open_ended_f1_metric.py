"""OpenEndedF1Metric: Token-level F1 between prediction and best-matching reference."""

from __future__ import annotations

from collections import Counter

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _normalise_simple


class OpenEndedF1Metric(VLMMetricBase):
    """Token-level F1 between prediction and best-matching reference.

    This is the SQuAD reading comprehension metric applied to VQA open-ended
    answers.  It gives partial credit: a prediction of "the big brown dog" scores
    partial credit against "brown dog" (they share "brown" and "dog").

    The score is the best F1 across all provided references, so multi-reference
    evaluation is natively supported.

    Interpretation
    --------------
    Token F1 ≥ 0.60 is typical for strong open-ended VQA models.
    Token F1 is more forgiving than ExactMatch and better correlated with human
    judgement for answers that allow paraphrasing.
    """

    name = "open_f1"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated state (start of each evaluation epoch)."""
        self._total_f1: float = 0.0
        self._count: int = 0

    @staticmethod
    def _token_f1(prediction: str, reference: str) -> float:
        pred_toks = _normalise_simple(prediction).split()
        ref_toks = _normalise_simple(reference).split()
        if not pred_toks or not ref_toks:
            return float(pred_toks == ref_toks)
        common = Counter(pred_toks) & Counter(ref_toks)
        n_common = sum(common.values())
        if n_common == 0:
            return 0.0
        precision = n_common / len(pred_toks)
        recall = n_common / len(ref_toks)
        return 2 * precision * recall / (precision + recall)

    def update(self, *, prediction: str, references: list[str]) -> None:  # type: ignore[override]
        """Update with one prediction and one or more reference answers.

        Args:
            prediction: Model output string.
            references: List of reference answer strings (take best match).
        """
        best = max(self._token_f1(prediction, ref) for ref in references)
        self._total_f1 += best
        self._count += 1

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        return {"token_f1": self._total_f1 / max(self._count, 1)}
