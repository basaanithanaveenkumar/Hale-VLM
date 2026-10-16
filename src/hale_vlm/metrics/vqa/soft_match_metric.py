"""SoftMatchMetric: VQA v2 soft accuracy — accounts for multi-annotator answer distributions."""

from __future__ import annotations

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _normalise_vqa


class SoftMatchMetric(VLMMetricBase):
    """VQA v2 soft accuracy — accounts for multi-annotator answer distributions.

    The official VQA v2 accuracy formula is::

        acc(pred) = min(count(pred in answers) / 3, 1.0)

    where ``answers`` is the list of 10 human annotations.  A prediction that 3+
    annotators agree with gets full credit; fewer agreements give partial credit.
    This rewards answers that at least some humans would give, even if not the
    majority.

    Interpretation
    --------------
    VQA-Acc ≥ 70% is considered strong on VQA v2.  State-of-the-art large VLMs
    reach 80–85% on VQA v2 val.
    """

    name = "vqa"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total_acc: float = 0.0
        self._count: int = 0

    def update(self, *, prediction: str, answers: list[str]) -> None:  # type: ignore[override]
        """Update with one (prediction, list-of-annotator-answers) triplet.

        Args:
            prediction: Model's answer string.
            answers:    List of human annotator answers (typically 10 for VQA v2).
                        Fewer than 10 is fine; the formula normalises by 3.
        """
        pred_norm = _normalise_vqa(prediction)
        ans_norm = [_normalise_vqa(a) for a in answers]
        match_count = sum(1 for a in ans_norm if a == pred_norm)
        acc = min(match_count / 3.0, 1.0)
        self._total_acc += acc
        self._count += 1

    def compute(self) -> dict[str, float]:
        return {"vqa_accuracy": self._total_acc / max(self._count, 1)}
