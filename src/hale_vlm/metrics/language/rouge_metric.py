"""ROUGEMetric: ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""

from __future__ import annotations

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _lcs_length, _ngrams, _tokenise


class ROUGEMetric(VLMMetricBase):
    """ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    ROUGE (Lin 2004) measures recall of reference n-grams / longest common
    subsequences in the hypothesis.  It is the dominant metric for summarisation
    and commonly reported alongside BLEU for captioning.

    - **ROUGE-1**: unigram overlap — vocabulary coverage
    - **ROUGE-2**: bigram overlap — local phrase fluency
    - **ROUGE-L**: LCS-based F1 — sentence-level structure similarity

    ROUGE-L F1 ≥ 0.55 is typical for strong captioning models on MSCOCO.

    Interpretation
    --------------
    High recall (R) = reference words appear in hypothesis (complete but may be verbose).
    High precision (P) = hypothesis words appear in reference (concise but may miss content).
    F1 = harmonic mean; standard reported value.
    """

    name = "rouge"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Accumulators per ROUGE variant: (precision_sum, recall_sum, f1_sum, count)
        self._stats: dict[str, list[float]] = {
            "rouge_1": [0.0, 0.0, 0.0, 0],
            "rouge_2": [0.0, 0.0, 0.0, 0],
            "rouge_l": [0.0, 0.0, 0.0, 0],
        }

    def _prf(self, match: int, pred_len: int, ref_len: int) -> tuple[float, float, float]:
        """Precision, recall, F1 given matched count and lengths."""
        p = match / pred_len if pred_len > 0 else 0.0
        r = match / ref_len if ref_len > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    def update(  # type: ignore[override]
        self,
        *,
        hypothesis: str,
        reference: str,
    ) -> None:
        """Add one (hypothesis, reference) pair.

        For multi-reference evaluation, call update once per reference and take
        the best result, or pass the concatenated references.
        """
        hyp = _tokenise(hypothesis)
        ref = _tokenise(reference)

        # ROUGE-1 (unigram)
        h1 = _ngrams(hyp, 1)
        r1 = _ngrams(ref, 1)
        match1 = sum(min(h1[g], r1[g]) for g in h1)
        p1, rc1, f1_1 = self._prf(match1, sum(h1.values()), sum(r1.values()))
        self._stats["rouge_1"][0] += p1
        self._stats["rouge_1"][1] += rc1
        self._stats["rouge_1"][2] += f1_1
        self._stats["rouge_1"][3] += 1

        # ROUGE-2 (bigram)
        h2 = _ngrams(hyp, 2)
        r2 = _ngrams(ref, 2)
        match2 = sum(min(h2[g], r2[g]) for g in h2)
        p2, rc2, f1_2 = self._prf(match2, sum(h2.values()), sum(r2.values()))
        self._stats["rouge_2"][0] += p2
        self._stats["rouge_2"][1] += rc2
        self._stats["rouge_2"][2] += f1_2
        self._stats["rouge_2"][3] += 1

        # ROUGE-L (longest common subsequence)
        lcs = _lcs_length(hyp, ref)
        pl, rcl, fl = self._prf(lcs, len(hyp), len(ref))
        self._stats["rouge_l"][0] += pl
        self._stats["rouge_l"][1] += rcl
        self._stats["rouge_l"][2] += fl
        self._stats["rouge_l"][3] += 1

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, (p_sum, r_sum, f_sum, n) in self._stats.items():
            n = max(n, 1)
            out[f"{key}_p"] = p_sum / n
            out[f"{key}_r"] = r_sum / n
            out[f"{key}_f"] = f_sum / n
        return out
