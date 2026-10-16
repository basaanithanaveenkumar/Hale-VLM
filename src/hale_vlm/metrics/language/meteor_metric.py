"""METEORMetric: Approximate METEOR with unigram alignment and fragmentation penalty."""

from __future__ import annotations

from collections import Counter

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _tokenise


class METEORMetric(VLMMetricBase):
    """Approximate METEOR with unigram alignment and fragmentation penalty.

    METEOR (Banerjee & Lavie 2005) was designed to address BLEU's low correlation
    with human judgement on short texts.  Key differences from BLEU:
    - Recall-oriented: rewards hypotheses that cover the reference
    - Fragmentation penalty: punishes hypotheses that match in many short chunks
      rather than in contiguous phrases

    This implementation uses the unigram F-mean (no WordNet synonyms or stemming),
    which approximates the full METEOR score.  Full METEOR requires external resources.

    Parameters
    ----------
    alpha : float
        Weight of precision vs recall in harmonic mean.  Default 0.9 (recall-heavy).
    gamma : float
        Fragmentation penalty weight.  Default 0.5.
    theta : float
        Fragmentation penalty exponent.  Default 3.

    Interpretation
    --------------
    METEOR ∈ [0, 1].  Values ≥ 0.30 on MSCOCO indicate strong captioning performance.
    METEOR correlates better with human judgement than BLEU-4 for short captions.
    """

    name = "meteor"

    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 0.5,
        theta: float = 3.0,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.reset()

    def reset(self) -> None:
        self._total: float = 0.0
        self._count: int = 0

    @staticmethod
    def _count_chunks(hyp: list[str], ref: list[str]) -> int:
        """Count contiguous matching chunks between hyp and ref (greedy left-to-right).

        A chunk is a maximal consecutive run of matched unigrams that appear in
        the same order in both sequences.
        """
        ref_set = set(ref)
        in_chunk = False
        chunks = 0
        for tok in hyp:
            if tok in ref_set:
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
            else:
                in_chunk = False
        return max(chunks, 1)

    def update(  # type: ignore[override]
        self,
        *,
        hypothesis: str,
        reference: str,
    ) -> None:
        """Update with one (hypothesis, reference) pair."""
        hyp = _tokenise(hypothesis)
        ref = _tokenise(reference)
        if not hyp or not ref:
            return

        # Unigram matches (clipped to reference count)
        hyp_c = Counter(hyp)
        ref_c = Counter(ref)
        matches = sum(min(hyp_c[t], ref_c[t]) for t in hyp_c)

        p = matches / len(hyp) if hyp else 0.0
        r = matches / len(ref) if ref else 0.0
        if p + r == 0:
            self._count += 1
            return

        # Harmonic mean weighted by alpha (recall-heavy when alpha < 0.5 for P weight)
        # METEOR definition: Fmean = P*R / (alpha*P + (1-alpha)*R)
        fmean = p * r / (self.alpha * p + (1 - self.alpha) * r)

        # Fragmentation penalty: pen = gamma * (chunks/matches)^theta
        if matches > 0:
            chunks = self._count_chunks(hyp, ref)
            pen = self.gamma * (chunks / matches) ** self.theta
        else:
            pen = 1.0

        score = fmean * (1.0 - pen)
        self._total += max(score, 0.0)
        self._count += 1

    def compute(self) -> dict[str, float]:
        return {"meteor": self._total / max(self._count, 1)}
