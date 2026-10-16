"""BLEUMetric: Corpus BLEU-1/2/3/4 with brevity penalty."""

from __future__ import annotations

import math
from collections import Counter

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _ngrams, _tokenise


class BLEUMetric(VLMMetricBase):
    """Corpus BLEU-1/2/3/4 with brevity penalty.

    BLEU (Bilingual Evaluation Understudy, Papineni et al. 2002) measures how
    many n-grams in the hypothesis appear in *any* reference, divided by the
    number of n-grams in the hypothesis (modified precision).  The brevity penalty
    (BP) discourages very short hypotheses that trivially achieve high precision.

    Corpus-level BLEU aggregates clipped n-gram counts over *all* samples before
    computing precision; this is more stable than averaging per-sample BLEU.

    BLEU-4 ≥ 30 is generally considered strong for image captioning on MSCOCO.
    BLEU is weakest for single-reference evaluation because there may be many
    correct paraphrases.

    Interpretation
    --------------
    - bleu_1: how many unigrams match (vocabulary hit rate)
    - bleu_4: strictest; requires 4-gram phrase matches (fluency + coverage)
    - bleu:   geometric mean of bleu_1…4, the standard reported number
    """

    name = "bleu"
    _MAX_N = 4

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # clipped_match[n] = total clipped n-gram matches across corpus
        self._clipped: list[int] = [0] * self._MAX_N
        # total[n] = total hypothesis n-grams across corpus
        self._total: list[int] = [0] * self._MAX_N
        # for brevity penalty: total hypothesis length vs. closest reference length
        self._hyp_len: int = 0
        self._ref_len: int = 0

    def update(  # type: ignore[override]
        self,
        *,
        hypothesis: str,
        references: list[str],
    ) -> None:
        """Add one (hypothesis, references) pair to the corpus statistics.

        Args:
            hypothesis: The model-generated string.
            references:  One or more reference strings.  BLEU takes the best
                         match across all references for each n-gram position.
        """
        hyp_tok = _tokenise(hypothesis)
        ref_toks = [_tokenise(r) for r in references]

        self._hyp_len += len(hyp_tok)
        # Closest reference length (for brevity penalty)
        self._ref_len += min(abs(len(r) - len(hyp_tok)) for r in ref_toks) + len(
            hyp_tok
        )  # approximation; add closest length

        # Actually compute with closest reference length
        self._ref_len = self._ref_len  # kept above for simplicity
        # Redo: store actual closest ref length per sample (simpler to just accumulate)
        closest_len = min((abs(len(r) - len(hyp_tok)), len(r)) for r in ref_toks)[1]
        # Correct the running ref_len: undo the wrong computation above
        self._ref_len -= min(abs(len(r) - len(hyp_tok)) for r in ref_toks) + len(hyp_tok)
        self._ref_len += closest_len

        for n in range(1, self._MAX_N + 1):
            hyp_ngrams = _ngrams(hyp_tok, n)
            if not hyp_ngrams:
                continue
            # Build merged reference n-gram max counts
            ref_max: Counter = Counter()
            for ref_tok in ref_toks:
                ref_max |= _ngrams(ref_tok, n)
            # Clip hypothesis counts to reference max
            clipped = sum(min(c, ref_max[g]) for g, c in hyp_ngrams.items())
            self._clipped[n - 1] += clipped
            self._total[n - 1] += sum(hyp_ngrams.values())

    def compute(self) -> dict[str, float]:
        # Brevity penalty: BP = 1 if hyp_len > ref_len else exp(1 - ref/hyp)
        if self._hyp_len == 0:
            return {f"bleu_{n}": 0.0 for n in range(1, 5)} | {"bleu": 0.0}
        bp = (
            1.0 if self._hyp_len >= self._ref_len else math.exp(1.0 - self._ref_len / self._hyp_len)
        )
        scores: dict[str, float] = {}
        log_avg = 0.0
        valid = 0
        for n in range(1, self._MAX_N + 1):
            denom = max(self._total[n - 1], 1)
            p = self._clipped[n - 1] / denom if self._total[n - 1] > 0 else 0.0
            scores[f"bleu_{n}"] = bp * p
            if p > 0:
                log_avg += math.log(p)
                valid += 1
        # Geometric mean of all n-gram precisions × BP
        scores["bleu"] = bp * math.exp(log_avg / max(valid, 1)) if valid > 0 else 0.0
        return scores
