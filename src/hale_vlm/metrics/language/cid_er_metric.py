"""CIDErMetric: Simplified CIDEr-D: consensus-based TF-IDF n-gram similarity."""

from __future__ import annotations

import math
from collections import Counter

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _ngrams, _tokenise


class CIDErMetric(VLMMetricBase):
    """Simplified CIDEr-D: consensus-based TF-IDF n-gram similarity.

    CIDEr (Vedantam et al. 2015) was designed specifically for image captioning.
    Key insight: n-grams that appear in *all* references are more informative
    than n-grams that appear in only one.  It weights n-grams by their TF-IDF
    score where:
    - TF: how often the n-gram appears in this set of references
    - IDF: log(total images / images containing this n-gram + 1)
      — common n-grams across the whole dataset are down-weighted

    This implementation computes IDF from the references seen so far (corpus IDF),
    which means CIDEr values are not directly comparable between evaluation runs
    of different sizes.  For the standard MSCOCO split (5000 val images) the IDF
    stabilises quickly.

    Interpretation
    --------------
    CIDEr ∈ [0, 10] (scaled by ×10 in the original paper).  CIDEr ≥ 1.0 is
    considered strong on MSCOCO.  It correlates highly with human consensus.
    """

    name = "cider"
    _N = 4  # n-gram orders 1..4 (original CIDEr-D)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Each entry: (hypothesis tokens, list of reference token lists)
        """Clear the accumulated state (start of each evaluation epoch)."""
        self._corpus: list[tuple[list[str], list[list[str]]]] = []

    def update(  # type: ignore[override]
        self,
        *,
        hypothesis: str,
        references: list[str],
    ) -> None:
        """Add one (hypothesis, references) pair to the corpus."""
        hyp_tok = _tokenise(hypothesis)
        ref_toks = [_tokenise(r) for r in references]
        self._corpus.append((hyp_tok, ref_toks))

    def _tfidf_weights(
        self, ngram_counts: list[Counter], doc_freq: Counter, n_docs: int
    ) -> list[Counter]:
        """Compute TF-IDF weighted n-gram vectors for each document."""
        weighted = []
        for counts in ngram_counts:
            w: Counter = Counter()
            total = max(sum(counts.values()), 1)
            for gram, cnt in counts.items():
                tf = cnt / total
                idf = math.log((n_docs + 1) / (doc_freq[gram] + 1))
                w[gram] = tf * idf
            weighted.append(w)
        return weighted

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        if not self._corpus:
            return {"cider": 0.0}

        n_docs = len(self._corpus)
        total_score = 0.0

        for n in range(1, self._N + 1):
            # Build document frequency from all references
            doc_freq: Counter = Counter()
            all_ref_ngrams: list[list[Counter]] = []
            all_hyp_ngrams: list[Counter] = []

            for hyp_tok, ref_toks in self._corpus:
                hyp_ng = _ngrams(hyp_tok, n)
                all_hyp_ngrams.append(hyp_ng)
                ref_ngs = [_ngrams(r, n) for r in ref_toks]
                all_ref_ngrams.append(ref_ngs)
                # IDF: n-gram in doc if it appears in any reference
                seen = set()
                for rng in ref_ngs:
                    for gram in rng:
                        if gram not in seen:
                            doc_freq[gram] += 1
                            seen.add(gram)

            # Compute per-sample CIDEr-n score
            n_score = 0.0
            for i, (hyp_tok, ref_toks) in enumerate(self._corpus):
                hyp_w = self._tfidf_weights([all_hyp_ngrams[i]], doc_freq, n_docs)[0]
                ref_ws = self._tfidf_weights(all_ref_ngrams[i], doc_freq, n_docs)

                # Average over references
                sim_sum = 0.0
                for ref_w in ref_ws:
                    # Cosine similarity
                    dot = sum(hyp_w[g] * ref_w[g] for g in hyp_w if g in ref_w)
                    hyp_norm = math.sqrt(sum(v**2 for v in hyp_w.values())) or 1e-9
                    ref_norm = math.sqrt(sum(v**2 for v in ref_w.values())) or 1e-9
                    sim_sum += dot / (hyp_norm * ref_norm)
                n_score += sim_sum / max(len(ref_ws), 1)

            total_score += n_score / n_docs

        # Average over n-gram orders, scale by 10 (CIDEr convention)
        cider = (total_score / self._N) * 10.0
        return {"cider": cider}
