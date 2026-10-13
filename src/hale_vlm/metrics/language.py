"""Language generation quality metrics for Hale-VLM.

Metrics in this module evaluate the quality of generated text against reference
text, independently of visual content.  They are appropriate for captioning tasks,
open-ended VQA, and any task where the model produces free-form text output.

All implementations are pure Python with no external NLP library dependencies.

Metric overview
---------------
LossMetric        — cross-entropy training loss + perplexity; used for training
                    monitoring (lower = better language model fit)
BLEUMetric        — n-gram precision with brevity penalty; standard for MT/captioning
                    (BLEU-4 ≥ 30 is strong for image captioning on MSCOCO)
ROUGEMetric       — n-gram / LCS recall; standard for summarisation & captioning
                    (ROUGE-L F ≥ 0.55 is strong on MSCOCO)
METEORMetric      — alignment-based metric with stemming + synonym tolerance;
                    correlates better with human judgement than BLEU for short text
CIDErMetric       — consensus-based TF-IDF metric; reward for matching frequent
                    reference n-grams (CIDEr ≥ 1.0 is strong on MSCOCO)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from typing import Sequence

from hale_vlm.metrics.base import VLMMetricBase


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Count all n-grams in a token list."""
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the Longest Common Subsequence between two token lists.

    Uses the classic O(|a|·|b|) DP algorithm.  For long texts consider limiting
    input length before calling.
    """
    m, n = len(a), len(b)
    # Use two rolling rows to keep memory at O(min(m,n))
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


# ── LossMetric ────────────────────────────────────────────────────────────────

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


# ── BLEUMetric ────────────────────────────────────────────────────────────────

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
        self._ref_len += min(
            abs(len(r) - len(hyp_tok)) for r in ref_toks
        ) + len(hyp_tok)  # approximation; add closest length

        # Actually compute with closest reference length
        self._ref_len = self._ref_len  # kept above for simplicity
        # Redo: store actual closest ref length per sample (simpler to just accumulate)
        closest_len = min(
            (abs(len(r) - len(hyp_tok)), len(r)) for r in ref_toks
        )[1]
        # Correct the running ref_len: undo the wrong computation above
        self._ref_len -= min(
            abs(len(r) - len(hyp_tok)) for r in ref_toks
        ) + len(hyp_tok)
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
            1.0
            if self._hyp_len >= self._ref_len
            else math.exp(1.0 - self._ref_len / self._hyp_len)
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


# ── ROUGEMetric ───────────────────────────────────────────────────────────────

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


# ── METEORMetric ──────────────────────────────────────────────────────────────

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


# ── CIDErMetric ───────────────────────────────────────────────────────────────

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
                    hyp_norm = math.sqrt(sum(v ** 2 for v in hyp_w.values())) or 1e-9
                    ref_norm = math.sqrt(sum(v ** 2 for v in ref_w.values())) or 1e-9
                    sim_sum += dot / (hyp_norm * ref_norm)
                n_score += sim_sum / max(len(ref_ws), 1)

            total_score += n_score / n_docs

        # Average over n-gram orders, scale by 10 (CIDEr convention)
        cider = (total_score / self._N) * 10.0
        return {"cider": cider}
