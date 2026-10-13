"""Visual Question Answering evaluation metrics for Hale-VLM.

VQA metrics measure how well a model answers visual questions.  They differ
from captioning metrics in that answers are typically short (1-3 words) and
there are multiple valid answer annotations per question.

Metric overview
---------------
ExactMatchMetric       — binary: normalised string equality
SoftMatchMetric        — VQA v2 soft accuracy with multi-annotator consensus
YesNoAccuracyMetric    — binary classification accuracy on yes/no questions
NumericalAccuracyMetric— number extraction + exact / ±5% tolerance
OpenEndedF1Metric      — token-level F1 (SQuAD-style) for open-ended questions

References
----------
- VQA v2: Goyal et al. "Making the V in VQA Matter", CVPR 2017
- SQuAD F1: Rajpurkar et al. "SQuAD: 100,000+ Questions...", EMNLP 2016
"""

from __future__ import annotations

import re
import string
from collections import Counter

from hale_vlm.metrics.base import VLMMetricBase


# ── shared normalisation ──────────────────────────────────────────────────────

_ARTICLES = {"a", "an", "the"}
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hedve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id": "I'd", "Idve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't",
    "itd": "it'd", "itdve": "it'd've", "itll": "it'll", "its": "it's",
    "itd": "it'd", "lets": "let's", "maam": "ma'am", "mayve": "may've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "shant": "shan't", "shed": "she'd", "shedve": "she'd've", "shell": "she'll",
    "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "thats": "that's", "theyd": "they'd", "theydve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "wedve": "we'd've",
    "well": "we'll", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whodve": "who'd've",
    "wholl": "who'll", "whomll": "whom'll", "whos": "who's", "whove": "who've",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "yall": "y'all", "youd": "you'd", "youdve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}


def _normalise_vqa(text: str) -> str:
    """Apply VQA v2 answer normalisation.

    Steps (matching official VQA eval script):
    1. Lower-case
    2. Remove period at end if not a number
    3. Replace commas in numbers with nothing (1,000 → 1000)
    4. Remove articles (a, an, the) at word boundaries
    5. Strip punctuation except apostrophes
    6. Strip leading/trailing whitespace
    """
    text = text.lower().strip()
    # Remove trailing period if not part of a number
    if text.endswith(".") and not re.match(r".*\d\.$", text):
        text = text[:-1]
    # Normalise number commas
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation (keep apostrophes in words)
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_simple(text: str) -> str:
    """Light normalisation: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


# ── ExactMatchMetric ──────────────────────────────────────────────────────────

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


# ── SoftMatchMetric ───────────────────────────────────────────────────────────

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


# ── YesNoAccuracyMetric ───────────────────────────────────────────────────────

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
        return {
            "yes_no_acc": self._correct / max(self._total, 1),
            "yes_count": float(self._yes_count),
            "no_count": float(self._no_count),
            "unknown_count": float(self._total - self._yes_count - self._no_count),
        }


# ── NumericalAccuracyMetric ───────────────────────────────────────────────────

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
        n = max(self._total, 1)
        return {
            "num_exact": self._exact / n,
            f"num_within_{int(self.tolerance * 100)}pct": self._within_tol / n,
        }


# ── OpenEndedF1Metric ─────────────────────────────────────────────────────────

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
        return {"token_f1": self._total_f1 / max(self._count, 1)}
