""":class:`InstructionFollowingMetric`."""

from __future__ import annotations

from collections import Counter

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _normalise


class InstructionFollowingMetric(VLMMetricBase):
    """Measures instruction-following quality for robot action generation.

    Three complementary signals:

    1. **keyword_hit_rate**: fraction of instruction action-keywords that appear
       in the generated response.  Instruction keywords are extracted as verbs
       and object nouns (heuristically: tokens that are not stop-words).
       Measures whether the model "understood" what was asked.

    2. **task_success_rate**: binary label provided by the evaluation harness
       (e.g., from a robot simulator or human annotator) indicating whether the
       generated response led to task completion.  This is the ground truth signal.

    3. **instruction_follow_f1**: token-level F1 between generated output and
       instruction (adapted SQuAD-style).  Measures linguistic proximity to
       the original instruction — useful when no success signal is available.

    Interpretation
    --------------
    - keyword_hit_rate ≥ 0.70: model mentions most key action concepts.
    - task_success_rate ≥ 0.60: model generates useful robot actions more than
      60% of the time.
    - instruction_follow_f1 ≥ 0.50: reasonable linguistic adherence.
    """

    name = "instruction_following"

    # Common stop-words to ignore when extracting keywords from instructions
    _STOP = {
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "of",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "it",
        "its",
        "that",
        "this",
        "then",
        "so",
        "if",
        "when",
        "where",
        "what",
        "which",
        "who",
        "how",
        "all",
        "each",
        "every",
        "any",
        "some",
        "no",
        "not",
        "very",
        "just",
        "also",
        "up",
        "out",
        "into",
        "onto",
        "over",
        "under",
        "now",
        "please",
        "robot",
        "you",
        "your",
        "go",
        "get",
        "put",
        "pick",
    }

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated state (start of each evaluation epoch)."""
        self._kw_hits: float = 0.0
        self._success: int = 0
        self._f1_sum: float = 0.0
        self._count: int = 0

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful tokens from instruction (simple stop-word filter)."""
        tokens = _normalise(text).split()
        return [t for t in tokens if t not in self._STOP and len(t) > 2]

    def _token_f1(self, pred: str, ref: str) -> float:
        pred_toks = _normalise(pred).split()
        ref_toks = _normalise(ref).split()
        if not pred_toks or not ref_toks:
            return float(pred_toks == ref_toks)
        common = Counter(pred_toks) & Counter(ref_toks)
        n_common = sum(common.values())
        if n_common == 0:
            return 0.0
        p = n_common / len(pred_toks)
        r = n_common / len(ref_toks)
        return 2 * p * r / (p + r)

    def update(  # type: ignore[override]
        self,
        *,
        generated: str,
        instruction: str,
        success: bool,
    ) -> None:
        """Update with one (generated, instruction, success) triple.

        Args:
            generated:   The model's generated text / action description.
            instruction: The original instruction given to the model.
            success:     Whether the action achieved the goal (from env/annotator).
        """
        # Keyword hit rate
        keywords = self._extract_keywords(instruction)
        if keywords:
            gen_tokens = set(_normalise(generated).split())
            hits = sum(1 for kw in keywords if kw in gen_tokens)
            self._kw_hits += hits / len(keywords)
        else:
            self._kw_hits += 1.0  # no extractable keywords → trivially satisfied

        # Task success
        self._success += int(success)

        # Instruction follow F1
        self._f1_sum += self._token_f1(generated, instruction)

        self._count += 1

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        n = max(self._count, 1)
        return {
            "keyword_hit_rate": self._kw_hits / n,
            "task_success_rate": self._success / n,
            "instruction_follow_f1": self._f1_sum / n,
        }
