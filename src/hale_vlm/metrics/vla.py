"""Vision-Language-Action (VLA) task metrics for Hale-VLM.

When Hale-VLM is used as the language backbone for a robot policy (VLA mode),
the model must not only generate coherent text but also produce action-relevant
outputs — either as discrete tokens representing robot actions, or as structured
instruction-following behaviour.

Metric overview
---------------
InstructionFollowingMetric  — keyword/phrase matching + task success rate
ActionTokenAccuracyMetric   — discrete action token accuracy (VLA-token tasks)

These metrics complement standard language metrics (BLEU/ROUGE) by measuring
task-oriented correctness rather than linguistic similarity.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from hale_vlm.metrics.base import VLMMetricBase


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


# ── InstructionFollowingMetric ────────────────────────────────────────────────

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
        "a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "from",
        "of", "and", "or", "but", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "it", "its", "that",
        "this", "then", "so", "if", "when", "where", "what", "which", "who",
        "how", "all", "each", "every", "any", "some", "no", "not", "very",
        "just", "also", "up", "out", "into", "onto", "over", "under", "now",
        "please", "robot", "you", "your", "go", "get", "put", "pick",
    }

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
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
        n = max(self._count, 1)
        return {
            "keyword_hit_rate": self._kw_hits / n,
            "task_success_rate": self._success / n,
            "instruction_follow_f1": self._f1_sum / n,
        }


# ── ActionTokenAccuracyMetric ─────────────────────────────────────────────────

class ActionTokenAccuracyMetric(VLMMetricBase):
    """Per-token and sequence-level accuracy for discretised action outputs.

    When a VLA model tokenises robot actions as discrete integers (e.g., using a
    codebook or bin-discretisation of continuous actions), this metric measures:

    - **action_token_acc**: per-token accuracy averaged across all positions and
      all samples.  Each token position is counted independently.
      Useful for measuring how well the model learns the action vocabulary.

    - **action_sequence_exact_match**: fraction of sequences where every token
      matches.  This is the strictest measure — a single wrong token fails the
      whole sequence.

    - **action_token_acc_per_step**: average token accuracy per timestep position
      (returned as ``step_0_acc``, ``step_1_acc``, etc. up to ``max_report_steps``).
      Reveals whether the model gets early steps right but degrades later.

    Interpretation
    --------------
    - action_token_acc ≥ 0.85: model has learned most of the action vocabulary.
    - action_sequence_exact_match depends strongly on sequence length; ≥ 0.30 for
      long sequences (8+ tokens) is strong.
    """

    name = "action_tokens"

    def __init__(self, max_report_steps: int = 8) -> None:
        """
        Args:
            max_report_steps: Report per-step accuracy up to this many timesteps.
        """
        self.max_report_steps = max_report_steps
        self.reset()

    def reset(self) -> None:
        self._total_correct: int = 0
        self._total_tokens: int = 0
        self._exact_matches: int = 0
        self._seq_count: int = 0
        # Per-step: list of (correct, total) per position index
        self._step_correct: list[int] = [0] * self.max_report_steps
        self._step_total: list[int] = [0] * self.max_report_steps

    def update(  # type: ignore[override]
        self,
        *,
        pred_tokens: list[int],
        gt_tokens: list[int],
    ) -> None:
        """Update with one (predicted token sequence, ground-truth token sequence).

        Sequences are compared position-by-position up to the shorter length.
        Trailing tokens in the longer sequence count as incorrect.

        Args:
            pred_tokens: List of predicted discrete action token IDs.
            gt_tokens:   List of ground-truth discrete action token IDs.
        """
        max_len = max(len(pred_tokens), len(gt_tokens))
        all_match = True
        for i in range(max_len):
            p = pred_tokens[i] if i < len(pred_tokens) else -1
            g = gt_tokens[i] if i < len(gt_tokens) else -2  # different sentinel
            correct = int(p == g)
            self._total_correct += correct
            self._total_tokens += 1
            if not correct:
                all_match = False
            if i < self.max_report_steps:
                self._step_correct[i] += correct
                self._step_total[i] += 1

        self._exact_matches += int(all_match)
        self._seq_count += 1

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {
            "action_token_acc": self._total_correct / max(self._total_tokens, 1),
            "action_sequence_exact_match": self._exact_matches / max(self._seq_count, 1),
        }
        # Per-step breakdown
        for i in range(self.max_report_steps):
            if self._step_total[i] > 0:
                result[f"step_{i}_acc"] = self._step_correct[i] / self._step_total[i]
        return result
