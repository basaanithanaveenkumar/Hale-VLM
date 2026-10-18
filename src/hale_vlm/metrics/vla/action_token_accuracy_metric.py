""":class:`ActionTokenAccuracyMetric`."""

from __future__ import annotations

from hale_vlm.metrics.base import VLMMetricBase


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
        """Clear the accumulated state (start of each evaluation epoch)."""
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
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        result: dict[str, float] = {
            "action_token_acc": self._total_correct / max(self._total_tokens, 1),
            "action_sequence_exact_match": self._exact_matches / max(self._seq_count, 1),
        }
        # Per-step breakdown
        for i in range(self.max_report_steps):
            if self._step_total[i] > 0:
                result[f"step_{i}_acc"] = self._step_correct[i] / self._step_total[i]
        return result
