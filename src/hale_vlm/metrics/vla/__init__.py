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

from ._common import _normalise as _normalise
from .action_token_accuracy_metric import ActionTokenAccuracyMetric
from .instruction_following_metric import InstructionFollowingMetric

__all__ = [
    "ActionTokenAccuracyMetric",
    "InstructionFollowingMetric",
]
