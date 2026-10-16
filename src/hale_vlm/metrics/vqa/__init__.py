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

from ._common import _ARTICLES as _ARTICLES
from ._common import _CONTRACTIONS as _CONTRACTIONS
from ._common import _normalise_simple as _normalise_simple
from ._common import _normalise_vqa as _normalise_vqa
from .exact_match_metric import ExactMatchMetric
from .numerical_accuracy_metric import NumericalAccuracyMetric
from .open_ended_f1_metric import OpenEndedF1Metric
from .soft_match_metric import SoftMatchMetric
from .yes_no_accuracy_metric import YesNoAccuracyMetric

__all__ = [
    "ExactMatchMetric",
    "NumericalAccuracyMetric",
    "OpenEndedF1Metric",
    "SoftMatchMetric",
    "YesNoAccuracyMetric",
]
