"""Hale-VLM evaluation metrics.

All metrics follow the reset() / update(**kwargs) / compute() -> dict[str, float]
accumulator pattern defined in VLMMetricBase.

Organised by task family:

Language (captioning / generation quality)
  LossMetric, BLEUMetric, ROUGEMetric, METEORMetric, CIDErMetric

VQA (visual question answering)
  ExactMatchMetric, SoftMatchMetric, YesNoAccuracyMetric,
  NumericalAccuracyMetric, OpenEndedF1Metric

Grounding (spatial localisation)
  BoxIoUMetric, PointingAccuracyMetric

Multimodal alignment
  EmbeddingAlignmentMetric, RetrievalRecallMetric

VLA (vision-language-action, robot policy evaluation)
  InstructionFollowingMetric, ActionTokenAccuracyMetric
"""

from hale_vlm.metrics.base import VLMMetricBase
from hale_vlm.metrics.grounding import (
    BoxIoUMetric,
    PointingAccuracyMetric,
)
from hale_vlm.metrics.language import (
    BLEUMetric,
    CIDErMetric,
    LossMetric,
    METEORMetric,
    ROUGEMetric,
)
from hale_vlm.metrics.multimodal import (
    EmbeddingAlignmentMetric,
    RetrievalRecallMetric,
)
from hale_vlm.metrics.vla import (
    ActionTokenAccuracyMetric,
    InstructionFollowingMetric,
)
from hale_vlm.metrics.vqa import (
    ExactMatchMetric,
    NumericalAccuracyMetric,
    OpenEndedF1Metric,
    SoftMatchMetric,
    YesNoAccuracyMetric,
)

__all__ = [
    # base
    "VLMMetricBase",
    # language
    "LossMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "METEORMetric",
    "CIDErMetric",
    # vqa
    "ExactMatchMetric",
    "SoftMatchMetric",
    "YesNoAccuracyMetric",
    "NumericalAccuracyMetric",
    "OpenEndedF1Metric",
    # grounding
    "BoxIoUMetric",
    "PointingAccuracyMetric",
    # multimodal
    "EmbeddingAlignmentMetric",
    "RetrievalRecallMetric",
    # vla
    "InstructionFollowingMetric",
    "ActionTokenAccuracyMetric",
]
