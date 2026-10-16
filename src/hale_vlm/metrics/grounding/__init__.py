"""Visual grounding and region-level metrics for Hale-VLM.

Grounding tasks require the model to localise objects in images — either by
predicting a bounding box or a point.  These metrics measure spatial accuracy.

Metric overview
---------------
BoxIoUMetric          — IoU between predicted and GT boxes; Acc@0.5 / Acc@0.75
PointingAccuracyMetric— whether a predicted point lands inside the GT box

All coordinates are expected to be normalised to [0, 1] relative to image size.
"""

from __future__ import annotations

from ._common import _box_iou as _box_iou
from .box_io_u_metric import BoxIoUMetric
from .pointing_accuracy_metric import PointingAccuracyMetric

__all__ = [
    "BoxIoUMetric",
    "PointingAccuracyMetric",
]
