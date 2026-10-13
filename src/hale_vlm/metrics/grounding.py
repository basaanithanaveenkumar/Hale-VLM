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

from hale_vlm.metrics.base import VLMMetricBase


# ── helpers ───────────────────────────────────────────────────────────────────

def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes (normalised coordinates).

    IoU = intersection_area / union_area.
    Returns 0.0 if boxes do not overlap or either box is degenerate.
    """
    # Intersection
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ── BoxIoUMetric ──────────────────────────────────────────────────────────────

class BoxIoUMetric(VLMMetricBase):
    """Bounding-box IoU and accuracy-at-threshold for visual grounding.

    The model predicts a bounding box [x1, y1, x2, y2] (normalised 0–1).
    This metric reports:

    - **mean_iou**: average IoU over all samples.
      IoU = 0 means no overlap; IoU = 1 means perfect overlap.
      Typical strong grounding models: mean_iou ≥ 0.60.

    - **acc_at_0.5** (standard): fraction of samples where IoU ≥ 0.5.
      This is the COCO-style "correct" threshold.  Values ≥ 0.80 are strong.

    - **acc_at_0.75** (strict): fraction of samples where IoU ≥ 0.75.
      Requires near-precise localisation.  Values ≥ 0.50 are strong.

    All boxes must be in [x1, y1, x2, y2] format with x1 < x2, y1 < y2,
    normalised to the image size so that all values are in [0, 1].
    """

    name = "box_iou"

    def __init__(
        self,
        thresholds: list[float] | None = None,
    ) -> None:
        """
        Args:
            thresholds: IoU thresholds for accuracy metrics.  Defaults to [0.5, 0.75].
        """
        self.thresholds = thresholds or [0.5, 0.75]
        self.reset()

    def reset(self) -> None:
        self._iou_sum: float = 0.0
        self._count: int = 0
        self._thresh_correct: dict[float, int] = {t: 0 for t in self.thresholds}

    def update(  # type: ignore[override]
        self,
        *,
        pred_box: list[float],
        gt_box: list[float],
    ) -> None:
        """Update with one (predicted box, ground-truth box) pair.

        Args:
            pred_box: [x1, y1, x2, y2] normalised to [0, 1].
            gt_box:   [x1, y1, x2, y2] normalised to [0, 1].
        """
        iou = _box_iou(pred_box, gt_box)
        self._iou_sum += iou
        self._count += 1
        for t in self.thresholds:
            if iou >= t:
                self._thresh_correct[t] += 1

    def compute(self) -> dict[str, float]:
        n = max(self._count, 1)
        result: dict[str, float] = {"mean_iou": self._iou_sum / n}
        for t in self.thresholds:
            key = f"acc_at_{str(t).replace('.', '_')}"
            result[key] = self._thresh_correct[t] / n
        return result


# ── PointingAccuracyMetric ────────────────────────────────────────────────────

class PointingAccuracyMetric(VLMMetricBase):
    """Point-in-box accuracy for pointing / referring tasks.

    Some grounding tasks ask the model to output a single 2D point rather than
    a full bounding box (e.g. "point to the cat").  This metric checks whether
    the predicted point lands inside the ground-truth bounding box.

    This is less strict than IoU (a perfect centre point scores the same as any
    point inside the box) but appropriate when exact localisation is not required.

    Interpretation
    --------------
    Acc = 1.0 means all predicted points were inside the GT box.
    Random baseline on a typical scene: ~15–30% (objects occupy 15–30% of image).
    Strong pointing models: ≥ 80% on RefCOCO/RefCOCO+.
    """

    name = "pointing"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._correct: int = 0
        self._total: int = 0

    def update(  # type: ignore[override]
        self,
        *,
        pred_point: tuple[float, float],
        gt_box: list[float],
    ) -> None:
        """Update with one (predicted point, GT box) pair.

        Args:
            pred_point: (x, y) in [0, 1] normalised coordinates.
            gt_box:     [x1, y1, x2, y2] in [0, 1] normalised coordinates.
        """
        x, y = pred_point
        x1, y1, x2, y2 = gt_box
        inside = (x1 <= x <= x2) and (y1 <= y <= y2)
        self._correct += int(inside)
        self._total += 1

    def compute(self) -> dict[str, float]:
        return {"pointing_acc": self._correct / max(self._total, 1)}
