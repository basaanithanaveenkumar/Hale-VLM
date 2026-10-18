"""BoxIoUMetric: Bounding-box IoU and accuracy-at-threshold for visual grounding."""

from __future__ import annotations

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _box_iou


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
        """Clear the accumulated state (start of each evaluation epoch)."""
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
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        n = max(self._count, 1)
        result: dict[str, float] = {"mean_iou": self._iou_sum / n}
        for t in self.thresholds:
            key = f"acc_at_{str(t).replace('.', '_')}"
            result[key] = self._thresh_correct[t] / n
        return result
