"""PointingAccuracyMetric: Point-in-box accuracy for pointing / referring tasks."""

from __future__ import annotations

from hale_vlm.metrics.base import VLMMetricBase


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
