"""Shared helpers and constants of :mod:`grounding` (split from the original module)."""

from __future__ import annotations


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
