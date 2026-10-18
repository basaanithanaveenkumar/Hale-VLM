"""EmbeddingAlignmentMetric: Per-pair cosine similarity between image and text embeddings."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _l2_normalise


class EmbeddingAlignmentMetric(VLMMetricBase):
    """Per-pair cosine similarity between image and text embeddings.

    Measures the average dot product between L2-normalised image and text
    embeddings across a batch.  This approximates the CLIP score from
    Hessel et al. 2021 ("CLIPScore: A Reference-free Evaluation Metric for
    Image Captioning").

    Interpretation
    --------------
    - **cosine_sim_mean** ∈ [-1, 1]: mean cosine similarity across pairs.
      Random embeddings → ~0.  Well-aligned pairs → > 0.25.
    - **alignment_score** = cosine_sim_mean × 100: scale matches the CLIPScore
      convention where ≥ 25 is considered good alignment.
    - **cosine_sim_std**: standard deviation — low std means the model assigns
      similar similarity to all pairs (under-discriminative); high std means
      the model clearly separates matching from non-matching pairs.
    """

    name = "alignment"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated state (start of each evaluation epoch)."""
        self._sims: list[float] = []

    def update(  # type: ignore[override]
        self,
        *,
        image_emb: Tensor,
        text_emb: Tensor,
    ) -> None:
        """Update with one (image_emb, text_emb) matching pair.

        Args:
            image_emb: 1-D float tensor of shape [D].
            text_emb:  1-D float tensor of shape [D].
        """
        img = _l2_normalise(image_emb.float().detach().cpu().squeeze())
        txt = _l2_normalise(text_emb.float().detach().cpu().squeeze())
        sim = float(torch.dot(img, txt).item())
        self._sims.append(sim)

    def compute(self) -> dict[str, float]:
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        if not self._sims:
            return {"cosine_sim_mean": 0.0, "cosine_sim_std": 0.0, "alignment_score": 0.0}
        mean = sum(self._sims) / len(self._sims)
        var = sum((s - mean) ** 2 for s in self._sims) / len(self._sims)
        std = math.sqrt(var)
        return {
            "cosine_sim_mean": mean,
            "cosine_sim_std": std,
            "alignment_score": mean * 100.0,  # CLIPScore convention
        }
