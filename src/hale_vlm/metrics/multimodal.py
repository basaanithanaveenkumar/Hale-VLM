"""Multimodal alignment metrics for Hale-VLM.

These metrics evaluate how well the vision and language components of the model
are aligned — i.e., whether the image embeddings and text embeddings are close
in the shared embedding space.

Metric overview
---------------
EmbeddingAlignmentMetric  — per-pair cosine similarity (like CLIP score)
RetrievalRecallMetric     — image-text retrieval R@1/5/10 on a held-out pool

These metrics require pre-computed embeddings (e.g., from the model's vision
tower and language model final hidden state), not raw text/image predictions.
They are most meaningful when the model has a shared embedding space (CLIP-style
contrastive training) but can also measure cross-modal alignment in VLMs with
projection layers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from hale_vlm.metrics.base import VLMMetricBase


def _l2_normalise(x: Tensor) -> Tensor:
    """Return L2-normalised version of a 1D or 2D tensor."""
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-8))


# ── EmbeddingAlignmentMetric ──────────────────────────────────────────────────

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


# ── RetrievalRecallMetric ─────────────────────────────────────────────────────

class RetrievalRecallMetric(VLMMetricBase):
    """Image-text and text-image retrieval recall at k.

    Accumulates image embeddings and paired text embeddings over a validation
    set, then at ``compute()`` time constructs a similarity matrix and computes
    Recall@K for both retrieval directions.

    This is the standard evaluation protocol for CLIP-style models on MSCOCO
    and Flickr30K.  Each image has exactly one matching text embedding
    (identified by ``pair_id``); ties in retrieval ranking are broken by index.

    Metrics returned
    ----------------
    - **i2t_r{k}**: fraction of images where the correct text is in the top-k
      retrieved texts (image-to-text retrieval).
    - **t2i_r{k}**: fraction of texts where the correct image is in the top-k
      retrieved images (text-to-image retrieval).
    - **rsum**: sum of all 6 R@K values; single number for model selection.

    Interpretation
    --------------
    On MSCOCO 5K test set, strong CLIP models achieve i2t_r1 ≥ 65%, t2i_r1 ≥ 50%.
    rsum ≥ 500 (out of 600 max) is excellent on MSCOCO 5K.
    """

    name = "retrieval"

    def __init__(self, ks: list[int] | None = None) -> None:
        """
        Args:
            ks: Recall thresholds.  Defaults to [1, 5, 10].
        """
        self.ks = ks or [1, 5, 10]
        self.reset()

    def reset(self) -> None:
        # list of (pair_id, image_emb, text_emb)
        self._items: list[tuple[int, Tensor, Tensor]] = []

    def update(  # type: ignore[override]
        self,
        *,
        image_emb: Tensor,
        text_emb: Tensor,
        pair_id: int,
    ) -> None:
        """Accumulate one matched (image, text) pair.

        Args:
            image_emb: 1-D tensor [D].
            text_emb:  1-D tensor [D].
            pair_id:   Integer identifier linking this image to its text.  Must be
                       unique per matching pair in the evaluation set.
        """
        img = _l2_normalise(image_emb.float().detach().cpu().squeeze())
        txt = _l2_normalise(text_emb.float().detach().cpu().squeeze())
        self._items.append((pair_id, img, txt))

    def compute(self) -> dict[str, float]:
        if not self._items:
            return {f"i2t_r{k}": 0.0 for k in self.ks} | \
                   {f"t2i_r{k}": 0.0 for k in self.ks} | {"rsum": 0.0}

        n = len(self._items)
        ids, img_embs, txt_embs = zip(*self._items)

        # Stack into matrices [N, D]
        imgs = torch.stack(list(img_embs))   # [N, D]
        txts = torch.stack(list(txt_embs))   # [N, D]

        # Similarity matrices [N, N]
        sim_i2t = imgs @ txts.T   # sim[i, j] = similarity(image_i, text_j)
        sim_t2i = txts @ imgs.T   # sim[i, j] = similarity(text_i, image_j)

        # Build ground-truth index mapping: each entry i retrieves itself (pair_id matches)
        id_to_idx = {pid: idx for idx, pid in enumerate(ids)}

        def _recall_at_k(sim_matrix: Tensor, k: int) -> float:
            """Fraction of queries where correct item is in top-k results."""
            correct = 0
            for i in range(n):
                # Ground truth is the item with the same pair_id
                gt_idx = id_to_idx[ids[i]]
                # Top-k retrieved indices (excluding self for t2i, same for both)
                topk = sim_matrix[i].topk(min(k, n)).indices.tolist()
                if gt_idx in topk:
                    correct += 1
            return correct / n

        result: dict[str, float] = {}
        rsum = 0.0
        for k in self.ks:
            i2t = _recall_at_k(sim_i2t, k)
            t2i = _recall_at_k(sim_t2i, k)
            result[f"i2t_r{k}"] = i2t * 100.0
            result[f"t2i_r{k}"] = t2i * 100.0
            rsum += i2t * 100.0 + t2i * 100.0
        result["rsum"] = rsum
        return result
