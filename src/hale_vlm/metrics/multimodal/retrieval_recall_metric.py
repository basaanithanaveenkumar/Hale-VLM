"""RetrievalRecallMetric: Image-text and text-image retrieval recall at k."""

from __future__ import annotations

import torch
from torch import Tensor

from hale_vlm.metrics.base import VLMMetricBase

from ._common import _l2_normalise


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
        """Clear the accumulated state (start of each evaluation epoch)."""
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
        """Return the metric values accumulated since the last :meth:`reset`.

        Returns:
            A flat mapping of metric name to scalar value.
        """
        if not self._items:
            return (
                {f"i2t_r{k}": 0.0 for k in self.ks}
                | {f"t2i_r{k}": 0.0 for k in self.ks}
                | {"rsum": 0.0}
            )

        n = len(self._items)
        ids, img_embs, txt_embs = zip(*self._items)

        # Stack into matrices [N, D]
        imgs = torch.stack(list(img_embs))  # [N, D]
        txts = torch.stack(list(txt_embs))  # [N, D]

        # Similarity matrices [N, N]
        sim_i2t = imgs @ txts.T  # sim[i, j] = similarity(image_i, text_j)
        sim_t2i = txts @ imgs.T  # sim[i, j] = similarity(text_i, image_j)

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
