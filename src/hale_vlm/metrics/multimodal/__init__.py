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

from ._common import _l2_normalise as _l2_normalise
from .embedding_alignment_metric import EmbeddingAlignmentMetric
from .retrieval_recall_metric import RetrievalRecallMetric

__all__ = [
    "EmbeddingAlignmentMetric",
    "RetrievalRecallMetric",
]
