"""Language generation quality metrics for Hale-VLM.

Metrics in this module evaluate the quality of generated text against reference
text, independently of visual content.  They are appropriate for captioning tasks,
open-ended VQA, and any task where the model produces free-form text output.

All implementations are pure Python with no external NLP library dependencies.

Metric overview
---------------
LossMetric        — cross-entropy training loss + perplexity; used for training
                    monitoring (lower = better language model fit)
BLEUMetric        — n-gram precision with brevity penalty; standard for MT/captioning
                    (BLEU-4 ≥ 30 is strong for image captioning on MSCOCO)
ROUGEMetric       — n-gram / LCS recall; standard for summarisation & captioning
                    (ROUGE-L F ≥ 0.55 is strong on MSCOCO)
METEORMetric      — alignment-based metric with stemming + synonym tolerance;
                    correlates better with human judgement than BLEU for short text
CIDErMetric       — consensus-based TF-IDF metric; reward for matching frequent
                    reference n-grams (CIDEr ≥ 1.0 is strong on MSCOCO)
"""

from __future__ import annotations

from ._common import _lcs_length as _lcs_length
from ._common import _ngrams as _ngrams
from ._common import _tokenise as _tokenise
from .bleu_metric import BLEUMetric
from .cid_er_metric import CIDErMetric
from .loss_metric import LossMetric
from .meteor_metric import METEORMetric
from .rouge_metric import ROUGEMetric

__all__ = [
    "BLEUMetric",
    "CIDErMetric",
    "LossMetric",
    "METEORMetric",
    "ROUGEMetric",
]
