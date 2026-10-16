"""Phase-tagged VLM dataset registrations for Hale-VLM (pretrain/mid-train/post-train).

Registered in PHASE_DATASETS (see data/phase_registry.py), not DATASETS.
"""

from __future__ import annotations

from .alla_vavflan_adapter import AllaVAVFLANAdapter
from .blip_pretrain_adapter import BLIPPretrainAdapter
from .cc3_m_adapter import CC3MAdapter
from .cc12_m_adapter import CC12MAdapter
from .chart_qa_adapter import ChartQAAdapter
from .data_comp1_b_adapter import DataComp1BAdapter
from .infographics_vqa_adapter import InfographicsVQAAdapter
from .l_la_va_instruct665_k_adapter import LLaVAInstruct665KAdapter
from .l_la_va_plus_adapter import LLaVAPlusAdapter
from .l_la_va_pretrain558_k_adapter import LLaVAPretrain558KAdapter
from .laion_aesthetics_adapter import LAIONAestheticsAdapter
from .recap_data_comp1_b_adapter import RecapDataComp1BAdapter
from .red_caps_adapter import RedCapsAdapter
from .science_qa_adapter import ScienceQAAdapter
from .seed_bench_adapter import SEEDBenchAdapter
from .share_gpt4_vpt_adapter import ShareGPT4VPTAdapter
from .text_vqa_adapter import TextVQAAdapter
from .wit_adapter import WITAdapter

__all__ = [
    "AllaVAVFLANAdapter",
    "BLIPPretrainAdapter",
    "CC12MAdapter",
    "CC3MAdapter",
    "ChartQAAdapter",
    "DataComp1BAdapter",
    "InfographicsVQAAdapter",
    "LAIONAestheticsAdapter",
    "LLaVAInstruct665KAdapter",
    "LLaVAPlusAdapter",
    "LLaVAPretrain558KAdapter",
    "RecapDataComp1BAdapter",
    "RedCapsAdapter",
    "SEEDBenchAdapter",
    "ScienceQAAdapter",
    "ShareGPT4VPTAdapter",
    "TextVQAAdapter",
    "WITAdapter",
]
