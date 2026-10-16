"""Built-in SmolVLM paper dataset registrations."""

from __future__ import annotations

from hale_vlm.data.catalog import (
    SMOLVLM_ALL_DATASETS,
    SMOLVLM_CONTEXT_DATASETS,
    SMOLVLM_REJECTED_DATASETS,
    SMOLVLM_VIDEO_DATASETS,
    SMOLVLM_VISION_DATASETS,
)

from ._common import BUILTIN_DATASET_NAMES
from .dclm_adapter import DCLMAdapter
from .docmatix_adapter import DocmatixAdapter
from .dolma_books_adapter import DolmaBooksAdapter
from .fine_video_adapter import FineVideoAdapter
from .fine_web_edu_adapter import FineWebEduAdapter
from .l_la_va_one_vision_data_adapter import LLaVAOneVisionDataAdapter
from .l_la_va_video178_k_adapter import LLaVAVideo178KAdapter
from .m4_instruct_data_adapter import M4InstructDataAdapter
from .magpie_adapter import MagpieAdapter
from .mammoth_data_adapter import MammothDataAdapter
from .math_writing_adapter import MathWritingAdapter
from .movie_chat_adapter import MovieChatAdapter
from .share_gpt4_video_adapter import ShareGPT4VideoAdapter
from .smol_lm2_math_adapter import SmolLM2MathAdapter
from .smol_talk_rejected_adapter import SmolTalkRejectedAdapter
from .the_cauldron_adapter import TheCauldronAdapter
from .the_stack_adapter import TheStackAdapter
from .v_ript_adapter import VRiptAdapter
from .video_star_adapter import VideoStarAdapter
from .vista400_k_adapter import Vista400KAdapter

__all__ = [
    "BUILTIN_DATASET_NAMES",
    "DCLMAdapter",
    "DocmatixAdapter",
    "DolmaBooksAdapter",
    "FineVideoAdapter",
    "FineWebEduAdapter",
    "LLaVAOneVisionDataAdapter",
    "LLaVAVideo178KAdapter",
    "M4InstructDataAdapter",
    "MagpieAdapter",
    "MammothDataAdapter",
    "MathWritingAdapter",
    "MovieChatAdapter",
    "SMOLVLM_ALL_DATASETS",
    "SMOLVLM_CONTEXT_DATASETS",
    "SMOLVLM_REJECTED_DATASETS",
    "SMOLVLM_VIDEO_DATASETS",
    "SMOLVLM_VISION_DATASETS",
    "ShareGPT4VideoAdapter",
    "SmolLM2MathAdapter",
    "SmolTalkRejectedAdapter",
    "TheCauldronAdapter",
    "TheStackAdapter",
    "VRiptAdapter",
    "VideoStarAdapter",
    "Vista400KAdapter",
]
