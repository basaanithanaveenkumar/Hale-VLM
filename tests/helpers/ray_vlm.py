"""Train the VLM with tiny stand-in HF models inside Ray workers.

``unittest.mock.patch`` only affects the process that applies it, so the patches that
``tests/integration/test_trainer.py`` uses must be re-applied inside every Ray worker.
"""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from unittest.mock import patch

from tests.helpers.tiny_models import TinyCausalLM, TinyTokenizer, TinyVisionModel


def tiny_hf_patches() -> ExitStack:
    """Context manager replacing every HF download with a tiny local model / tokenizer."""
    stack = ExitStack()
    stack.enter_context(
        patch(
            "hale_vlm.vision.encoders.SiglipVisionModel.from_pretrained",
            TinyVisionModel.from_pretrained,
        )
    )
    stack.enter_context(
        patch(
            "hale_vlm.llm.backbones.AutoModelForCausalLM.from_pretrained",
            TinyCausalLM.from_pretrained,
        )
    )
    tokenizer = lambda *_a, **_k: TinyTokenizer()  # noqa: E731
    stack.enter_context(patch("hale_vlm.llm.backbones.AutoTokenizer.from_pretrained", tokenizer))
    stack.enter_context(patch("hale_vlm.data.multimodal.AutoTokenizer.from_pretrained", tokenizer))
    return stack


def train_with_tiny_models(cfg) -> None:
    """``train_fn`` for ``launch_on_ray``: the real ``vlm`` trainer, tiny models."""
    from hale_core.training import train

    from hale_vlm.training.evaluator import VLMEvaluator
    from hale_vlm.training.launch import build_data_module

    with tiny_hf_patches():
        train(
            cfg,
            None,
            trainer="vlm",
            data_module_factory=partial(build_data_module, cfg),
            evaluator_factory=VLMEvaluator,
        )
