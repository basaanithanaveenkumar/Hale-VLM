"""Offline, CI-sized Hugging Face assets for the VLM deployment pipelines.

Real deployments start from SigLIP / CLIP + a Hugging Face LLM. For CI and laptops the same
code paths are exercised with *tiny* models of the *same architectures* written to disk with
no network access:

* ``vision/``   - a 2-layer :class:`~transformers.SiglipVisionModel` (32×32 images, 4×4 patches);
* ``llm/``      - a 2-layer :class:`~transformers.LlamaForCausalLM` with GQA and RoPE;
* ``llm/``      - a word-level fast tokenizer (``<pad>``, ``<eos>``, ``<unk>`` + small vocabulary).

Because they are genuine ``transformers`` models, every stage that needs an HF checkpoint
(TensorRT-LLM bundle, mlx-lm conversion) runs unchanged on them.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

#: Words of the toy vocabulary (the tokenizer adds the special tokens in front).
_WORDS = (
    "a",
    "red",
    "green",
    "blue",
    "square",
    "circle",
    "on",
    "white",
    "black",
    "background",
    "the",
    "is",
    "what",
    "color",
    "shape",
    "image",
    "describe",
    "this",
    "it",
    "and",
    ".",
    "?",
)


class TinyVLMAssets:
    """Create (once) and locate the tiny vision tower and language model.

    Args:
        root: output directory; ``root/vision`` and ``root/llm`` are created.
        hidden_size: LLM hidden size (vision uses half of it).
        seed: weight initialisation seed (deterministic assets → reproducible parity numbers).
    """

    def __init__(self, root: str | Path, hidden_size: int = 64, seed: int = 0) -> None:
        self.root = Path(root)
        self.hidden_size = hidden_size
        self.seed = seed

    @property
    def vision_dir(self) -> Path:
        """Directory of the tiny random SigLIP checkpoint."""
        return self.root / "vision"

    @property
    def llm_dir(self) -> Path:
        """Directory of the tiny random LLM checkpoint and its tokenizer."""
        return self.root / "llm"

    def ensure(self) -> TinyVLMAssets:
        """Write any missing asset; existing ones are reused (idempotent, safe for parallel CI)."""
        import torch

        torch.manual_seed(self.seed)
        if not (self.vision_dir / "config.json").exists():
            self._write_vision()
        if not (self.llm_dir / "config.json").exists():
            self._write_llm()
        return self

    def _write_vision(self) -> None:
        from transformers import SiglipVisionConfig, SiglipVisionModel

        config = SiglipVisionConfig(
            hidden_size=self.hidden_size // 2,
            intermediate_size=self.hidden_size,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=32,
            patch_size=4,
        )
        SiglipVisionModel(config).save_pretrained(self.vision_dir)
        logger.info("tiny SigLIP vision tower -> {}", self.vision_dir)

    def _write_llm(self) -> None:
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        vocab = {tok: i for i, tok in enumerate(("<pad>", "<eos>", "<unk>", *_WORDS))}
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
        )
        config = LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=self.hidden_size,
            intermediate_size=self.hidden_size * 2,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            pad_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=False,
        )
        LlamaForCausalLM(config).save_pretrained(self.llm_dir)
        tokenizer.save_pretrained(self.llm_dir)
        logger.info("tiny Llama + tokenizer -> {}", self.llm_dir)

    def run_overrides(self) -> dict:
        """``run_overrides`` that point a VLM run config at these assets."""
        return {
            "model": {
                "vision": {
                    "encoder": "siglip",
                    "model_id": str(self.vision_dir),
                    "image_size": 32,
                    "num_image_tokens": 64,
                },
                "llm": {
                    "backbone": "custom",
                    "model_id": str(self.llm_dir),
                    "dtype": "float32",
                    "use_lora": False,
                    "freeze_llm": True,
                },
            }
        }
