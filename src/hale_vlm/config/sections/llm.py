"""Language backbone of the VLM (any fine-tunable LLM registered in HaleBlocks)."""

from __future__ import annotations

from hale_core.config.sections.common import StrictModel
from hale_core.config.sections.llm_backbone import LLMBackboneConfig, LoRASettings
from hale_core.nn.llm import is_registered, list_llm_backbones
from pydantic import field_validator

from hale_vlm.exceptions import VLMConfigError


class LLMConfig(StrictModel):
    """Which LLM backs the VLM and how it is fine-tuned.

    ``backbone`` is any name from ``hale_core.nn.llm.list_llm_backbones()`` - e.g. ``qwen3-8b``,
    ``qwen2.5-1.5b``, ``smollm2-360m``, ``llama3.2-3b``, ``phi4-mini``, ``hale-autoregressive``
    (a HaloDiffusionLLM checkpoint, set ``checkpoint``) - or ``custom`` with an explicit
    ``model_id``. Fine-tuning: ``use_lora`` (adapters only) > ``freeze_llm`` (no LLM gradients)
    > full fine-tuning.
    """

    backbone: str = "qwen3-8b"
    model_id: str = "Qwen/Qwen3-8B"
    """Hub repo / local path; left at the default, the backbone's registered repo is used."""
    checkpoint: str | None = None
    """HaloDiffusionLLM checkpoint for ``hale-*`` backbones."""
    dtype: str = "bfloat16"
    freeze_llm: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    lora_train_embeddings: bool = False
    """Also train the token embeddings / LM head (useful when special tokens must be learned)."""
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"
    image_token: str = "<image>"
    reasoning_mode: bool = False

    @field_validator("backbone")
    @classmethod
    def _known_backbone(cls, name: str) -> str:
        if name != "custom" and not is_registered(name):
            raise VLMConfigError(
                f"unknown LLM backbone {name!r}; choose one of {list_llm_backbones()} or 'custom'"
            )
        return name

    @field_validator("dtype")
    @classmethod
    def _known_dtype(cls, dtype: str) -> str:
        if dtype not in ("float32", "float16", "bfloat16"):
            raise VLMConfigError(f"dtype must be float32, float16 or bfloat16, got {dtype!r}")
        return dtype

    @property
    def finetune(self) -> str:
        """``lora`` / ``frozen`` / ``full`` from the two flags (LoRA wins)."""
        return "lora" if self.use_lora else ("frozen" if self.freeze_llm else "full")

    def to_backbone_config(self) -> LLMBackboneConfig:
        """The HaleBlocks :class:`LLMBackboneConfig` this section describes."""
        # model_id overrides the registered repo only when changed from the field default;
        # ``custom`` always uses it, and HaloDiffusionLLM backbones load from ``checkpoint``.
        is_default = self.model_id == LLMConfig.model_fields["model_id"].default
        if self.backbone == "custom":
            model_id: str | None = self.model_id
        elif self.backbone.startswith("hale-") or is_default:
            model_id = None
        else:
            model_id = self.model_id
        return LLMBackboneConfig(
            name=self.backbone,
            model_id=model_id,
            checkpoint=self.checkpoint,
            finetune=self.finetune,
            lora=LoRASettings(
                r=self.lora_r,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                train_embeddings=self.lora_train_embeddings,
            ),
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            trust_remote_code=self.trust_remote_code,
            gradient_checkpointing=self.gradient_checkpointing,
        )
