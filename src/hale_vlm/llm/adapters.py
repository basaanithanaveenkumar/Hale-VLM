"""LoRA and parameter-freezing helpers for LLM fine-tuning.

The VLM's backbone is configured by HaleBlocks (``hale_core.nn.llm.configure_finetuning``);
these helpers apply the same policy to an arbitrary module and count parameters.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn
from hale_core.nn.llm import get_llm_backbone_spec
from loguru import logger
from peft import LoraConfig, get_peft_model

from hale_vlm.config.sections.llm import LLMConfig

#: LoRA targets when ``lora_target_modules`` is unset and the backbone is ``custom``.
CUSTOM_LORA_TARGETS: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]


def lora_target_modules(cfg: LLMConfig) -> list[str]:
    """Explicit targets, else the registered backbone's defaults, else attention projections."""
    if cfg.lora_target_modules:
        return cfg.lora_target_modules
    if cfg.backbone != "custom":
        targets = list(get_llm_backbone_spec(cfg.backbone).lora_targets)
        if targets:
            return targets
    return list(CUSTOM_LORA_TARGETS)


def apply_lora(model: nn.Module, cfg: LLMConfig) -> nn.Module:
    """Wrap the causal LM with PEFT LoRA adapters; base weights stay frozen."""
    targets = lora_target_modules(cfg)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        "applied LoRA backbone={} targets={} trainable={:,} / {:,} params",
        cfg.backbone,
        targets,
        trainable,
        total,
    )
    return peft_model


def configure_llm_trainability(model: nn.Module, cfg: LLMConfig) -> nn.Module:
    """Freeze or LoRA-wrap the LLM according to config."""
    if cfg.use_lora:
        model = apply_lora(model, cfg)
    elif cfg.freeze_llm:
        model.requires_grad_(False)
    return model


def iter_trainable_parameters(*modules: nn.Module) -> Iterator[nn.Parameter]:
    """Yield every parameter with ``requires_grad`` across ``modules``."""
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                yield param


def count_parameters(*modules: nn.Module) -> tuple[int, int]:
    """Count parameters across ``modules``.

    Returns:
        ``(trainable, total)`` element counts.
    """
    trainable = 0
    total = 0
    for module in modules:
        for param in module.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
    return trainable, total
