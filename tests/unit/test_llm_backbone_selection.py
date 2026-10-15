"""Any registered fine-tunable LLM can back the VLM; the config maps onto HaleBlocks' backbone config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hale_core.nn.llm import get_llm_backbone_spec, list_llm_backbones

from hale_vlm.config.sections.llm import LLMConfig
from hale_vlm.llm.backbones import resolve_llm_config


@pytest.mark.parametrize("name", list_llm_backbones("hf"))
def test_every_registered_hf_backbone_is_selectable(name: str) -> None:
    cfg = resolve_llm_config(LLMConfig(backbone=name))
    assert cfg.model_id == get_llm_backbone_spec(name).model_id
    backbone = cfg.to_backbone_config()
    assert backbone.name == name and backbone.finetune == "lora"


def test_finetune_modes_and_lora_settings_map_through() -> None:
    cfg = LLMConfig(backbone="phi4-mini", use_lora=True, lora_r=8, lora_alpha=16, lora_train_embeddings=True)
    b = cfg.to_backbone_config()
    assert (b.finetune, b.lora.r, b.lora.alpha, b.lora.train_embeddings) == ("lora", 8, 16, True)
    assert LLMConfig(use_lora=False, freeze_llm=True).finetune == "frozen"
    assert LLMConfig(use_lora=False, freeze_llm=False).finetune == "full"


def test_hale_backbone_uses_its_checkpoint_and_custom_uses_model_id() -> None:
    hale = LLMConfig(backbone="hale-autoregressive", checkpoint="ckpt/last.pt").to_backbone_config()
    assert hale.checkpoint == "ckpt/last.pt" and hale.model_id is None
    custom = LLMConfig(backbone="custom", model_id="org/my-llm").to_backbone_config()
    assert custom.model_id == "org/my-llm"
    with pytest.raises(ValidationError):
        LLMConfig(backbone="hale-autoregressive").to_backbone_config()  # no checkpoint


def test_unknown_backbone_and_reasoning_alias() -> None:
    with pytest.raises(ValidationError, match="unknown LLM backbone"):
        LLMConfig(backbone="gpt-5")
    assert resolve_llm_config(LLMConfig(backbone="deepseek-r1-qwen-7b")).reasoning_mode is True


@pytest.mark.parametrize("config", ["qwen3_5_4b_base_lora", "gemma4_e4b_lora", "cosmos_reason2_2b_lora"])
def test_2026_backbone_example_configs_load(config: str) -> None:
    from pathlib import Path

    from hale_vlm.config import load_vlm_config

    path = Path(__file__).resolve().parents[2] / "configs" / "examples" / f"{config}.yaml"
    llm = resolve_llm_config(load_vlm_config(str(path)).model.llm)
    spec = get_llm_backbone_spec(llm.backbone)
    assert spec.is_multimodal_checkpoint and llm.model_id == spec.model_id
    assert llm.to_backbone_config().lora.train_embeddings
