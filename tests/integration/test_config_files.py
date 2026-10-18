"""Every shipped run config validates against the schema; overrides apply on top."""

from pathlib import Path

import pytest

from hale_vlm.config import load_vlm_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
RUN_CONFIGS = sorted(CONFIGS.glob("*.yaml")) + sorted((CONFIGS / "examples").glob("*.yaml"))


@pytest.mark.integration
@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_run_config_validates(path):
    assert load_vlm_config(path).variant


@pytest.mark.integration
def test_cfg_options_override_inherited_values():
    cfg = load_vlm_config(
        CONFIGS / "overfit.yaml", ["train.steps=7", "model.llm.use_lora=false", "device=cpu"]
    )
    assert cfg.train.steps == 7
    assert cfg.model.llm.use_lora is False
    assert cfg.device == "cpu"
    assert cfg.train.scheduler.warmup_steps == 5


@pytest.mark.integration
def test_malformed_override_is_a_config_error():
    from hale_core.exceptions import ConfigError

    with pytest.raises(ConfigError):
        load_vlm_config(CONFIGS / "base.yaml", ["no_equals_sign"])


@pytest.mark.integration
def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_vlm_config(CONFIGS / "does_not_exist.yaml")
