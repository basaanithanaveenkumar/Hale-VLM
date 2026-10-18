"""Hale-VLM configuration: typed run config plus the YAML loader (``inherits`` / ``_base_``)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from hale_core.config.config import Config
from hale_core.registry import register_config
from loguru import logger
from pydantic import ValidationError

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.config.sections.model import VLMModelConfig

register_config("vlm")(VLMRunConfig)


def load_vlm_config(path: str | Path, overrides: Iterable[str] = ()) -> VLMRunConfig:
    """Load, merge and validate a VLM run config.

    Resolves the ``inherits:`` / ``_base_`` chain, applies ``--cfg-options`` style overrides on
    top, then validates the merged mapping into :class:`VLMRunConfig` so no raw dict leaves the
    boundary.

    Args:
        path: YAML or JSON config file.
        overrides: ``key.path=value`` strings (YAML-parsed values), e.g.
            ``["train.steps=10", "model.llm.use_lora=false"]``.

    Returns:
        The validated run config; ``experiment.source_yaml`` records the resolved path.

    Raises:
        ConfigError: for unreadable files, inheritance cycles or malformed overrides.
        pydantic.ValidationError: if the merged config violates the schema.

    Example:
        >>> cfg = load_vlm_config("configs/base.yaml", ["train.steps=5"])  # doctest: +SKIP
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        logger.error("config file not found: {}", resolved)
        raise FileNotFoundError(resolved)
    merged = Config.fromfile(resolved, overrides)
    try:
        cfg = VLMRunConfig.model_validate(merged.to_dict())
    except ValidationError:
        logger.exception("invalid VLM config {}", resolved)
        raise
    if cfg.experiment.source_yaml is not None:
        cfg.experiment.source_yaml = str(resolved)
    logger.info("loaded vlm config variant={} from {}", cfg.variant, resolved)
    return cfg


__all__ = ["VLMModelConfig", "VLMRunConfig", "load_vlm_config"]
