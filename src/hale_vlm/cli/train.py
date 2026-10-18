"""Training CLI entry point (``hale-vlm-train``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hale_core.config.experiment import apply_experiment_layout
from loguru import logger

from hale_vlm.config import load_vlm_config
from hale_vlm.training.launch import train_vlm


def main() -> None:
    """Parse arguments, load the config with overrides and run (Ray or local) training.

    Example:
        ``hale-vlm-train configs/overfit.yaml --cfg-options train.steps=20``
    """
    parser = argparse.ArgumentParser(description="Train a Hale-VLM model")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--run-name", type=str, default=None, help="Optional experiment run name")
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config overrides, e.g. train.steps=10 model.llm.use_lora=false",
    )
    args = parser.parse_args()

    cfg = load_vlm_config(args.config, args.cfg_options)
    if args.resume:
        cfg.train.resume = True
    apply_experiment_layout(cfg, create=not args.resume, name=args.run_name)

    # Local by default; on Ray Train when the config sets train.ray.enabled: true.
    result = train_vlm(cfg)
    if cfg.train.ray.enabled:
        logger.info(
            "ray run finished: metrics={} checkpoint={}", result.metrics, result.checkpoint_path
        )


if __name__ == "__main__":
    main()
