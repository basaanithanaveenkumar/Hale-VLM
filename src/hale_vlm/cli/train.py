"""Training CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from hale_core.config.experiment import apply_experiment_layout

from hale_vlm.config import load_vlm_config
from hale_vlm.training.launch import train_vlm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Hale-VLM model")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--run-name", type=str, default=None, help="Optional experiment run name")
    args = parser.parse_args()

    cfg = load_vlm_config(str(args.config))
    if args.resume:
        cfg.train.resume = True
    apply_experiment_layout(cfg, create=not args.resume, name=args.run_name)

    # Local by default; on Ray Train when the config sets train.ray.enabled: true.
    result = train_vlm(cfg)
    if cfg.train.ray.enabled:
        print(f"ray run finished: metrics={result.metrics} checkpoint={result.checkpoint_path}")


if __name__ == "__main__":
    main()
