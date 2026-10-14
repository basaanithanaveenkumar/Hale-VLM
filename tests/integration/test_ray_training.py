"""VLM training on Ray Train: CLI entry dispatch + a real 2-worker DDP run with tiny models."""

from pathlib import Path

import pytest

from hale_vlm.config import load_vlm_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
TESTS_DIR = Path(__file__).resolve().parents[1]


def _tiny_cfg(tmp_path):
    cfg = load_vlm_config(CONFIGS / "qwen3_8b_overfit.yaml")
    cfg.train.steps = 4
    cfg.train.batch_size = 2
    cfg.train.checkpoint_path = str(tmp_path / "ckpt" / "last.pt")
    cfg.train.resume = False
    cfg.logging.backend = "noop"
    cfg.logging.log_file = None
    cfg.experiment.enabled = False
    cfg.eval.every_n_epochs = None
    cfg.device = "cpu"
    cfg.model.max_length = 32
    cfg.model.vision.num_image_tokens = 8
    return cfg


def test_train_vlm_dispatches_to_ray_with_worker_factories(tmp_path, monkeypatch):
    import hale_core.training.ray_entry as ray_entry

    from hale_vlm.training.launch import build_data_module, train_vlm

    captured = {}
    monkeypatch.setattr(
        ray_entry,
        "launch_on_ray",
        lambda cfg, fn, *a, **kw: captured.update(cfg=cfg, args=a, kw=kw) or "ran",
    )
    cfg = _tiny_cfg(tmp_path)
    cfg.train.ray = cfg.train.ray.model_copy(update={"enabled": True, "num_workers": 2})
    assert train_vlm(cfg) == "ran"
    assert captured["kw"]["trainer"] == "vlm"
    assert captured["kw"]["data_module_factory"].func is build_data_module  # built in each worker


@pytest.mark.integration
def test_vlm_trains_on_two_ray_workers(tmp_path):
    ray = pytest.importorskip("ray")
    from hale_core.training.ray_entry import launch_on_ray

    from tests.helpers.ray_vlm import train_with_tiny_models

    cfg = _tiny_cfg(tmp_path)
    cfg.train.ray = cfg.train.ray.model_copy(
        update={"enabled": True, "num_workers": 2, "storage_path": str(tmp_path / "ray")}
    )
    # `tests` is shipped to the workers so they can import the tiny models and the patches.
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"py_modules": [str(TESTS_DIR)]},
    )
    try:
        result = launch_on_ray(cfg, train_with_tiny_models)
    finally:
        ray.shutdown()
    assert result.ok, result.error
    assert result.metrics["global_step"] > 0
    assert result.checkpoint_path is not None and Path(result.checkpoint_path, "last.pt").is_file()
