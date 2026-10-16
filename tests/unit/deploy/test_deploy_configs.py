"""Every config under configs/deploy builds a valid pipeline (no model is loaded)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hale_core.deploy import DeployRunner

REPO = Path(__file__).resolve().parents[3]
CONFIGS = sorted(p for p in (REPO / "configs" / "deploy").glob("*.yaml"))


def test_configs_exist() -> None:
    names = {p.stem for p in CONFIGS}
    for prefix in ("vision_tensorrt", "vision_jetson", "llm_tensorrt_llm", "vision_mlx", "llm_mlx_lm",
                   "vision_migraphx", "vision_ryzen_ai", "vision_aihub", "vision_executorch", "llm_aihub",
                   "tiny_vision_onnx", "tiny_prefill_mlx"):
        assert any(n.startswith(prefix) for n in names), prefix


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_builds_a_valid_pipeline(path: Path, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(REPO)
    runner = DeployRunner.from_file(path, work_dir=tmp_path)
    pipeline, ctx = runner.build()
    pipeline.validate()
    assert type(ctx.adapter).__module__.startswith("hale_vlm.deploy.adapters")
    assert ctx.device.get("vendor") in {"nvidia", "apple", "amd", "qualcomm", None}
