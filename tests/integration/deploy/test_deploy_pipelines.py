"""Real deployment runs of the offline tiny VLM through ``hale-vlm-deploy``'s configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hale_core.deploy import DeployRunner

from hale_vlm.cli.deploy import main as deploy_main

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "configs" / "deploy"

#: MLX graph conversion of these parts needs three HaleBlocks MLX-bridge changes (1-element conv
#: padding lists, deferred shape guards in torch.export, symbolic-int comparison ops); see
#: docs/deploy.md. The tests turn into XPASS once the framework has them.


def run(name: str, tmp_path: Path, assets: str, monkeypatch):
    monkeypatch.chdir(REPO)
    return DeployRunner.from_file(
        DEPLOY / f"{name}.yaml", [f"adapter.tiny_assets={assets}"], tmp_path
    ).run()


@pytest.mark.parametrize(
    "name,output",
    [
        ("tiny_vision_onnx_ort", "image_embeds"),
        ("tiny_language_model_onnx_ort", "logits"),
        ("tiny_prefill_onnx_ort", "logits"),
    ],
)
def test_onnx_runtime_parity_at_all_shapes(
    name, output, tmp_path, tiny_assets_dir, monkeypatch
) -> None:
    report = run(name, tmp_path, tiny_assets_dir, monkeypatch)
    assert report.ok
    run_dir = next(tmp_path.iterdir())
    parity = json.loads((run_dir / "onnx_parity" / "parity.json").read_text())
    assert set(parity) == {"min", "opt", "max"}
    assert all(corner[output]["max_abs"] < 1e-4 for corner in parity.values())


def test_hf_checkpoint_route_for_llm_runtimes(tmp_path, tiny_assets_dir, monkeypatch) -> None:
    report = run("tiny_text_lm_mlx_lm", tmp_path, tiny_assets_dir, monkeypatch)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["HuggingFaceCheckpointStage"] == "ok"
    hf_dir = tmp_path / "vlm_tiny_text_lm" / "HuggingFaceCheckpointStage" / "hf"
    config = json.loads((hf_dir / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLM"] and (hf_dir / "tokenizer.json").exists()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert (
        AutoTokenizer.from_pretrained(hf_dir).convert_tokens_to_ids("<image>")
        == config["vocab_size"] - 1
    )
    assert AutoModelForCausalLM.from_pretrained(hf_dir).config.vocab_size == config["vocab_size"]


def test_language_model_mlx_bridge_parity(tmp_path, tiny_assets_dir, monkeypatch) -> None:
    pytest.importorskip("mlx.core")
    report = run("tiny_language_model_mlx_fp32", tmp_path, tiny_assets_dir, monkeypatch)
    assert report.ok
    parity = json.loads(
        (tmp_path / "vlm_tiny_language_model" / "mlx_function_parity" / "parity.json").read_text()
    )
    assert set(parity) == {"min", "opt", "max"} and all(
        c["logits"]["max_abs"] < 1e-4 for c in parity.values()
    )


@pytest.mark.parametrize("name", ["tiny_vision_mlx_fp32", "tiny_prefill_mlx_fp32"])
def test_vision_and_prefill_mlx_parity(name, tmp_path, tiny_assets_dir, monkeypatch) -> None:
    """SigLIP patch conv and transformers' mask guards convert through the MLX bridge."""
    pytest.importorskip("mlx.core")
    assert run(name, tmp_path, tiny_assets_dir, monkeypatch).ok


def test_cli_lists_vlm_adapters(capsys) -> None:
    assert deploy_main(["list"]) == 0
    assert "hale_vlm" not in capsys.readouterr().err
