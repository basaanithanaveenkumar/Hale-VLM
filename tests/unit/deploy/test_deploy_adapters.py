"""Hale-VLM deployment adapters on the offline tiny VLM (SigLIP + Llama, real HF classes)."""

from __future__ import annotations

import pytest
import torch
from hale_core.deploy.exceptions import ConfigError
from hale_core.deploy.registry import ADAPTERS as ROOT
from PIL import Image

import hale_vlm.deploy  # noqa: F401 - registers the hale_vlm scope
from hale_vlm.deploy import ADAPTERS, LanguageModelAdapter, VisionEncoderAdapter, VLMPrefillAdapter
from hale_vlm.deploy.calibration import CalibrationImageLoader


def test_registry_scope_and_parent_fallback() -> None:
    assert ROOT.get("hale_vlm.VisionEncoderAdapter") is VisionEncoderAdapter
    assert ADAPTERS.get("VLMPrefillAdapter") is VLMPrefillAdapter
    assert ADAPTERS.get("TorchModuleAdapter") is ROOT.get(
        "TorchModuleAdapter"
    )  # falls back to HaleBlocks
    assert "VisionEncoderAdapter" not in ROOT  # only reachable through the scope


def test_vision_encoder_signature_and_output(make_adapter) -> None:
    adapter = make_adapter(VisionEncoderAdapter)
    sig = adapter.signature()
    assert sig.input_names == ["pixel_values"] and sig.outputs == ("image_embeds",)
    assert sig.shape(sig.inputs[0], "max") == (2, 3, 32, 32)
    module = adapter.export_module(adapter.load())
    (out,) = module(*sig.as_args(sig.sample("trace")))
    assert out.shape == (2, adapter.num_image_tokens, adapter.hidden_size) == (2, 64, 64)


@pytest.mark.parametrize("mode", ["embeds", "ids"])
def test_language_model_signature_and_logits(make_adapter, mode) -> None:
    adapter = make_adapter(LanguageModelAdapter, mode=mode)
    sig = adapter.signature()
    assert sig.input_names[0] == ("inputs_embeds" if mode == "embeds" else "input_ids")
    assert sig.dims["seq"].opt == 64 + 16 and sig.dims["seq"].max == 128
    (logits,) = adapter.export_module(adapter.load())(*sig.as_args(sig.sample("trace")))
    assert logits.shape == (2, 80, adapter.vocab_size)


def test_prefill_matches_hale_vlm_forward_with_leading_placeholders(make_adapter) -> None:
    """The image-first export graph computes exactly what HaleVLM.forward does for that layout."""
    adapter = make_adapter(VLMPrefillAdapter)
    module = adapter.export_module(adapter.load())
    vlm = adapter.vlm
    n = adapter.num_image_tokens
    torch.manual_seed(0)
    pixels = torch.rand(2, 3, 32, 32) * 2 - 1
    text = torch.randint(3, 20, (2, 10))
    mask = torch.ones_like(text)
    (exported,) = module(pixels, text, mask)
    placeholders = torch.full((2, n), vlm.image_token_id)
    with torch.no_grad():
        reference = vlm(
            input_ids=torch.cat([placeholders, text], 1), attention_mask=None, pixel_values=pixels
        ).logits
    torch.testing.assert_close(exported, reference, atol=1e-5, rtol=1e-5)


def test_hf_checkpoint_is_the_merged_text_tower_with_image_token(make_adapter) -> None:
    adapter = make_adapter(LanguageModelAdapter)
    model, tokenizer = adapter.hf_checkpoint(adapter.load())
    assert type(model).__name__ == "LlamaForCausalLM"
    assert tokenizer.convert_tokens_to_ids("<image>") == adapter.vlm.image_token_id
    assert model.get_input_embeddings().num_embeddings == adapter.vocab_size


def test_checkpoint_loads_into_the_full_vlm(make_adapter, tmp_path) -> None:
    source = make_adapter(VisionEncoderAdapter)
    state = {k: v.clone() for k, v in source.vlm.state_dict().items()}
    key = next(k for k in state if k.startswith("projector."))
    state[key] = torch.full_like(state[key], 0.125)
    torch.save({"model_state_dict": state}, tmp_path / "vlm.pt")
    adapter = make_adapter(VisionEncoderAdapter, checkpoint=str(tmp_path / "vlm.pt"))
    part = adapter.load()
    assert torch.all(dict(part.named_parameters())[key].eq(0.125))


def test_calibration_images_follow_the_training_transform(make_adapter, tmp_path) -> None:
    for i, color in enumerate(("red", "green", "blue")):
        Image.new("RGB", (48, 40), color).save(tmp_path / f"img{i}.png")
    batches = list(CalibrationImageLoader(tmp_path, image_size=32, batch_size=2).batches(4))
    assert [b.shape[0] for b in batches] == [2, 1] and batches[0].shape[1:] == (3, 32, 32)
    assert float(batches[0].min()) >= -1.0 and float(batches[0].max()) <= 1.0
    vision = make_adapter(VisionEncoderAdapter, calibration_images=str(tmp_path))
    assert next(vision.calibration_batches(2))["pixel_values"].shape == (1, 3, 32, 32)
    lm = make_adapter(LanguageModelAdapter, calibration_images=str(tmp_path))
    batch = next(lm.calibration_batches(2))
    assert batch["inputs_embeds"].shape == (1, 64 + 16, 64) and batch["attention_mask"].shape == (
        1,
        80,
    )
    prefill = make_adapter(VLMPrefillAdapter, calibration_images=str(tmp_path))
    assert set(next(prefill.calibration_batches(1))) == {
        "pixel_values",
        "input_ids",
        "attention_mask",
    }
    assert make_adapter(VisionEncoderAdapter).calibration_batches(2) is None  # random fallback


def test_config_errors(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        VisionEncoderAdapter(run_config=str(tmp_path / "missing.yaml")).run_cfg
    with pytest.raises(ConfigError, match="calibration image folder"):
        CalibrationImageLoader(tmp_path / "nope", 32)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="no images"):
        next(CalibrationImageLoader(tmp_path / "empty", 32).batches(1))
