"""Q-Former / gated cross-attention wired into the full HaleVLM."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hale_core.registry import get_trainer

from hale_vlm.config import load_vlm_config
from hale_vlm.data.multimodal import MultimodalDataModule
from hale_vlm.models.vlm import HaleVLM
from hale_vlm.training.evaluator import VLMEvaluator
from hale_vlm.vision.connectors import GatedCrossAttentionProjector, QFormer
from tests.helpers.tiny_models import TinyCausalLM, TinyTokenizer, TinyVisionModel

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

HALE_CASES = [
    ("qwen3_8b_qformer.yaml", QFormer),
    ("qwen3_8b_gated_cross_attention.yaml", GatedCrossAttentionProjector),
]


@pytest.fixture
def hf_mocks():
    with (
        patch(
            "hale_vlm.vision.encoders.SiglipVisionModel.from_pretrained",
            TinyVisionModel.from_pretrained,
        ),
        patch(
            "hale_vlm.llm.backbones.AutoModelForCausalLM.from_pretrained",
            TinyCausalLM.from_pretrained,
        ),
        patch(
            "hale_vlm.llm.backbones.AutoTokenizer.from_pretrained",
            lambda *_args, **_kwargs: TinyTokenizer(),
        ),
        patch(
            "hale_vlm.data.multimodal.AutoTokenizer.from_pretrained",
            lambda *_args, **_kwargs: TinyTokenizer(),
        ),
    ):
        yield


def _shrink_hale(cfg) -> None:
    """Fit the connector to the 32-dim tiny vision tower / LLM used in tests."""
    cfg.model.vision.qformer.num_queries = 4
    cfg.model.vision.qformer.hidden_dim = 16
    cfg.model.vision.qformer.num_heads = 4
    cfg.model.vision.gated_cross_attention.num_latents = 4
    cfg.model.vision.gated_cross_attention.hidden_dim = 16
    cfg.model.vision.gated_cross_attention.num_heads = 4


@pytest.mark.integration
@pytest.mark.parametrize(("config_name", "connector_cls"), HALE_CASES)
def test_hale_vlm_uses_connector(hf_mocks, config_name, connector_cls):
    cfg = load_vlm_config(CONFIGS / config_name)
    _shrink_hale(cfg)
    model = HaleVLM(vocab_size=0, cfg=cfg)

    assert isinstance(model.projector, connector_cls)
    assert model.num_image_tokens == 4
    images = torch.randn(2, 3, 224, 224)
    assert model.encode_images(images).shape == (2, 4, model.llm.hidden_size)
    assert any(p.requires_grad for p in model.projector.parameters())


@pytest.mark.integration
@pytest.mark.parametrize(("config_name", "connector_cls"), HALE_CASES)
def test_hale_trainer_learns_with_connector(hf_mocks, tmp_path, config_name, connector_cls):
    cfg = load_vlm_config(CONFIGS / "qwen3_8b_overfit.yaml")
    connector_cfg = load_vlm_config(CONFIGS / config_name)
    cfg.model.vision.projector_type = connector_cfg.model.vision.projector_type
    _shrink_hale(cfg)
    cfg.train.steps = 12
    cfg.train.batch_size = 2
    cfg.train.checkpoint_path = str(tmp_path / "last.pt")
    cfg.train.checkpoint_every_epoch = False
    cfg.train.resume = False
    cfg.logging.backend = "noop"
    cfg.logging.log_file = None
    cfg.experiment.enabled = False
    cfg.eval.every_n_epochs = None
    cfg.device = "cpu"
    cfg.model.max_length = 32

    trainer = get_trainer("vlm")(
        cfg,
        data_module=MultimodalDataModule(cfg, tokenizer=None),
        evaluator=VLMEvaluator(),
    )
    model, _tokenizer, losses = trainer.fit()

    assert isinstance(model.projector, connector_cls)
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(torch.tensor(losses)))


@pytest.mark.integration
def test_hale_vlm_passes_token_ids_aligned_with_spliced_images(hf_mocks):
    """Backbones whose layers read token identity (Gemma 4) get ids matching the spliced sequence."""
    cfg = load_vlm_config(CONFIGS / "qwen3_8b_qformer.yaml")
    _shrink_hale(cfg)
    model = HaleVLM(vocab_size=0, cfg=cfg)
    seen = {}
    forward = model.llm.core.forward

    def spy(**kwargs):
        seen.update(kwargs)
        return forward(**kwargs)

    model.llm.core.forward = spy
    image_id, n = model.image_token_id, model.num_image_tokens
    ids = torch.tensor([[5, *[image_id] * n, 6, 7], [5, 6, 7, 8, 9, 10, 11]])
    model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(2, 3, 224, 224))
    token_ids, embeds = seen["token_ids"], seen["inputs_embeds"]
    assert token_ids.shape == embeds.shape[:2]
    assert token_ids[0, 1 : 1 + n].eq(image_id).all() and token_ids[0, 0] == 5
    assert torch.equal(token_ids[1], ids[1])
