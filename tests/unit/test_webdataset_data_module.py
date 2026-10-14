"""MultimodalDataModule with data.webdataset.enabled: identical batches, lazy conversion."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("webdataset")

from hale_vlm.config import load_vlm_config  # noqa: E402
from hale_vlm.data.multimodal import MultimodalDataModule  # noqa: E402
from tests.helpers.tiny_models import TinyTokenizer  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _cfg(tmp_path, enabled: bool):
    cfg = load_vlm_config(CONFIGS / "qwen3_8b_overfit.yaml")
    cfg.train.batch_size = 2
    cfg.model.max_length = 24
    cfg.model.vision.image_size = 32
    cfg.data.n_overfit_copies = 6
    cfg.data.webdataset = cfg.data.webdataset.model_copy(
        update={
            "enabled": enabled,
            "root": str(tmp_path / "shards"),
            "samples_per_shard": 2,
            "shuffle": False,
        }
    )
    return cfg


def _batches(loader):
    return [{k: v for k, v in b.items()} for b in loader]


def test_val_batches_identical_with_webdataset(tmp_path):
    plain = _batches(MultimodalDataModule(_cfg(tmp_path, False), TinyTokenizer()).val_loader())
    streamed = _batches(MultimodalDataModule(_cfg(tmp_path, True), TinyTokenizer()).val_loader())
    assert len(plain) == len(streamed) == 2  # 4 overfit validation copies, batch 2
    for a, b in zip(plain, streamed):
        assert a.keys() == b.keys()
        for key in a:
            if torch.is_tensor(a[key]):
                assert torch.equal(a[key], b[key]), key
            else:
                assert a[key] == b[key], key


def test_train_split_is_streamed_and_reused(tmp_path):
    cfg = _cfg(tmp_path, True)
    with patch.object(
        MultimodalDataModule, "_load_split", wraps=MultimodalDataModule._load_split, autospec=True
    ) as load:
        first = MultimodalDataModule(cfg, TinyTokenizer()).train_loader()
        assert sum(b["input_ids"].shape[0] for b in first) == 6
        second = MultimodalDataModule(cfg, TinyTokenizer()).train_loader()
        assert len(second) == 3
    assert load.call_count == 1  # the second module streamed the shards, no dataset rebuild


def test_fingerprint_ignores_training_hyperparameters(tmp_path):
    a = MultimodalDataModule(_cfg(tmp_path, True), TinyTokenizer())
    cfg_b = _cfg(tmp_path, True)
    cfg_b.train.lr = 123.0
    cfg_b.model.max_length = 25
    b = MultimodalDataModule(cfg_b, TinyTokenizer())
    assert a.dataset_fingerprint("train") != b.dataset_fingerprint("train")  # max_length matters
    cfg_c = _cfg(tmp_path, True)
    cfg_c.train.lr = 123.0
    assert a.dataset_fingerprint("train") == MultimodalDataModule(
        cfg_c, TinyTokenizer()
    ).dataset_fingerprint("train")
