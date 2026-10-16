"""Shared fixtures of the deployment tests: offline tiny assets and adapter factories."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_CONFIG = str(REPO / "configs" / "base.yaml")


@pytest.fixture(scope="session")
def tiny_assets_dir(tmp_path_factory) -> str:
    from hale_vlm.deploy.testing import TinyVLMAssets

    root = tmp_path_factory.mktemp("tiny_vlm_assets")
    TinyVLMAssets(root).ensure()
    return str(root)


@pytest.fixture
def make_adapter(tiny_assets_dir):
    """``make_adapter(Cls, **kw)`` → an adapter over the offline tiny VLM (max_length 128)."""

    def factory(cls, **kwargs):
        kwargs.setdefault("max_batch", 2)
        kwargs.setdefault("opt_text_len", 16)
        return cls(run_config=RUN_CONFIG, tiny_assets=tiny_assets_dir, run_overrides={"model": {"max_length": 128}},
                   **kwargs)

    return factory
