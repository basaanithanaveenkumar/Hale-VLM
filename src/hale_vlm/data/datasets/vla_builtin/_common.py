"""Shared helpers and constants of :mod:`vla_builtin` (split from the original module)."""

from __future__ import annotations

from typing import TypeVar

from hale_vlm.data.adapters.vla import VLADataAdapter
from hale_vlm.data.types import VLADatasetSpec, VLAStage
from hale_vlm.data.vla.community_paths import SMOLVLA_COMMUNITY_HF_PATHS, hf_path_to_registry_name
from hale_vlm.data.vla_catalog import (
    EMBODIMENT_BY_STAGE,
)
from hale_vlm.data.vla_registry import register_vla_dataset

T = TypeVar("T", bound=VLADataAdapter)


def _make_community_adapter(hf_path: str) -> type[VLADataAdapter]:
    name = hf_path_to_registry_name(hf_path)

    @register_vla_dataset(name)
    class _CommunityAdapter(VLADataAdapter):
        spec = VLADatasetSpec(
            name=name,
            hf_path=hf_path,
            stage=VLAStage.COMMUNITY,
            embodiment=EMBODIMENT_BY_STAGE[VLAStage.COMMUNITY],
            description=f"Community SO-100 dataset: {hf_path}",
            paper_reference="SmolVLA Appendix A.1",
        )

    _CommunityAdapter.__name__ = f"Community_{name.replace('-', '_')}"
    _CommunityAdapter.__qualname__ = _CommunityAdapter.__name__
    return _CommunityAdapter


for _hf_path in SMOLVLA_COMMUNITY_HF_PATHS:
    _make_community_adapter(_hf_path)
