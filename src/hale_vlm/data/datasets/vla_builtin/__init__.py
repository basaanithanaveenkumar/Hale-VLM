"""Built-in SmolVLA paper VLA dataset registrations."""

from __future__ import annotations

from hale_vlm.data.vla_catalog import SMOLVLA_COMMUNITY_DATASETS

from ._common import T
from ._common import _make_community_adapter as _make_community_adapter
from .libero_adapter import LiberoAdapter
from .meta_world_mt50_adapter import MetaWorldMT50Adapter
from .svla_so100_pick_place_adapter import SvlaSO100PickPlaceAdapter
from .svla_so100_sorting_adapter import SvlaSO100SortingAdapter
from .svla_so100_stacking_adapter import SvlaSO100StackingAdapter
from .svla_so101_pick_place_adapter import SvlaSO101PickPlaceAdapter

__all__ = [
    "LiberoAdapter",
    "MetaWorldMT50Adapter",
    "SMOLVLA_COMMUNITY_DATASETS",
    "SvlaSO100PickPlaceAdapter",
    "SvlaSO100SortingAdapter",
    "SvlaSO100StackingAdapter",
    "SvlaSO101PickPlaceAdapter",
    "T",
]
