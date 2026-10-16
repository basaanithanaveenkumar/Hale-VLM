"""Shared helpers and constants of :mod:`sequential` (split from the original module)."""

from __future__ import annotations


def all_builtin_dataset_names() -> list[str]:
    from hale_vlm.data.catalog import SMOLVLM_ALL_DATASETS

    return list(SMOLVLM_ALL_DATASETS)
