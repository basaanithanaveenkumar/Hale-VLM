"""Animated explainers ("films") for VLM backbones: Qwen3-VL and SmolVLM.

Every film is a :class:`BackboneFilm` that yields frames; :class:`GifWriter` turns them into a
compact GIF. Films used by the notebooks in ``notebooks/backbones/vlm/`` are registered in
:data:`BACKBONE_FILMS` by :mod:`film_catalog` and rendered with
``python -m hale_vlm.visualization.backbone_films``.

Imports are lazy (PEP 562): ``import hale_vlm.visualization.backbone_films`` does not pull in
matplotlib, torch or transformers until a film is actually built.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BACKBONE_FILMS": "film_registry",
    "BackboneFilm": "backbone_film",
    "BackboneFilmRenderer": "backbone_film_renderer",
    "DeepStackFilm": "deepstack_film",
    "DynamicResolutionFilm": "dynamic_resolution_film",
    "FilmCanvas": "film_canvas",
    "FilmCatalog": "film_catalog",
    "FilmPalette": "film_palette",
    "GifWriter": "gif_writer",
    "ImageTilingFilm": "image_tiling_film",
    "MergeMappingProbe": "merge_mapping_probe",
    "MRoPEPositionFilm": "mrope_position_film",
    "PipelineFilm": "pipeline_film",
    "PipelineStage": "pipeline_stage",
    "SpaceToDepthFilm": "space_to_depth_film",
    "SyntheticScene": "synthetic_scene",
    "TokenChip": "token_chip",
    "TokenStreamFilm": "token_stream_film",
    "register_film": "film_registry",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
