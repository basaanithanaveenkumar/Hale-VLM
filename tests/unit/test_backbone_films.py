"""Unit tests for hale_vlm.visualization.backbone_films (animated VLM backbone explainers)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from hale_vlm.visualization.backbone_films import (  # noqa: E402
    BACKBONE_FILMS,
    BackboneFilmRenderer,
    DynamicResolutionFilm,
    MergeMappingProbe,
    SpaceToDepthFilm,
    TokenChip,
    TokenStreamFilm,
)
from hale_vlm.visualization.backbone_films.__main__ import main  # noqa: E402

EXPECTED_FILMS = {
    "qwen3_vl_pipeline",
    "qwen3_vl_dynamic_resolution",
    "qwen3_vl_patch_merge",
    "qwen3_vl_mrope",
    "qwen3_vl_deepstack",
    "qwen3_vl_token_stream",
    "smolvlm_pipeline",
    "smolvlm_image_tiling",
    "smolvlm_pixel_shuffle",
    "smolvlm_token_stream",
}


def test_catalog_registers_every_film():
    import hale_vlm.visualization.backbone_films.film_catalog  # noqa: F401

    assert EXPECTED_FILMS <= set(BACKBONE_FILMS)


def test_qwen3_vl_merge_groups_are_2x2_blocks():
    groups = MergeMappingProbe.qwen3_vl_merge(4, 6, merge=2)
    assert groups[0] == [0, 1, 6, 7]
    assert groups[1] == [2, 3, 8, 9]
    assert len(groups) == 6 and sorted(p for g in groups for p in g) == list(range(24))


def test_smolvlm_pixel_shuffle_groups_are_rxr_blocks_row_major():
    groups = MergeMappingProbe.smolvlm_pixel_shuffle(8, 4)
    assert groups[0] == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    assert groups[1][0] == 4 and groups[2][0] == 32


@pytest.mark.parametrize("hw", [(480, 640), (900, 300), (4000, 3000), (20, 30)])
def test_smart_resize_matches_transformers(hw):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    kwargs = dict(factor=32, min_pixels=64 * 32 * 32, max_pixels=1280 * 32 * 32)
    assert DynamicResolutionFilm.smart_resize(*hw, **kwargs) == smart_resize(*hw, **kwargs)


def test_space_to_depth_rejects_inconsistent_mapping():
    with pytest.raises(ValueError, match="exactly once"):
        SpaceToDepthFilm("t", (2, 2), (1, 1), [[0, 0, 1, 2]], "in", "out")
    with pytest.raises(ValueError, match="out_grid"):
        SpaceToDepthFilm("t", (2, 2), (2, 1), [[0, 1, 2, 3]], "in", "out")


def test_token_stream_requires_chips():
    with pytest.raises(ValueError):
        TokenStreamFilm("empty", [])


def test_render_writes_animated_gif_without_leaking_pyplot_figures(tmp_path):
    import matplotlib.pyplot as plt
    from PIL import Image

    before = len(plt.get_fignums())
    film = TokenStreamFilm("demo", [TokenChip("<s>", "special"), TokenChip("<image>", "token", 4)])
    path = film.render(tmp_path / "demo.gif")
    with Image.open(path) as gif:
        # 3 frames are drawn; the identical last two are merged by the GIF optimiser.
        assert gif.n_frames >= 2 and gif.is_animated
    assert len(plt.get_fignums()) == before


def test_renderer_writes_named_films(tmp_path):
    paths = BackboneFilmRenderer(tmp_path).render(["qwen3_vl_token_stream"])
    assert [p.name for p in paths] == ["qwen3_vl_token_stream.gif"] and paths[0].stat().st_size > 0


def test_cli_unknown_film_exits_2(tmp_path, capsys):
    assert main(["no_such_film", "--out", str(tmp_path)]) == 2
    assert "unknown" in capsys.readouterr().err
