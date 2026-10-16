"""The films shipped with Hale-VLM, registered by name (each name is the GIF file stem)."""

from __future__ import annotations

from hale_vlm.visualization.backbone_films.deepstack_film import DeepStackFilm
from hale_vlm.visualization.backbone_films.dynamic_resolution_film import DynamicResolutionFilm
from hale_vlm.visualization.backbone_films.film_registry import register_film
from hale_vlm.visualization.backbone_films.image_tiling_film import ImageTilingFilm
from hale_vlm.visualization.backbone_films.merge_mapping_probe import MergeMappingProbe
from hale_vlm.visualization.backbone_films.mrope_position_film import MRoPEPositionFilm
from hale_vlm.visualization.backbone_films.pipeline_film import PipelineFilm
from hale_vlm.visualization.backbone_films.pipeline_stage import PipelineStage
from hale_vlm.visualization.backbone_films.space_to_depth_film import SpaceToDepthFilm
from hale_vlm.visualization.backbone_films.token_chip import TokenChip
from hale_vlm.visualization.backbone_films.token_stream_film import TokenStreamFilm


class FilmCatalog:
    """Factories for every shipped film; importing this module registers them.

    Numbers in the films follow the released checkpoints: Qwen3-VL's SigLIP-2 so400m encoder
    (patch 16, 2x2 merge, 27 blocks, DeepStack taps 8/16/24) and SmolVLM-256M/500M (512-px tiles,
    SigLIP base patch 16, pixel shuffle r = 4, 64 tokens per tile).
    """

    @staticmethod
    @register_film("qwen3_vl_pipeline")
    def qwen3_vl_pipeline() -> PipelineFilm:
        return PipelineFilm(
            "Qwen3-VL backbone",
            [
                PipelineStage(
                    "smart_resize",
                    "keep aspect ratio, sides multiple of 32 px",
                    "(3, 448, 640)",
                    "patch",
                ),
                PipelineStage(
                    "patchify", "16×16×2 patches in 2×2-block order", "(1120, 1536)", "patch"
                ),
                PipelineStage(
                    "ViT (SigLIP-2)",
                    "Conv3d embed + interpolated pos + 2-D RoPE, 27 blocks",
                    "(1120, 1152)",
                    "patch",
                ),
                PipelineStage(
                    "2×2 merger",
                    "concat 4 neighbours, LN, MLP to LLM width",
                    "(280, 4096)",
                    "token",
                ),
                PipelineStage(
                    "scatter", "replace <|image_pad|> embeddings", "(1, L, 4096)", "special"
                ),
                PipelineStage(
                    "Interleaved M-RoPE", "3-D (t, h, w) positions", "cos/sin (L, 128)", "position"
                ),
                PipelineStage(
                    "Qwen3 LLM",
                    "GQA + QK-norm + SwiGLU; DeepStack adds to layers 0-2",
                    "(1, L, 4096)",
                    "text",
                ),
            ],
        )

    @staticmethod
    @register_film("qwen3_vl_dynamic_resolution")
    def qwen3_vl_dynamic_resolution() -> DynamicResolutionFilm:
        return DynamicResolutionFilm()

    @staticmethod
    @register_film("qwen3_vl_patch_merge")
    def qwen3_vl_patch_merge() -> SpaceToDepthFilm:
        mapping = MergeMappingProbe.qwen3_vl_merge(6, 8, merge=2)
        return SpaceToDepthFilm(
            "Qwen3-VL: 2×2 patch merge",
            (6, 8),
            (3, 4),
            mapping,
            "ViT patches: 48 × 1152",
            "merged tokens: 12 × 4608 → MLP → LLM width",
            animate_tokens=3,
        )

    @staticmethod
    @register_film("qwen3_vl_mrope")
    def qwen3_vl_mrope() -> MRoPEPositionFilm:
        return MRoPEPositionFilm()

    @staticmethod
    @register_film("qwen3_vl_deepstack")
    def qwen3_vl_deepstack() -> DeepStackFilm:
        return DeepStackFilm()

    @staticmethod
    @register_film("qwen3_vl_token_stream")
    def qwen3_vl_token_stream() -> TokenStreamFilm:
        chips = [
            TokenChip("<|im_start|>", "special"),
            TokenChip("user"),
            TokenChip("\\n", newline=True),
            TokenChip("<|vision_start|>", "special"),
            TokenChip("<|image_pad|>", "token", 280),
            TokenChip("<|vision_end|>", "special", newline=True),
            TokenChip("Describe"),
            TokenChip("the"),
            TokenChip("scene"),
            TokenChip("."),
            TokenChip("<|im_end|>", "special", newline=True),
        ]
        notes = {
            4: "448×640 image → 14×20 merged grid → 280 placeholders, one per merged token",
            5: "the placeholders are later overwritten by the merger output (masked_scatter)",
        }
        return TokenStreamFilm("Qwen3-VL: what the LLM sees", chips, notes)

    @staticmethod
    @register_film("smolvlm_pipeline")
    def smolvlm_pipeline() -> PipelineFilm:
        return PipelineFilm(
            "SmolVLM backbone",
            [
                PipelineStage(
                    "resize + split",
                    "tiles of 512 px + 1 global view",
                    "(13, 3, 512, 512)",
                    "patch",
                ),
                PipelineStage(
                    "SigLIP encoder",
                    "patch 16, NaViT-style positions, 12 blocks",
                    "(13, 1024, 768)",
                    "patch",
                ),
                PipelineStage(
                    "pixel shuffle", "r = 4: 16 neighbours → 1 token", "(13, 64, 12288)", "token"
                ),
                PipelineStage(
                    "projection", "one Linear (no bias) to LLM width", "(13, 64, 576)", "token"
                ),
                PipelineStage(
                    "merge into prompt", "fill <image> slots tile by tile", "(1, L, 576)", "special"
                ),
                PipelineStage("SmolLM2 decoder", "Llama-style, 1-D RoPE", "(1, L, 576)", "text"),
            ],
            per_row=3,
        )

    @staticmethod
    @register_film("smolvlm_image_tiling")
    def smolvlm_image_tiling() -> ImageTilingFilm:
        return ImageTilingFilm()

    @staticmethod
    @register_film("smolvlm_pixel_shuffle")
    def smolvlm_pixel_shuffle() -> SpaceToDepthFilm:
        mapping = MergeMappingProbe.smolvlm_pixel_shuffle(8, 4)
        return SpaceToDepthFilm(
            "SmolVLM: pixel shuffle (r = 4)",
            (8, 8),
            (2, 2),
            mapping,
            "SigLIP patches (demo 8×8; real tile 32×32)",
            "tokens: 16× fewer, 16× wider",
            animate_tokens=2,
        )

    @staticmethod
    @register_film("smolvlm_token_stream")
    def smolvlm_token_stream() -> TokenStreamFilm:
        chips = [
            TokenChip("<|im_start|>", "special"),
            TokenChip("User"),
            TokenChip(":", newline=True),
        ]
        for row in range(1, 3):
            for col in range(1, 4):
                chips += [
                    TokenChip("<fake_token_around_image>", "special"),
                    TokenChip(f"<row_{row}_col_{col}>", "position"),
                    TokenChip("<image>", "token", 64),
                ]
            chips[-1] = TokenChip("<image>", "token", 64, newline=True)
        chips += [
            TokenChip("<fake_token_around_image>", "special"),
            TokenChip("<global-img>", "position"),
            TokenChip("<image>", "token", 64),
            TokenChip("<fake_token_around_image>", "special", newline=True),
            TokenChip("Describe"),
            TokenChip("it"),
            TokenChip("."),
            TokenChip("<end_of_utterance>", "special"),
        ]
        notes = {
            4: "each tile: a position tag, then 64 <image> slots (one per shuffled token)",
            22: "the global view comes last, with its own tag",
        }
        return TokenStreamFilm("SmolVLM: what the LLM sees (2×3 tiles)", chips, notes)
