"""Measure which patches each merged token is built from, by running the real library code."""

from __future__ import annotations


class MergeMappingProbe:
    """Pushes an *index image* through the library's own reshape code and reads the result back.

    Every patch is filled with its own raster index, so after the operation each output token's
    channels spell out which patches it was concatenated from, in channel order. This answers
    "which patches end up in which token?" from the code itself instead of from a diagram, and the
    result drives :class:`SpaceToDepthFilm`.
    """

    @staticmethod
    def qwen3_vl_merge(
        grid_h: int, grid_w: int, merge: int = 2, patch: int = 16, temporal_patch: int = 2
    ) -> list[list[int]]:
        """Patch groups of Qwen3-VL's ``merge x merge`` patch merger.

        The image processor's ``patchify`` reorders patches block-major, so the merger's plain
        ``view(-1, hidden * merge**2)`` concatenates each spatial ``merge x merge`` block.

        Args:
            grid_h: patch rows (image height / ``patch``).
            grid_w: patch columns.
            merge: spatial merge size (2 in Qwen2/2.5/3-VL).
            patch: pixel side of a patch.
            temporal_patch: frames per temporal patch (an image is repeated to fill it).

        Returns:
            ``(grid_h // merge) * (grid_w // merge)`` lists of ``merge**2`` raster patch indices.
        """
        import torch
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

        index = torch.arange(grid_h * grid_w, dtype=torch.float32).view(1, 1, grid_h, grid_w)
        pixels = index.repeat_interleave(patch, 2).repeat_interleave(patch, 3).repeat(1, 3, 1, 1)
        # ``patchify`` uses no instance state, so it can be called unbound on the class.
        flat, _, _ = Qwen2VLImageProcessor.patchify(None, pixels, patch, merge, temporal_patch)
        order = flat[0, :, 0].long()  # raster index of each sequence position
        return order.view(-1, merge * merge).tolist()  # merger: consecutive merge**2 positions

    @staticmethod
    def smolvlm_pixel_shuffle(grid: int, scale: int) -> list[list[int]]:
        """Patch groups of SmolVLM's ``pixel_shuffle`` (space-to-depth by ``scale``).

        Args:
            grid: patches per side of the (square) tile, e.g. 32 for a 512-px tile with patch 16.
            scale: shuffle factor ``r``; ``r**2`` patches become one token.

        Returns:
            ``(grid // scale) ** 2`` lists of ``scale**2`` raster patch indices.
        """
        import torch
        from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector

        index = torch.arange(grid * grid, dtype=torch.float32).view(1, grid * grid, 1)
        # ``pixel_shuffle`` reads nothing from ``self``; the hidden size is 1, so every output
        # channel holds exactly one source patch index.
        shuffled = SmolVLMConnector.pixel_shuffle(None, index, scale)
        return shuffled[0].long().tolist()
