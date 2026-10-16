"""Hale-VLM deployment: the VLM split into exportable parts for ``hale_core.deploy``.

.. code-block:: text

    HaleVLM ──▶ VisionEncoderAdapter  pixel_values → image_embeds        (TensorRT / NPU / MLX, per image)
            ──▶ LanguageModelAdapter  embeds|ids + mask → logits         (graph route)
                                      hf_checkpoint()                    (TensorRT-LLM / mlx-lm route)
            ──▶ VLMPrefillAdapter     pixel_values + ids + mask → logits (single graph, parity reference)

Importing this package registers the adapters in the ``hale_vlm`` scope; configs reference them
as ``type: hale_vlm.VisionEncoderAdapter`` after ``custom_imports: [hale_vlm.deploy]``.
"""

from hale_vlm.deploy.adapters import (
    LanguageModelAdapter,
    VisionEncoderAdapter,
    VLMPartAdapter,
    VLMPrefillAdapter,
)
from hale_vlm.deploy.registry import ADAPTERS, SCOPE

__all__ = [
    "ADAPTERS",
    "SCOPE",
    "LanguageModelAdapter",
    "VLMPartAdapter",
    "VLMPrefillAdapter",
    "VisionEncoderAdapter",
]
