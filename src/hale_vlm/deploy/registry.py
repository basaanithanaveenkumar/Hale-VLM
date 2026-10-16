"""Hale-VLM scope of the deployment registries (mmdet-style child registry).

Adapters registered here are addressed from configs as ``hale_vlm.<Name>``; the child falls
back to HaleBlocks' root ``ADAPTERS`` for generic adapters (``TorchModuleAdapter`` ...).
"""

from __future__ import annotations

from hale_core.deploy.registry import ADAPTERS as CORE_ADAPTERS
from hale_core.deploy.registry import Registry

#: Scope name used in config ``type`` keys: ``type: hale_vlm.VisionEncoderAdapter``.
SCOPE = "hale_vlm"

ADAPTERS = Registry("adapter", parent=CORE_ADAPTERS, scope=SCOPE)

__all__ = ["ADAPTERS", "SCOPE"]
