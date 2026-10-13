#!/usr/bin/env python3
"""Profile Hale-VLM components: vision projector, GPT backbone, and assembled VLM stub.

Because the full VLM requires downloading HuggingFace weights (Qwen3-8B / SigLIP),
this script profiles architecture-equivalent stub modules that replicate the exact
layer structure with randomly-initialised weights. Pass --real to load actual weights.

Usage:
    python scripts/profile_model.py                          # stub VLM, CPU
    python scripts/profile_model.py --component projector    # vision projector only
    python scripts/profile_model.py --component backbone     # LLM backbone only
    python scripts/profile_model.py --component vlm          # full VLM stub (default)
    python scripts/profile_model.py --device cuda --torch-profiler --trace out/trace.json
    python scripts/profile_model.py --sort flops --top 30 --json out/profile.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "HaleBlocks" / "src"))

import torch
import torch.nn as nn

from hale_core.profiling import ModelProfiler
from hale_core.nn import GPTBackbone


# ---------------------------------------------------------------------------
# Stub modules that match the real Hale-VLM architecture
# ---------------------------------------------------------------------------

class StubVisionEncoder(nn.Module):
    """Architecture-equivalent stub for SigLIP ViT-B/16 (400M image size, 196 patches)."""

    def __init__(self, patch_size: int = 14, num_patches: int = 196, hidden: int = 768):
        super().__init__()
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, hidden)
        self.pos_embed   = nn.Embedding(num_patches + 1, hidden)
        self.layers      = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=12, dim_feedforward=hidden * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.hidden_size = hidden
        self._num_patches = num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b = pixel_values.shape[0]
        # Flatten patches: (B, C, H, W) → (B, num_patches, patch_dim)
        p = pixel_values.reshape(b, self._num_patches, -1)
        x = self.patch_embed(p)
        ids = torch.arange(self._num_patches, device=x.device).unsqueeze(0).expand(b, -1)
        x = x + self.pos_embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class StubVisionProjector(nn.Module):
    """Two-layer MLP projector: vision_dim → hidden → llm_dim."""

    def __init__(self, vision_dim: int = 768, llm_dim: int = 4096):
        super().__init__()
        hidden = llm_dim * 2
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StubVLM(nn.Module):
    """Vision encoder + projector + GPT-style LLM backbone (Qwen3-8B-equivalent depth)."""

    def __init__(
        self,
        vocab_size: int = 152064,
        d_model: int = 4096,
        n_heads: int = 32,
        n_layers: int = 4,        # use 4 for fast profiling; real model = 32
        d_ff: int = 11008,
        seq_len: int = 128,
        num_patches: int = 196,
    ):
        super().__init__()
        self.vision  = StubVisionEncoder(num_patches=num_patches, hidden=1152)
        self.proj    = StubVisionProjector(vision_dim=1152, llm_dim=d_model)
        self.backbone = GPTBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_length=seq_len + num_patches,
            tie_embeddings=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        vis = self.vision(pixel_values)          # (B, num_patches, 1152)
        vis = self.proj(vis)                     # (B, num_patches, d_model)
        tok = self.backbone.embedding(input_ids) # (B, seq, d_model)
        combined = torch.cat([vis, tok], dim=1)[:, :input_ids.shape[1], :]
        return self.backbone.stack(combined)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_projector(args):
    model = StubVisionProjector(vision_dim=1152, llm_dim=args.d_model)
    dummy = torch.randn(args.batch_size, 196, 1152)
    return model, dummy


def build_backbone(args):
    model = GPTBackbone(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_length=args.seq_len,
        tie_embeddings=False,
    )
    dummy = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    return model, dummy


def build_vlm(args):
    model = StubVLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        seq_len=args.seq_len,
        num_patches=196,
    )
    input_ids    = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    pixel_values = torch.randn(args.batch_size, 3, 224, 224)
    return model, (input_ids, pixel_values)


_BUILDERS = {
    "projector": build_projector,
    "backbone":  build_backbone,
    "vlm":       build_vlm,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hale-VLM model profiler")
    p.add_argument("--component",     default="vlm",  choices=list(_BUILDERS))
    p.add_argument("--device",        default="cpu")
    p.add_argument("--batch-size",    type=int, default=1)
    p.add_argument("--seq-len",       type=int, default=128)
    p.add_argument("--vocab-size",    type=int, default=8192)
    p.add_argument("--d-model",       type=int, default=256,  help="use 4096 for Qwen3-8B scale")
    p.add_argument("--n-heads",       type=int, default=4,    help="use 32 for Qwen3-8B scale")
    p.add_argument("--n-layers",      type=int, default=4,    help="use 32 for Qwen3-8B scale")
    p.add_argument("--d-ff",          type=int, default=1024, help="use 11008 for Qwen3-8B scale")
    p.add_argument("--top",           type=int, default=40)
    p.add_argument("--sort",          default="latency", choices=["latency", "flops", "matmuls"])
    p.add_argument("--warmup",        type=int, default=1)
    p.add_argument("--torch-profiler", action="store_true")
    p.add_argument("--trace",         default=None)
    p.add_argument("--json",          default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"\n[profile] Hale-VLM  component={args.component}  device={device}  "
          f"batch={args.batch_size}  seq={args.seq_len}  "
          f"d_model={args.d_model}  n_layers={args.n_layers}")

    model, dummy = _BUILDERS[args.component](args)
    model = model.to(device).eval()

    if isinstance(dummy, tuple):
        dummy = tuple(t.to(device) for t in dummy)
    else:
        dummy = dummy.to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(*dummy) if isinstance(dummy, tuple) else model(dummy)

    profiler = ModelProfiler(
        model,
        use_cuda_events=(device.type == "cuda"),
        use_torch_profiler=args.torch_profiler,
        torch_profiler_trace=args.trace,
    )

    with torch.no_grad(), profiler.profile():
        model(*dummy) if isinstance(dummy, tuple) else model(dummy)

    report = profiler.report()
    print("\n" + report.summary_table(top_k=args.top, sort_by=args.sort))

    if args.torch_profiler:
        print("\n── torch.profiler key averages ──")
        print(report.torch_key_averages())

    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[profile] JSON → {args.json}")

    if args.trace:
        print(f"[profile] Chrome trace → {args.trace}")


if __name__ == "__main__":
    main()
