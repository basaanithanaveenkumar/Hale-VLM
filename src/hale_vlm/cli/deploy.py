"""``hale-vlm-deploy``: the HaleBlocks deployment CLI with Hale-VLM's adapters registered."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Register the VLM adapters, then delegate to :class:`hale_core.deploy.cli.DeployCLI`."""
    import hale_vlm.deploy  # noqa: F401 - registers hale_vlm.* adapters
    from hale_core.deploy.cli import DeployCLI

    return DeployCLI(prog="hale-vlm-deploy").main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
