"""``python -m hale_vlm.visualization.backbone_films [names...] --out DIR``: render the films."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns a process exit code."""
    from hale_vlm.visualization.backbone_films.backbone_film_renderer import BackboneFilmRenderer

    parser = argparse.ArgumentParser(prog="hale-vlm-films", description=__doc__)
    parser.add_argument("names", nargs="*", help="films to render (default: all)")
    parser.add_argument("--out", default="assets/backbone_films", help="output directory")
    args = parser.parse_args(argv)
    try:
        BackboneFilmRenderer(args.out).render(args.names or None)
    except KeyError as err:
        # A usage error is the CLI's user-facing output (like argparse's), not a log record.
        sys.stderr.write(f"{err}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
