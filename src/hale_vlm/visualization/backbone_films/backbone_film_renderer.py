"""Render registered backbone films to GIF files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from hale_vlm.visualization.backbone_films.film_registry import BACKBONE_FILMS


class BackboneFilmRenderer:
    """Builds films from the registry and writes ``<out_dir>/<name>.gif`` for each.

    Args:
        out_dir: destination directory.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)

    def render(self, names: Iterable[str] | None = None) -> list[Path]:
        """Render ``names`` (default: every registered film) and return the written paths.

        Args:
            names: registry keys; ``None`` renders all of them, in registration order.

        Raises:
            KeyError: for an unknown film name (lists the known ones).
        """
        import hale_vlm.visualization.backbone_films.film_catalog  # noqa: F401 - registers films

        selected = list(names) if names else list(BACKBONE_FILMS)
        paths = []
        for name in selected:
            film = BACKBONE_FILMS.get(name)()
            logger.info("rendering {}", name)
            paths.append(film.render(self.out_dir / f"{name}.gif"))
        return paths
