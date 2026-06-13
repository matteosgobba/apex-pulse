"""Project-relative path utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT_ENV = "F1_PREDICTION_PROJECT_ROOT"


def get_project_root(start: Path | None = None) -> Path:
    """Find the nearest project root containing ``pyproject.toml``."""
    configured_root = os.getenv(PROJECT_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError(f"{PROJECT_ROOT_ENV} does not point to a project root: {root}")
        return root

    search_starts = [Path(start).resolve() if start else Path.cwd().resolve()]
    package_root_candidate = Path(__file__).resolve().parents[3]
    if package_root_candidate not in search_starts:
        search_starts.append(package_root_candidate)

    for search_start in search_starts:
        for candidate in (search_start, *search_start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate

    raise FileNotFoundError("Could not locate a project root containing pyproject.toml")


def resolve_project_path(path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve a path against the project root without requiring it to exist."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = (project_root or get_project_root()).resolve()
    return (root / candidate).resolve()


def ensure_directory(path: Path) -> Path:
    """Create a directory and return its resolved path."""
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def slugify(value: str) -> str:
    """Convert a user-facing identifier to a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Cannot create a path from an empty identifier")
    return slug
