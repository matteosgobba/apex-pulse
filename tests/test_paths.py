from pathlib import Path

import pytest

from f1_prediction.utils.paths import (
    ensure_directory,
    get_project_root,
    resolve_project_path,
    slugify,
)


def test_get_project_root_searches_parent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("F1_PREDICTION_PROJECT_ROOT", raising=False)
    (tmp_path / "pyproject.toml").touch()
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)

    assert get_project_root(nested) == tmp_path


def test_resolve_project_path_uses_project_root(tmp_path: Path) -> None:
    assert resolve_project_path("data/raw", tmp_path) == (tmp_path / "data/raw").resolve()


def test_ensure_directory_creates_nested_directory(tmp_path: Path) -> None:
    directory = ensure_directory(tmp_path / "one" / "two")

    assert directory.is_dir()
    assert directory == (tmp_path / "one" / "two").resolve()


def test_slugify_builds_safe_path_component() -> None:
    assert slugify(" Emilia-Romagna GP ") == "emilia-romagna-gp"
