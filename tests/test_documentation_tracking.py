from pathlib import Path


def test_project_documentation_files_are_not_ignored() -> None:
    ignored_patterns = {
        line.strip()
        for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "AGENTS.md" not in ignored_patterns
    assert "README.md" not in ignored_patterns
    assert "PROJECT_HANDOFF.md" not in ignored_patterns
