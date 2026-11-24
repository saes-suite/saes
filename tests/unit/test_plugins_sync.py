"""Cross-repository drift checks between `saes` and `saes-plugins` skills."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import saes
import saes.domain.exceptions as saes_exceptions

SKILLS_DIR = Path(__file__).resolve().parents[3] / "saes-plugins" / "plugins" / "saes" / "skills"


def _require_skills_dir() -> Path:
    if not SKILLS_DIR.is_dir():
        pytest.skip("saes-plugins sibling repository not present")
    return SKILLS_DIR


def _skill_files() -> list[Path]:
    skills_dir = _require_skills_dir()
    files = sorted(skills_dir.glob("*/SKILL.md"))
    assert files, f"No SKILL.md files found under {skills_dir}"
    return files


def test_documented_top_level_and_statistics_symbols_exist() -> None:
    """Verify every `saes.<symbol>` and `saes.statistics.<symbol>` in skills exists."""
    top_level_refs: set[str] = set()
    stats_refs: set[str] = set()

    for skill_file in _skill_files():
        text = skill_file.read_text(encoding="utf-8")
        top_level_refs.update(re.findall(r"\bsaes\.([a-zA-Z_][a-zA-Z0-9_]*)", text))
        stats_refs.update(re.findall(r"\bsaes\.statistics\.([a-zA-Z_][a-zA-Z0-9_]*)", text))

    assert top_level_refs, "Expected at least one saes.<symbol> reference in SKILL.md files"
    assert stats_refs, "Expected at least one saes.statistics.<symbol> reference in SKILL.md files"

    missing_top = sorted(sym for sym in top_level_refs if not hasattr(saes, sym))
    assert not missing_top, f"Skills reference missing saes symbols: {missing_top}"

    missing_stats = sorted(sym for sym in stats_refs if not hasattr(saes.statistics, sym))
    assert not missing_stats, f"Skills reference missing saes.statistics symbols: {missing_stats}"


def test_documented_exceptions_exist() -> None:
    """Verify every exception class documented in `saes-python-api/SKILL.md` exists."""
    api_skill = _require_skills_dir() / "saes-python-api" / "SKILL.md"
    text = api_skill.read_text(encoding="utf-8")

    match = re.search(r"## Exceptions.*?(?=\n## |\Z)", text, flags=re.DOTALL)
    assert match, "Exceptions section not found in saes-python-api/SKILL.md"

    documented_errors = set(re.findall(r"\b([A-Z][a-zA-Z0-9]*Error)\b", match.group(0)))
    assert documented_errors, "No exception classes found in Exceptions section"

    missing = sorted(err for err in documented_errors if not hasattr(saes_exceptions, err))
    assert not missing, f"Skills reference missing exceptions in saes.domain.exceptions: {missing}"


def test_minimum_saes_version_matches_package() -> None:
    """Verify `saes.__version__` satisfies the floor declared in `saes-python-api/SKILL.md`."""
    import subprocess

    api_skill = _require_skills_dir() / "saes-python-api" / "SKILL.md"
    text = api_skill.read_text(encoding="utf-8")

    match = re.search(r"Target library:\s*`saes\s*>=\s*(\d+\.\d+\.\d+)`", text)
    assert match, "Target library floor (`saes >= X.Y.Z`) not found in saes-python-api/SKILL.md"

    floor_tuple = tuple(int(part) for part in match.group(1).split("."))
    current_tuple = tuple(int(part) for part in saes.__version__.split(".")[:3])

    if current_tuple < floor_tuple:
        # On a `major/*` release branch, release.yml stamps (major+1).0.0 on PR merge.
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            check=False,
        ).stdout.strip()
        next_major = (current_tuple[0] + 1, 0, 0)
        if branch.startswith(("major/", "major-")) and next_major >= floor_tuple:
            return

    assert current_tuple >= floor_tuple, (
        f"saes {saes.__version__} is older than plugin minimum {match.group(1)}"
    )
