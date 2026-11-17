"""Sanity checks for aci_repro.

Tests must use synthetic inputs only (no files in data/ or cache/), so that
`uv run pytest` passes on a fresh clone.
"""

from aci_repro.paths import CACHE, DATA, FIGURES, ROOT


def test_paths_point_into_repo():
    assert (ROOT / "pyproject.toml").exists()
    for d in (DATA, CACHE, FIGURES):
        assert d.parent == ROOT and d.is_dir()
