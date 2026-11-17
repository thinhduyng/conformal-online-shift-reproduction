"""Project directories, resolved from the repo root.

Use these instead of relative paths like "../data" so code works the same
from notebooks/, tests/, or the repo root.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"  # raw downloads (gitignored)
CACHE = ROOT / "cache"  # expensive intermediate results, e.g. GARCH betas (gitignored)
FIGURES = ROOT / "figures"  # final figures used by the report and slides (committed)

for _d in (DATA, CACHE, FIGURES):
    _d.mkdir(exist_ok=True)
