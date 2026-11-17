"""Shared plot style so figures from all three notebooks look consistent.

Usage in a notebook:

    from aci_repro.plotting import use_style, METHOD_COLORS, savefig
    use_style()
    ax.plot(t, cov, color=METHOD_COLORS["aci"], label="ACI")
    savefig(fig, "exp1_coverage")
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from aci_repro.paths import FIGURES

# One fixed color per method, used in every figure. Add methods here, not in notebooks.
METHOD_COLORS = {
    "aci": "#1f77b4",
    "standard": "#d62728",  # non-adaptive split conformal (gamma = 0)
    "target": "#555555",  # target coverage line, 1 - alpha
}


def use_style() -> None:
    """Apply the shared matplotlib style. Call once at the top of each notebook."""
    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 4.0),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def savefig(fig: Figure, name: str, formats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Save `fig` to figures/<name>.<fmt>. PDF for the report, PNG for slides."""
    for fmt in formats:
        fig.savefig(FIGURES / f"{name}.{fmt}")
