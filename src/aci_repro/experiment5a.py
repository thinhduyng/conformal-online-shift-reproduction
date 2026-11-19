"""Experiment 5a: Ablation — Fixed vs. Variable eta_t in DtACI (Section 3.1 & Appendix D).

Compares fixed heuristic learning rate eta ≈ 2.72 against adaptive eta_t on stock data.
Produces:
  - Figure D.1: Local coverage comparison on UNNORMALIZED scores
  - Figure D.2: Local coverage comparison on NORMALIZED scores
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from aci_repro.paths import CACHE
from aci_repro.plotting import use_style, savefig
from aci_repro.conformal_methods import run_method
from aci_repro.data_stocks import STOCK_FILES


COLORS = {
    "Fixed η":    "#1f77b4",   # blue
    "Variable ηₜ": "#9467bd",  # purple
}


def local_coverage(err_seq: np.ndarray, window: int = 500) -> np.ndarray:
    kernel = np.ones(window) / window
    return 1.0 - np.convolve(err_seq, kernel, mode="same")


def main():
    stock_names = list(STOCK_FILES.keys())
    alpha = 0.1

    for score_key, fig_name, ylim in [
        ("unnorm", "fig_d1_ablation_eta_unnormalized", (0.80, 0.98)),
        ("norm", "fig_d2_ablation_eta_normalized", (0.84, 0.96)),
    ]:
        print(f"\n=== ABLATION eta ({score_key}) ===")
        use_style()
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        stock_order = ["Nvidia", "AMD", "BlackBerry", "Fannie Mae"]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for name, (r, c) in zip(stock_order, positions):
            ax = axes[r, c]

            # Load cached betas
            garch_data = np.load(CACHE / f"{name.replace(' ', '_')}_garch.npz", allow_pickle=True)
            betas = garch_data[f"betas_{score_key}"]
            dates = garch_data["beta_dates"]
            if not np.issubdtype(dates.dtype, np.datetime64):
                dates = dates.astype("datetime64[ns]")

            # Run DtACI with fixed eta
            print(f"  [{name}] Fixed eta...")
            res_fixed = run_method("dtaci", np.asarray(betas, dtype=float).tolist(), alpha, eta=2.72)
            lc_fixed = local_coverage(np.asarray(res_fixed["err"], dtype=float))

            # Run DtACI with variable eta_t
            print(f"  [{name}] Variable eta_t...")
            res_var = run_method("dtaci", np.asarray(betas, dtype=float).tolist(), alpha, eta=2.72, I=500, variable_eta=True)
            lc_var = local_coverage(np.asarray(res_var["err"], dtype=float))

            # Global coverage
            gc_fixed = 1.0 - np.mean(np.asarray(res_fixed["err"], dtype=float))
            gc_var = 1.0 - np.mean(np.asarray(res_var["err"], dtype=float))

            ax.plot(dates, lc_fixed, color=COLORS["Fixed η"], alpha=0.75, linewidth=1.2,
                    label=f"Fixed η (global={gc_fixed:.3f})")
            ax.plot(dates, lc_var, color=COLORS["Variable ηₜ"], alpha=0.75, linewidth=1.2,
                    label=f"Variable ηₜ (global={gc_var:.3f})")

            # Dashed global lines
            ax.axhline(y=gc_fixed, color=COLORS["Fixed η"], linestyle="--", linewidth=0.7, alpha=0.5)
            ax.axhline(y=gc_var, color=COLORS["Variable ηₜ"], linestyle="--", linewidth=0.7, alpha=0.5)
            ax.axhline(y=1.0 - alpha, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

            ax.set_ylim(*ylim)
            ax.text(0.05, 0.95, name, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            if r == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time")
            if c == 0:
                ax.set_ylabel("Coverage")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, frameon=False)
        fig.subplots_adjust(top=0.92)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        savefig(fig, fig_name)
        plt.close(fig)
        print(f"  Saved {fig_name}")

    print("\n✓ Experiment 5a complete!")


if __name__ == "__main__":
    main()
