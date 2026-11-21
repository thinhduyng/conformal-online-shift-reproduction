"""Experiment 2: Online Prediction in the Stock Market (Section 4.2.1 & Appendix D).

Runs DtACI, AgACI, MVP, and Constant Baseline on stock volatility data,
then generates:
  - Figure 4.4: Local coverage with NORMALIZED scores (4-panel)
  - Figure 4.5: Local coverage with UNNORMALIZED scores (4-panel)
  - Figure 4.6: Bernoulli Q-Q plot for normalized DtACI errors
  - Figure D.4: Bernoulli Q-Q plot for unnormalized DtACI errors
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple

from aci_repro.paths import CACHE, FIGURES
from aci_repro.plotting import use_style, savefig, METHOD_COLORS
from aci_repro.conformal_methods import run_method
from aci_repro.data_stocks import STOCK_FILES


# ── Colour palette matching the paper ──────────────────────────────────────
COLORS = {
    "DtACI":    "#1f77b4",   # blue
    "AgACI":    "#d62728",   # red
    "MVP":      "#2ca02c",   # green
    "Constant": "#7f7f7f",   # grey
}
METHOD_ORDER = ["DtACI", "Constant", "MVP", "AgACI"]


# ── Helper: local coverage (moving average) ────────────────────────────────
def local_coverage(err_seq: np.ndarray, window: int = 500) -> np.ndarray:
    """Compute moving-average local coverage: 1 - (1/w) sum_{s} err_s."""
    kernel = np.ones(window) / window
    return 1.0 - np.convolve(err_seq, kernel, mode="same")


# ── Load cached GARCH data ─────────────────────────────────────────────────
def load_cached(stock_name: str) -> Dict[str, np.ndarray]:
    cache_file = CACHE / f"{stock_name.replace(' ', '_')}_garch.npz"
    data = np.load(cache_file, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ── Run all methods on one stock ───────────────────────────────────────────
def run_all_methods(betas: np.ndarray, label: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Run DtACI, AgACI, MVP, Constant Baseline on a beta sequence."""
    alpha = 0.1
    b = np.asarray(betas, dtype=float).tolist()
    out = {}
    for label_m, name in [("DtACI", "dtaci"), ("AgACI", "agaci"), ("MVP", "mvp"), ("Constant", "constant")]:
        print(f"  [{label}] Running {label_m}...")
        res = run_method(name, b, alpha)
        out[label_m] = {"alphas": np.asarray(res["alphas"]), "err": np.asarray(res["err"], dtype=float)}
    return out


# ── Figure 4.4 / 4.5: Local coverage plots (2×2) ──────────────────────────
def plot_local_coverage(
    all_results: Dict[str, Dict[str, Dict]],
    all_dates: Dict[str, np.ndarray],
    score_type: str,
    fig_name: str,
    ylim: Tuple[float, float] = (0.37, 1.0),
    window: int = 500,
    alpha: float = 0.1,
):
    """Plot 2×2 local coverage grid for 4 stocks."""
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    stock_order = ["Nvidia", "AMD", "BlackBerry", "Fannie Mae"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (name, (r, c)) in zip(stock_order, positions):
        ax = axes[r, c]
        results = all_results[name]
        dates = all_dates[name]

        for method_name in METHOD_ORDER:
            res = results[method_name]
            errs = res["err"]
            lc = local_coverage(errs, window=window)
            color = COLORS.get(method_name, "black")
            label = method_name if method_name != "Constant" else "Constant Baseline"
            ax.plot(dates, lc, color=color, alpha=0.75, linewidth=1.2, label=label)

            # Global coverage dashed line
            global_cov = 1.0 - np.mean(errs)
            ax.axhline(y=global_cov, color=color, linestyle="--", linewidth=0.8, alpha=0.5)

        # Target line
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

    # Shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10, frameon=False)
    fig.subplots_adjust(top=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, fig_name)
    plt.close(fig)
    print(f"  Saved {fig_name}")


# ── Figure 4.6 / D.4: Bernoulli Q-Q plots ─────────────────────────────────
def plot_qq(
    all_results: Dict[str, Dict[str, Dict]],
    fig_name: str,
    alpha: float = 0.1,
    window: int = 500,
    n_bernoulli: int = 100000,
    seed: int = 42,
):
    """Plot 2×2 Q-Q plot grid for DtACI coverage error gaps vs Bernoulli."""
    use_style()
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0.05, 0.95, 19)

    # Generate reference Bernoulli error gaps
    bern = rng.binomial(1, alpha, size=n_bernoulli).astype(float)
    kernel = np.ones(window) / window
    bern_smooth = np.convolve(bern, kernel, mode="same")
    bern_gaps = np.abs(bern_smooth - alpha)[window:]  # discard burn-in

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    stock_order = ["Nvidia", "AMD", "BlackBerry", "Fannie Mae"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (name, (r, c)) in zip(stock_order, positions):
        ax = axes[r, c]
        res = all_results[name]["DtACI"]
        errs = res["err"]
        err_smooth = np.convolve(errs, kernel, mode="same")
        dtaci_gaps = np.abs(err_smooth - alpha)[window:]  # discard burn-in

        q_dtaci = np.quantile(dtaci_gaps, quantiles)
        q_bern = np.quantile(bern_gaps, quantiles)

        ax.scatter(q_dtaci, q_bern, s=30, zorder=3, color=COLORS["DtACI"])
        lim = max(q_dtaci.max(), q_bern.max()) * 1.15
        lim = max(lim, 0.03)
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.0, alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.text(0.05, 0.95, name, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

        if r == 1:
            ax.set_xlabel("Quantiles of DtACI Error Gaps")
        if c == 0:
            ax.set_ylabel("Quantiles of Bern. Error Gaps")

    plt.tight_layout()
    savefig(fig, fig_name)
    plt.close(fig)
    print(f"  Saved {fig_name}")


# ── Main entry point ───────────────────────────────────────────────────────
def main():
    stock_names = list(STOCK_FILES.keys())
    
    # Load cached data
    print("Loading cached GARCH data...")
    cached = {name: load_cached(name) for name in stock_names}
    
    # ── Normalized scores (Figures 4.4, 4.6) ──
    print("\n=== NORMALIZED SCORES ===")
    norm_results = {}
    norm_dates = {}
    for name in stock_names:
        d = cached[name]
        betas = d["betas_norm"]
        # Convert dates
        dates = d["beta_dates"]
        if not np.issubdtype(dates.dtype, np.datetime64):
            dates = dates.astype("datetime64[ns]")
        norm_dates[name] = dates
        norm_results[name] = run_all_methods(betas, f"{name}/norm")
    
    print("\nPlotting Figure 4.4 (normalized local coverage)...")
    plot_local_coverage(norm_results, norm_dates, "normalized",
                        "fig_4_4_coverage_normalized", ylim=(0.80, 0.965))
    
    print("Plotting Figure 4.6 (normalized Q-Q)...")
    plot_qq(norm_results, "fig_4_6_qq_normalized")
    
    # ── Unnormalized scores (Figures 4.5, D.4) ──
    print("\n=== UNNORMALIZED SCORES ===")
    unnorm_results = {}
    unnorm_dates = {}
    for name in stock_names:
        d = cached[name]
        betas = d["betas_unnorm"]
        dates = d["beta_dates"]
        if not np.issubdtype(dates.dtype, np.datetime64):
            dates = dates.astype("datetime64[ns]")
        unnorm_dates[name] = dates
        unnorm_results[name] = run_all_methods(betas, f"{name}/unnorm")
    
    print("\nPlotting Figure 4.5 (unnormalized local coverage)...")
    plot_local_coverage(unnorm_results, unnorm_dates, "unnormalized",
                        "fig_4_5_coverage_unnormalized", ylim=(0.37, 1.0))
    
    print("Plotting Figure D.4 (unnormalized Q-Q)...")
    plot_qq(unnorm_results, "fig_d4_qq_unnormalized")

    # Save method results for Experiment 3 & 5 reuse
    print("\nCaching method results for Exp 3 & 5...")
    for score_type, results_dict in [("norm", norm_results), ("unnorm", unnorm_results)]:
        for name in stock_names:
            out = {}
            for method, res in results_dict[name].items():
                for key, val in res.items():
                    out[f"{method}_{key}"] = val
            cache_file = CACHE / f"{name.replace(' ', '_')}_exp2_{score_type}.npz"
            np.savez_compressed(cache_file, **out)
    
    print("\n✓ Experiment 2 complete! All figures saved to figures/")


if __name__ == "__main__":
    main()
