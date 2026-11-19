"""Experiment 3: Evaluating Reactivity / Conditional Coverage of DtACI (Section 4.2.2).

Verifies DtACI actually learns alpha_t* rather than merely oscillating reactively.
Produces:
  - Figure 4.7: Conditional coverage plots (normalized scores)
  - Figure D.5: Conditional coverage plots (unnormalized scores)

Uses Block Bootstrap (Algorithm 4) on the error sequence with M=100, block size b=100
to compute 95% confidence intervals for conditional coverage.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from aci_repro.paths import CACHE
from aci_repro.plotting import use_style, savefig
from aci_repro.data_stocks import STOCK_FILES


def compute_conditional_coverage(
    alpha_bar: np.ndarray,
    err_bar: np.ndarray,
    bands: np.ndarray,
    bandwidth: float,
    min_count: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical conditional coverage per bin of alpha_bar_t."""
    midpoints = bands + bandwidth / 2.0
    cond_cov = np.full(len(bands), np.nan)
    for i, left in enumerate(bands):
        mask = (alpha_bar > left) & (alpha_bar < left + bandwidth)
        count = int(np.sum(mask))
        if count >= min_count:
            cond_cov[i] = 1.0 - np.mean(err_bar[mask])
    return midpoints, cond_cov


def block_bootstrap_error_bars(
    alpha_bar: np.ndarray,
    err_bar: np.ndarray,
    bands: np.ndarray,
    bandwidth: float,
    block_size: int = 100,
    n_bootstrap: int = 100,
    min_count: int = 25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Block bootstrap on (alpha_bar, err_bar) sequences for CI on conditional coverage.
    
    Resamples contiguous blocks of (alpha_bar_t, err_t) pairs to preserve
    temporal dependence, then computes conditional coverage per bin.
    
    Returns (lo, hi) arrays of shape (n_bands,).
    """
    rng = np.random.default_rng(seed)
    T = len(alpha_bar)
    n_full = (T // block_size) * block_size
    n_blocks = n_full // block_size
    
    all_freqs = np.full((len(bands), n_bootstrap), np.nan)
    
    for b in range(n_bootstrap):
        # Resample block start indices with replacement
        chosen_blocks = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([
            np.arange(cb * block_size, (cb + 1) * block_size) for cb in chosen_blocks
        ])
        boot_alpha = alpha_bar[idx]
        boot_err = err_bar[idx]
        
        for i, left in enumerate(bands):
            mask = (boot_alpha > left) & (boot_alpha < left + bandwidth)
            count = int(np.sum(mask))
            if count >= min_count:
                all_freqs[i, b] = 1.0 - np.mean(boot_err[mask])
    
    lo = np.nanquantile(all_freqs, 0.025, axis=1)
    hi = np.nanquantile(all_freqs, 0.975, axis=1)
    return lo, hi


def plot_conditional_coverage(
    stock_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    fig_name: str,
    alpha: float = 0.1,
):
    """Plot 2x2 conditional coverage with bootstrap error bars."""
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    stock_order = ["Nvidia", "AMD", "BlackBerry", "Fannie Mae"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for name, (r, c) in zip(stock_order, positions):
        ax = axes[r, c]
        midpoints, cond_cov, lo, hi = stock_results[name]

        valid = ~np.isnan(cond_cov)
        mp_v = midpoints[valid]
        cc_v = cond_cov[valid]
        lo_v = np.maximum(lo[valid], 0.5)
        hi_v = np.minimum(hi[valid], 1.0)

        ax.errorbar(mp_v, cc_v, yerr=[cc_v - lo_v, hi_v - cc_v],
                    fmt="o", color="red", markersize=4, capsize=2, linewidth=1.0)
        ax.axhline(y=1.0 - alpha, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylim(0.5, 1.15)

        ax.text(0.05, 0.95, name, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

        if r == 1:
            ax.set_xlabel(r"$\bar{\alpha}_t$")
        if c == 0:
            ax.set_ylabel("Empirical Conditional Coverage")

    plt.tight_layout()
    savefig(fig, fig_name)
    plt.close(fig)
    print(f"  Saved {fig_name}")


def main():
    stock_names = list(STOCK_FILES.keys())
    alpha = 0.1
    bandwidth = 0.02

    for score_key, fig_name in [
        ("norm", "fig_4_7_cond_coverage_normalized"),
        ("unnorm", "fig_d5_cond_coverage_unnormalized"),
    ]:
        print(f"\n=== {score_key.upper()} SCORES (Experiment 3) ===")
        
        # Collect all mean_alpha to compute global bins
        all_mean_alpha = []
        for name in stock_names:
            data = np.load(CACHE / f"{name.replace(' ', '_')}_exp2_{score_key}.npz")
            all_mean_alpha.append(data["DtACI_alphas"])

        global_min = min(a.min() for a in all_mean_alpha)
        global_max = max(a.max() for a in all_mean_alpha)
        bands = np.arange(global_min, global_max, bandwidth)
        print(f"  Bands: {len(bands)} bins, range [{global_min:.4f}, {global_max:.4f}]")

        stock_results = {}
        for name in stock_names:
            print(f"  [{name}] Computing conditional coverage + bootstrap...")
            data = np.load(CACHE / f"{name.replace(' ', '_')}_exp2_{score_key}.npz")
            mean_alpha = data["DtACI_alphas"]
            mean_err = data["DtACI_err"]

            midpoints, cond_cov = compute_conditional_coverage(
                mean_alpha, mean_err, bands, bandwidth
            )
            lo, hi = block_bootstrap_error_bars(
                mean_alpha, mean_err, bands, bandwidth,
                block_size=100, n_bootstrap=100,
            )
            stock_results[name] = (midpoints, cond_cov, lo, hi)

        print(f"  Plotting {fig_name}...")
        plot_conditional_coverage(stock_results, fig_name, alpha=alpha)

    print("\n✓ Experiment 3 complete!")


if __name__ == "__main__":
    main()
