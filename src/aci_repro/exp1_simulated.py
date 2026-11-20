"""Section 4.1 simulated examples: Figures 4.1, 4.2 and 4.3."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from aci_repro.conformal_methods import agaci, dtaci, mvp
from aci_repro.paths import CACHE
from aci_repro.plotting import METHOD_COLORS, savefig, use_style

ALPHA: float = 0.1
T_DEFAULT: int = 6000
TRIALS_DEFAULT: int = 100
SMOOTH_SD: float = 0.006
JUMP_SMALL: float = 0.075
JUMP_LARGE: float = 1.5
JUMP_PERIOD: int = 500
WINDOW: int = 50

REGIMES: list[str] = ["stationary", "smooth", "jump"]
METHODS: list[str] = ["dtaci", "agaci", "mvp"]
LABELS: dict[str, str] = {
        "stationary": "Stationary",
        "smooth": "Smooth",
        "jump": "Jump",
        "dtaci": "DtACI",
        "agaci": "AgACI",
        "mvp": "MVP",
        }

Array = NDArray[np.float64]

# -----------------------------------------------------------------------------
# mu_t trajectories
# -----------------------------------------------------------------------------


def mu_stationary(T: int) -> Array:
    return np.zeros(T)


def mu_smooth(T: int, rng: np.random.Generator, sd: float = SMOOTH_SD) -> Array:
    eps = rng.normal(0.0, sd, size=T)
    mu = np.zeros(T)
    prev = 0.0
    for t in range(T - 1):
        mu[t + 1] = mu[t] + 0.5 * (mu[t] - prev) + 0.5 * eps[t]
        prev = mu[t]
    return mu


def mu_jump(
        T: int,
        small: float = JUMP_SMALL,
        large: float = JUMP_LARGE,
        period: int = JUMP_PERIOD,
        ) -> Array:
    t = np.arange(T)
    sign = np.where((t // period) % 2 == 0, 1.0, -1.0)
    size = np.where((t >= T // 3) & (t < 2 * T // 3), large, small)
    return sign * size


def make_mus(T: int, seed: int) -> dict[str, Array]:
    rng = np.random.default_rng(seed)
    return {
            "stationary": mu_stationary(T),
            "smooth": mu_smooth(T, rng),
            "jump": mu_jump(T),
            }

# -----------------------------------------------------------------------------
# Exact coverage quantities
# -----------------------------------------------------------------------------


def alpha_star(mu: Array, alpha: float = ALPHA) -> Array:
    return 1 - norm.cdf(mu + norm.ppf(1 - alpha))


def true_coverage(alphas: Array, mu: Array) -> Array:
    return norm.cdf(norm.ppf(1 - np.clip(alphas, 0.0, 1.0)) - mu)


def coverage_gap(alphas: Array, mu: Array, alpha: float = ALPHA) -> Array:
    return np.abs(true_coverage(alphas, mu) - (1 - alpha))

# -----------------------------------------------------------------------------
# Trials
# -----------------------------------------------------------------------------


def run_trial(mu: Array, alpha: float, seed: int) -> dict[str, Array]:
    rng = np.random.default_rng(seed)
    y = rng.normal(mu, 1.0)
    betas = (1 - norm.cdf(y)).tolist()

    res_d = dtaci(betas, alpha)
    res_a = agaci(betas, alpha)
    res_m = mvp(betas, alpha, seed=seed)

    return {
            "alpha_dtaci": np.asarray(res_d["alphas"]),
            "alpha_agaci": np.asarray(res_a["alphas"]),
            "alpha_mvp": np.asarray(res_m["alphas"]),
            "gamma_dtaci": np.asarray(res_d["gamma_bar"]),
            "gamma_agaci": np.asarray(res_a["gamma_bar"]),
            }


def _run_trial_job(args: tuple[str, int, Array, float, int]) -> tuple[str, int, dict[str, Array]]:
    regime, trial, mu, alpha, seed = args
    return regime, trial, run_trial(mu, alpha, seed)


def run_experiment(
        T: int = T_DEFAULT,
        trials: int = TRIALS_DEFAULT,
        alpha: float = ALPHA,
        seed: int = 0,
        workers: int | None = None,
        ) -> dict[str, Any]:
    mus = make_mus(T, seed)
    seeds = np.random.SeedSequence(seed).generate_state(len(REGIMES) * trials)
    keys = ["alpha_dtaci", "alpha_agaci", "alpha_mvp", "gamma_dtaci", "gamma_agaci"]
    out: dict[str, Any] = {
            r: {k: np.empty((trials, T)) for k in keys} for r in REGIMES
            }

    jobs = [
            (r, j, mus[r], alpha, int(seeds[i * trials + j]))
            for i, r in enumerate(REGIMES)
            for j in range(trials)
            ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done, (r, j, res) in enumerate(ex.map(_run_trial_job, jobs), 1):
            for k in keys:
                out[r][k][j] = res[k]
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} trials done", flush=True)

    return {"mus": mus, "results": out}


def _cache_path(T: int, trials: int, alpha: float, seed: int) -> str:
    return str(CACHE / f"exp1_T{T}_n{trials}_a{alpha}_s{seed}_p{JUMP_PERIOD}.npz")


def load_or_run(
        T: int,
        trials: int,
        alpha: float,
        seed: int,
        workers: int | None,
        use_cache: bool = True,
        ) -> dict[str, Any]:
    path = _cache_path(T, trials, alpha, seed)
    if use_cache and os.path.exists(path):
        print(f"Loading cached results from {path}")
        data = np.load(path)
        mus = {r: data[f"mu_{r}"] for r in REGIMES}
        results = {
                r: {k.split("/", 1)[1]: data[k] for k in data.files if k.startswith(f"{r}/")}
                for r in REGIMES
                }
        return {"mus": mus, "results": results}

    print(f"Running {len(REGIMES)} regimes x {trials} trials (T = {T})")
    exp = run_experiment(T, trials, alpha, seed, workers)
    flat = {f"mu_{r}": exp["mus"][r] for r in REGIMES}
    for r in REGIMES:
        for k, v in exp["results"][r].items():
            flat[f"{r}/{k}"] = v.astype(np.float32)
    np.savez_compressed(path, **flat)
    return exp

# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def moving_average(x: Array, w: int = WINDOW) -> Array:
    out = np.full(len(x), np.nan)
    if len(x) < w:
        return out
    c = np.concatenate([[0.0], np.cumsum(x)])
    half = w // 2
    out[half:len(x) - w + half + 1] = (c[w:] - c[:-w]) / w
    return out


def mean_ci(x: Array, w: int = WINDOW) -> tuple[Array, Array, Array]:
    m = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    return moving_average(m, w), moving_average(m - 1.96 * se, w), moving_average(m + 1.96 * se, w)

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def _regime_lines(ax: plt.Axes, regime: str, T: int) -> None:
    if regime == "jump":
        for x in (T // 3, 2 * T // 3):
            ax.axvline(x, color="grey", linestyle=":", linewidth=1)


def _band(ax: plt.Axes, t: Array, x: Array, color: str, label: str) -> None:
    m, lo, hi = mean_ci(x)
    ax.plot(t, m, color=color, label=label, linewidth=1.2)
    ax.fill_between(t, lo, hi, color=color, alpha=0.25, linewidth=0)


def figure_4_1(mus: dict[str, Array]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, r in zip(axes, REGIMES):
        mu = mus[r]
        t = np.arange(len(mu))
        ax.plot(t, moving_average(mu) if r == "smooth" else mu, color="black", linewidth=1)
        _regime_lines(ax, r, len(mu))
        ax.set_title(LABELS[r])
        ax.set_xlabel("Time")
    axes[0].set_ylabel(r"$\mu_t$")
    fig.tight_layout()
    return fig


def figure_4_2(exp: dict[str, Any], alpha: float = ALPHA) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, r in zip(axes, REGIMES):
        res, mu = exp["results"][r], exp["mus"][r]
        t = np.arange(len(mu))
        for m in METHODS:
            _band(ax, t, res[f"alpha_{m}"], METHOD_COLORS[m], LABELS[m])
        ax.plot(t, alpha_star(mu, alpha), color=METHOD_COLORS["alpha_star"],
                linewidth=1, label=r"$\alpha_t^*$")
        _regime_lines(ax, r, len(mu))
        ax.set_title(LABELS[r])
        ax.set_xlabel("Time")
    axes[0].set_ylabel(r"$\alpha_t$, $\alpha_t^*$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def figure_4_3(exp: dict[str, Any], alpha: float = ALPHA) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for i, r in enumerate(REGIMES):
        res, mu = exp["results"][r], exp["mus"][r]
        T = len(mu)
        t = np.arange(T)
        gap = {m: coverage_gap(res[f"alpha_{m}"], mu, alpha) for m in METHODS}

        ax = axes[i, 0]
        _band(ax, t, gap["agaci"] - gap["dtaci"], METHOD_COLORS["agaci"], "AgACI vs. DtACI")
        _band(ax, t, gap["mvp"] - gap["dtaci"], METHOD_COLORS["mvp"], "MVP vs. DtACI")
        ax.axhline(0, color="black", linewidth=1)
        _regime_lines(ax, r, T)
        ax.set_ylabel("Difference in Coverage Gap")
        ax.text(0.02, 0.92, LABELS[r], transform=ax.transAxes, va="top")

        ax = axes[i, 1]
        for m in ("dtaci", "agaci"):
            _band(ax, t, res[f"gamma_{m}"], METHOD_COLORS[m], LABELS[m])
        _regime_lines(ax, r, T)
        ax.set_ylabel(r"$\bar\gamma_t$")
        ax.text(0.02, 0.92, LABELS[r], transform=ax.transAxes, va="top")

    for ax in axes[-1]:
        ax.set_xlabel("Time")
    axes[0, 0].legend(loc="upper right", ncol=2, frameon=False)
    axes[0, 1].legend(loc="upper right", ncol=2, frameon=False)
    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--T", type=int, default=T_DEFAULT)
    parser.add_argument("--trials", type=int, default=TRIALS_DEFAULT)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    exp = load_or_run(args.T, args.trials, args.alpha, args.seed, args.workers,
                      use_cache=not args.no_cache)

    use_style()
    for name, fig in (
            ("exp1_fig4_1_mu", figure_4_1(exp["mus"])),
            ("exp1_fig4_2_alpha", figure_4_2(exp, args.alpha)),
            ("exp1_fig4_3_coverage_gap", figure_4_3(exp, args.alpha)),
            ):
        savefig(fig, name)
        plt.close(fig)
        print(f"Saved figures/{name}.{{pdf,png}}")


if __name__ == "__main__":
    main()
