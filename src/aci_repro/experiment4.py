"""Experiment 4 (COVID-19 case counts, Sec. 4.2.3) and Experiment 5b (fixed vs variable eta).

Produces
    fig_4_8_covid_local_coverage   Constant / MVP / AgACI / DtACI, 200-step local + global coverage
    fig_d3_covid_eta_ablation      DtACI with fixed eta vs variable eta_t
    fig_d7_covid_cases             7-day average case counts of the 4 counties
    covid_global_coverage.csv      global coverage table (goes into the deck)

Run
    uv run python -m aci_repro.experiment4 --source hf --hf-repo <user>/covid-county-panel --wandb
    uv run python -m aci_repro.experiment4 --source synthetic        # smoke test
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aci_repro.conformal_methods import global_coverage, local_coverage, run_method
from aci_repro.covid import COUNTIES, HORIZON, LAGS, TRAIN_WINDOW, cached_betas, county_betas, load_panel
from aci_repro.paths import FIGURES
from aci_repro.plotting import savefig, use_style

ALPHA = 0.1
LOCAL_WINDOW = 200
COLORS = {"DtACI": "#1f77b4", "AgACI": "#d62728", "MVP": "#2ca02c", "Constant": "#7f7f7f",
          "DtACI (var. eta)": "#9467bd"}
# label -> (method name in conformal_methods.METHODS, extra kwargs)
RUNS = {
    "Constant": ("constant", {}),
    "MVP": ("mvp", {}),
    "AgACI": ("agaci", {}),
    "DtACI": ("dtaci", {}),
    "DtACI (var. eta)": ("dtaci", {"variable_eta": True}),
}


def run_county(betas: np.ndarray) -> dict[str, dict]:
    """Run every method on one county's beta sequence (list[float] for conformal_methods)."""
    b = betas.tolist()
    out = {}
    for label, (name, kw) in RUNS.items():
        t0 = time.time()
        out[label] = run_method(name, b, ALPHA, **kw)
        print(f"    {label:<17s} {time.time() - t0:5.1f}s  global cov = {global_coverage(out[label]['err']):.3f}")
    return out


def plot_local(results: dict[str, dict], methods: list[str], name: str, title: str, prefix: str = "") -> None:
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for ax, (county, res) in zip(axes.ravel(), results.items()):
        for m in methods:
            err = res[m]["err"]
            (line,) = ax.plot(local_coverage(err, LOCAL_WINDOW), color=COLORS[m], lw=1.3, label=m)
            ax.axhline(global_coverage(err), color=COLORS[m], ls="--", lw=0.9)
        ax.axhline(1 - ALPHA, color="k", lw=0.6)
        ax.set_title(county)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("time step")
        ax.set_ylabel("local coverage")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    savefig(fig, f"{prefix}{name}")
    plt.close(fig)


def plot_cases(dates, co: np.ndarray, cols: dict[str, int], prefix: str = "") -> None:
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, (county, j) in zip(axes.ravel(), cols.items()):
        ax.plot(dates, co[:, j])
        ax.set_title(county)
        ax.set_ylabel("7-day avg cases")
    fig.suptitle("Confirmed COVID-19 cases (7-day average)")
    fig.tight_layout()
    savefig(fig, f"{prefix}fig_d7_covid_cases")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "csv", "hf"], default="csv")
    ap.add_argument("--hf-repo")
    ap.add_argument("--all-counties", action="store_true",
                    help="also report mean global coverage over every county (slow)")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    np.random.seed(0)

    panel = load_panel(a.source, a.hf_repo)
    T, n = panel.co.shape
    print(f"panel: {T} days x {n} counties, {panel.dates[0].date()} .. {panel.dates[-1].date()}")
    cols = {nm: panel.fips.index(f) for nm, f in COUNTIES.items() if f in panel.fips}
    if len(cols) < 4:
        print(f"[warn] counties missing after the no-missing-data filter: {set(COUNTIES) - set(cols)}")

    scores = cached_betas(panel, tag=a.source)
    print(f"beta mean = {np.nanmean(scores.beta):.3f} (about 0.5 if scores are exchangeable)")

    results = {}
    for nm, j in cols.items():
        print(f"  {nm}")
        results[nm] = run_county(county_betas(scores, j))

    table = pd.DataFrame({m: {nm: global_coverage(r[m]["err"]) for nm, r in results.items()} for m in RUNS})
    print("\nGlobal coverage (target 0.90):\n", table.round(3))
    pre = "synthetic_" if a.source == "synthetic" else ""  # never overwrite real figures
    table.round(4).to_csv(FIGURES / f"{pre}covid_global_coverage.csv")

    plot_cases(panel.dates, panel.co, cols, pre)
    plot_local(results, ["Constant", "MVP", "AgACI", "DtACI"], "fig_4_8_covid_local_coverage",
               "COVID-19: local coverage (200-step window); dashed = global coverage", pre)
    plot_local(results, ["DtACI", "DtACI (var. eta)"], "fig_d3_covid_eta_ablation",
               "COVID-19: DtACI with fixed eta vs variable eta_t", pre)

    all_cov = None
    if a.all_counties:
        all_cov = {m: [] for m in RUNS}
        for j in range(n):
            b = county_betas(scores, j).tolist()
            for label, (name, kw) in RUNS.items():
                all_cov[label].append(global_coverage(run_method(name, b, ALPHA, **kw)["err"]))
        all_cov = {m: float(np.mean(v)) for m, v in all_cov.items()}
        print("\nMean global coverage over ALL counties:", {m: round(v, 3) for m, v in all_cov.items()})

    if a.wandb:
        import wandb

        run = wandb.init(project="conformal-dtaci-repro", name=f"exp4-covid-{a.source}", config=dict(
            alpha=ALPHA, local_window=LOCAL_WINDOW, horizon=HORIZON, train_window=TRAIN_WINDOW,
            lags=LAGS, n_days=T, n_counties=n, source=a.source, seed=0))
        run.log({"global_coverage": wandb.Table(dataframe=table.reset_index())})
        if all_cov:
            run.log({f"all_counties/{m}": v for m, v in all_cov.items()})
        for f in ("fig_4_8_covid_local_coverage", "fig_d3_covid_eta_ablation", "fig_d7_covid_cases"):
            run.log({f: wandb.Image(str(FIGURES / f"{pre}{f}.png"))})
        print("W&B run:", run.url)
        run.finish()


if __name__ == "__main__":
    main()
