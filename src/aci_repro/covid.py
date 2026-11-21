"""COVID-19 county panel, 7-day-ahead forecaster, scaled scores and betas (Exp. 4 / 5b).

Pipeline (Section 4.2.3 of the paper)
    download_delphi()  -> data/covid_county_panel.csv.gz   (long: date, county, CO, F)
    upload_to_hf()     -> Hugging Face dataset (the course requires HF/Kaggle hosting)
    load_panel()       -> Panel(dates, fips, co[T,n], fa[T,n])   (no missing data)
    build_betas()      -> Scores(co_hat, score, beta)            (all arrays [T,n])

CLI
    uv run python -m aci_repro.covid download --start 2020-07-01 --end 2022-02-28
    uv run python -m aci_repro.covid upload --repo <user>/covid-county-panel
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aci_repro.paths import CACHE, DATA

SEED = 0
RAW_FILE = "covid_county_panel.csv.gz"
RAW_CSV = DATA / RAW_FILE

# Four urban counties highlighted in the paper (FIPS codes).
COUNTIES = {
    "San Francisco": "06075",
    "New York": "36061",
    "Miami-Dade": "12086",
    "Dallas": "48113",
}

HORIZON = 7  # forecast 7 days ahead
TRAIN_WINDOW = 14  # least squares refit on the last 14 days of origins
# ASSUMPTION: the summary of the paper says lags t-7j (j=1..3); the score denominator
# |CO_{t-7} - CO_t| suggests the most recent value (lag 0) is a feature. Both are
# supported; compare them in the deviation analysis.
LAGS = (0, 7, 14)

API = "https://api.delphi.cmu.edu/epidata/covidcast/"
SIGNALS = {
    "CO": ("jhu-csse", "confirmed_7dav_incidence_num"),
    "F": ("fb-survey", "smoothed_hh_cmnty_cli"),
}


# -----------------------------------------------------------------------------
# 1. Data acquisition
# -----------------------------------------------------------------------------
def _get(params: dict, retries: int = 4) -> list[dict]:
    """GET one Delphi request with retry/backoff. Returns the 'epidata' rows."""
    import requests

    key = os.environ.get("DELPHI_API_KEY")  # free key avoids the anonymous rate limit
    if key:
        params = {**params, "api_key": key}
    for k in range(retries):
        try:
            r = requests.get(API, params=params, timeout=90)
            r.raise_for_status()
            js = r.json()
            if js.get("result") == 1:
                return js["epidata"]
            if js.get("result") == -2:  # "no results"
                return []
            raise RuntimeError(js.get("message", js))
        except Exception as e:  # noqa: BLE001 - network flakiness is expected
            if k == retries - 1:
                print(f"[warn] request failed permanently: {e}")
                return []
            time.sleep(2**k)
    return []


def download_delphi(start: str, end: str, out: Path = RAW_CSV, geo_chunk: int = 200,
                    days_per_req: int = 15, workers: int = 4) -> Path:
    """Download county-level CO and F from the public Delphi COVIDcast API.

    Strategy (to keep the number of requests small): discover counties that report the
    Facebook signal on one mid-period day, then request (chunk of counties) x (block of
    days) per call. Set DELPHI_API_KEY (free) to avoid the anonymous rate limit.
    Signal names: https://cmu-delphi.github.io/delphi-epidata/api/covidcast_signals.html
    """
    mid = (pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2).strftime("%Y%m%d")
    src, sig = SIGNALS["F"]
    rows = _get(dict(data_source=src, signals=sig, time_type="day", geo_type="county",
                     time_values=mid, geo_values="*"))
    geos = sorted({r["geo_value"] for r in rows})
    print(f"counties reporting F on {mid}: {len(geos)}")
    if not geos:
        raise RuntimeError("Discovery request returned no counties: check signal names / API key.")

    days = pd.date_range(start, end)
    blocks = [days[i:i + days_per_req] for i in range(0, len(days), days_per_req)]
    chunks = [geos[i:i + geo_chunk] for i in range(0, len(geos), geo_chunk)]
    jobs = [(name, g, b) for name in SIGNALS for g in chunks for b in blocks]
    print(f"{len(jobs)} requests ...")

    def one(job):
        name, g, b = job
        s, sg = SIGNALS[name]
        tv = f"{b[0]:%Y%m%d}-{b[-1]:%Y%m%d}"
        ep = _get(dict(data_source=s, signals=sg, time_type="day", geo_type="county",
                       time_values=tv, geo_values=",".join(g)))
        return [(r["time_value"], r["geo_value"], name, r["value"]) for r in ep]

    with ThreadPoolExecutor(workers) as ex:
        allrows = [r for res in ex.map(one, jobs) for r in res]
    long = pd.DataFrame(allrows, columns=["date", "county", "sig", "value"])
    wide = long.pivot_table(index=["date", "county"], columns="sig", values="value").reset_index()
    wide["date"] = pd.to_datetime(wide["date"].astype(str), format="%Y%m%d")
    out.parent.mkdir(exist_ok=True)
    wide.to_csv(out, index=False)
    print(f"saved {out}  ({len(wide)} rows)")
    return out


def upload_to_hf(repo_id: str, path: Path = RAW_CSV) -> None:
    """Upload the prepared panel to a Hugging Face *dataset* repo (needs `hf auth login`)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=RAW_FILE, repo_id=repo_id,
                    repo_type="dataset")
    print(f"uploaded to https://huggingface.co/datasets/{repo_id}")


def download_from_hf(repo_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, RAW_FILE, repo_type="dataset"))


# -----------------------------------------------------------------------------
# 2. Panel
# -----------------------------------------------------------------------------
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    fips: list[str]
    co: np.ndarray  # (T, n) 7-day average confirmed cases
    fa: np.ndarray  # (T, n) 7-day average FB survey signal


def panel_from_long(df: pd.DataFrame) -> Panel:
    """Pivot a long table (date, county, CO, F) and drop counties with ANY missing value."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["county"] = df["county"].astype(str).str.zfill(5)
    co = df.pivot(index="date", columns="county", values="CO").sort_index()
    fa = df.pivot(index="date", columns="county", values="F").sort_index()
    full = pd.date_range(co.index.min(), co.index.max())
    co, fa = co.reindex(full), fa.reindex(full)  # missing days become NaN -> county dropped
    keep = [c for c in co.columns if c in fa.columns and not co[c].isna().any() and not fa[c].isna().any()]
    return Panel(co.index, keep, co[keep].to_numpy(float), fa[keep].to_numpy(float))


def make_synthetic(n: int = 60, T: int = 450, seed: int = SEED) -> Panel:
    """Multi-wave synthetic panel for smoke tests and unit tests (no downloads)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    co = np.zeros((T, n))
    for i in range(n):
        scale = rng.uniform(20, 800)
        for c in rng.uniform(0, T, size=3):
            co[:, i] += scale * rng.uniform(0.5, 1.5) * np.exp(-0.5 * ((t - c) / rng.uniform(15, 30)) ** 2)
    co *= rng.lognormal(0, 0.1, size=co.shape)
    fa = np.roll(co, -5, axis=0) / (co.max(0) + 1) * 0.3 + rng.normal(0, 0.01, co.shape)
    fips = list(COUNTIES.values()) + [f"9{j:04d}" for j in range(n - 4)]
    return Panel(pd.date_range("2020-07-01", periods=T), fips, co, fa)


def load_panel(source: str = "csv", repo: str | None = None, path: Path = RAW_CSV) -> Panel:
    """source: 'synthetic' | 'csv' (local file) | 'hf' (Hugging Face dataset repo)."""
    if source == "synthetic":
        return make_synthetic()
    if source == "hf":
        assert repo, "--hf-repo is required for source=hf"
        path = download_from_hf(repo)
    return panel_from_long(pd.read_csv(path, dtype={"county": str}))


# -----------------------------------------------------------------------------
# 3. Forecaster, scaled score, cross-county beta
# -----------------------------------------------------------------------------
@dataclass
class Scores:
    co_hat: np.ndarray  # (T, n) forecast of CO[u] made at day u - HORIZON (NaN where undefined)
    score: np.ndarray  # (T, n) scaled absolute residual S[u, i]
    beta: np.ndarray  # (T, n) conformal p-value-like beta[u, i] in [0, 1]


def build_betas(co: np.ndarray, fa: np.ndarray, lags: tuple[int, ...] = LAGS,
                horizon: int = HORIZON, window: int = TRAIN_WINDOW,
                den_floor: float = 1.0) -> Scores:
    """Refit a pooled OLS every day, forecast `horizon` days ahead, score and beta.

    Forecast of day u is made at origin t0 = u - horizon. The regression is fit on pooled
    (county, origin s) samples with s in [t0-horizon-window+1, t0-horizon], so every training
    target co[s+horizon] <= co[t0] is already observed at t0 (no look-ahead).
    Features: 1, co[s-l], fa[s-l] for l in `lags`.

    score[u,i] = |co_hat[u,i] - co[u,i]| / max(|co[u-horizon,i] - co[u,i]|, den_floor)
    beta[u,i]  = mean_j 1{ score[u-1,j] >= score[u,i] }
               = sup{a : score[u,i] <= empirical (1-a)-quantile of {score[u-1,j]}_j}
    """
    T, n = co.shape
    mlag = max(lags)

    def feats(s: int) -> np.ndarray:
        cols = [np.ones(n)]
        for l in lags:
            cols += [co[s - l], fa[s - l]]
        return np.column_stack(cols)

    co_hat = np.full((T, n), np.nan)
    for u in range(T):
        t0 = u - horizon
        origins = range(t0 - horizon - window + 1, t0 - horizon + 1)
        if t0 - horizon - window + 1 - mlag < 0:
            continue
        X = np.vstack([feats(s) for s in origins])
        y = np.concatenate([co[s + horizon] for s in origins])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        co_hat[u] = feats(t0) @ coef

    score = np.full((T, n), np.nan)
    for u in range(horizon, T):
        den = np.maximum(np.abs(co[u - horizon] - co[u]), den_floor)
        score[u] = np.abs(co_hat[u] - co[u]) / den

    beta = np.full((T, n), np.nan)
    for u in range(1, T):
        prev = score[u - 1][~np.isnan(score[u - 1])]
        if prev.size == 0 or np.isnan(score[u]).all():
            continue
        beta[u] = (prev[None, :] >= score[u][:, None]).mean(axis=1)
    return Scores(co_hat, score, beta)


def cached_betas(panel: Panel, tag: str, **kw) -> Scores:
    """Compute betas once and cache them in cache/ (gitignored)."""
    f = CACHE / f"covid_betas_{tag}.npz"
    if f.exists():
        z = np.load(f)
        return Scores(z["co_hat"], z["score"], z["beta"])
    s = build_betas(panel.co, panel.fa, **kw)
    np.savez_compressed(f, co_hat=s.co_hat, score=s.score, beta=s.beta)
    return s


def county_betas(s: Scores, j: int) -> np.ndarray:
    """Beta sequence of county column j with the undefined warm-up prefix removed."""
    b = s.beta[:, j]
    first = int(np.argmax(~np.isnan(b)))
    out = b[first:]
    assert not np.isnan(out).any(), "NaN inside the beta sequence (gap in the data?)"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", default="2020-07-01")
    d.add_argument("--end", default="2022-02-28")
    u = sub.add_parser("upload")
    u.add_argument("--repo", required=True)
    a = ap.parse_args()
    if a.cmd == "download":
        download_delphi(a.start, a.end)
    else:
        upload_to_hf(a.repo)
