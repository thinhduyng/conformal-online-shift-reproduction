"""Data loading and preprocessing for stock market volatility experiments.

Datasets:
- Nvidia (NVDA): 1999-01-22 to 2021-04-30
- AMD (AMD): 1991-07-22 to 2021-04-30 (last 7501 days)
- BlackBerry (BB): 1999-02-04 to 2021-04-30
- Fannie Mae (FNMA): 1991-07-23 to 2021-04-30 (last 7501 days)

Data is downloaded automatically from the author's public repository (isgibbs/DtACI)
if not present locally, avoiding storing raw datasets in git.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import ssl
from typing import Dict
import urllib.request
import numpy as np
import pandas as pd

from aci_repro.paths import DATA

STOCK_FILES = {
    "Nvidia": "NividiaDailyPrices.csv",
    "AMD": "AMDDailyPrices.csv",
    "BlackBerry": "BlackberryDailyPrices.csv",
    "Fannie Mae": "FannieMaeDailyPrices.csv",
}

# The original paper / author implementation truncates AMD and Fannie Mae to the last 7500 returns (7501 price days)
LOOKBACK_TRUNCATION = {
    "AMD": 7501,
    "Fannie Mae": 7501,
}

PUBLIC_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/isgibbs/DtACI/main/MajorDailyReturns"
)


def download_stock_file(filename: str, out_path: Path) -> Path:
    """Download a stock CSV file from the public repository (isgibbs/DtACI) if missing."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PUBLIC_RAW_BASE_URL}/{filename}"
    print(f"Downloading {filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
    print(f"Saved {out_path}")
    return out_path


def download_all_stocks(dest_dir: Path = DATA) -> None:
    """Download all stock datasets from the public repository."""
    for filename in STOCK_FILES.values():
        download_stock_file(filename, dest_dir / filename)


def load_stock_data(name: str, source: str = "auto") -> pd.DataFrame:
    """Load and preprocess a single stock's daily open prices and returns.
    
    Args:
        name: One of 'Nvidia', 'AMD', 'BlackBerry', 'Fannie Mae'.
        source: 'auto' (load local file, download from public repo if not found),
                'local' (only load local file),
                'public' (fetch directly from public URL).
        
    Returns:
        DataFrame with columns ['Date', 'Open', 'Return', 'RealizedVol'] sorted chronologically.
    """
    if name not in STOCK_FILES:
        raise ValueError(f"Unknown stock: {name}. Must be one of {list(STOCK_FILES.keys())}")
    
    filename = STOCK_FILES[name]
    csv_path = DATA / filename

    if source == "public":
        url = f"{PUBLIC_RAW_BASE_URL}/{filename}"
        try:
            df = pd.read_csv(url)
        except Exception:
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ctx) as resp:
                df = pd.read_csv(resp)
    else:
        if not csv_path.exists():
            download_stock_file(filename, csv_path)
        df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    
    # Sort chronologically (CSVs are originally in reverse chronological order)
    df = df.iloc[::-1].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    
    # Truncate if specified in author benchmark
    if name in LOOKBACK_TRUNCATION:
        k = LOOKBACK_TRUNCATION[name]
        df = df.iloc[-k:].reset_index(drop=True)
        
    # Compute daily return: R_t = (P_t - P_{t-1}) / P_{t-1}
    prices = df["Open"].to_numpy(dtype=float)
    returns = np.zeros(len(prices))
    returns[0] = np.nan
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    df["Return"] = returns
    df["RealizedVol"] = returns ** 2
    
    return df


def load_all_stocks(source: str = "auto") -> Dict[str, pd.DataFrame]:
    """Load all 4 stock datasets used in Experiments 2, 3, and 5a."""
    return {name: load_stock_data(name, source=source) for name in STOCK_FILES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download public stock datasets.")
    sub = parser.add_subparsers(dest="cmd")
    d = sub.add_parser("download", help="Download all stock datasets from public repository")
    args = parser.parse_args()
    download_all_stocks()
