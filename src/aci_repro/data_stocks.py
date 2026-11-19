"""Data loading and preprocessing for stock market volatility experiments.

Datasets:
- Nvidia (NVDA): 1999-01-22 to 2021-04-30
- AMD (AMD): 1991-07-22 to 2021-04-30 (last 7501 days)
- BlackBerry (BB): 1999-02-04 to 2021-04-30
- Fannie Mae (FNMA): 1991-07-23 to 2021-04-30 (last 7501 days)
"""

from typing import Dict
import pandas as pd
import numpy as np

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


def load_stock_data(name: str) -> pd.DataFrame:
    """Load and preprocess a single stock's daily open prices and returns.
    
    Args:
        name: One of 'Nvidia', 'AMD', 'BlackBerry', 'Fannie Mae'.
        
    Returns:
        DataFrame with columns ['Date', 'Open', 'Return', 'RealizedVol'] sorted chronologically.
    """
    if name not in STOCK_FILES:
        raise ValueError(f"Unknown stock: {name}. Must be one of {list(STOCK_FILES.keys())}")
    
    csv_path = DATA / STOCK_FILES[name]
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


def load_all_stocks() -> Dict[str, pd.DataFrame]:
    """Load all 4 stock datasets used in Experiments 2, 3, and 5a."""
    return {name: load_stock_data(name) for name in STOCK_FILES}
