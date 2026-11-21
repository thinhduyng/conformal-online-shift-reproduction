"""Rolling GARCH(1, 1) forecasting and conformity score / beta computation.

Implements the baseline forecaster and conformity scores from Section 4.2.1 and Appendix D:
1. GARCH(1, 1) model without mean: sigma_t^2 = omega + tau * V_{t-1} + lambda * sigma_{t-1}^2
2. Rolling lookback window of r = 1250 days.
3. Normalized conformity score: S_t = |V_t - sigma_t^2| / sigma_t^2
4. Unnormalized conformity score: S_t = |V_t - sigma_t^2|
5. Empirical quantile nonconformity score: beta_t via binary search against past scores.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from arch import arch_model

from aci_repro.paths import CACHE
from aci_repro.data_stocks import load_stock_data, STOCK_FILES


def compute_garch_scores(
    returns: np.ndarray,
    lookback: int = 1250,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit rolling GARCH(1, 1) and compute one-step variance predictions and scores.
    
    Args:
        returns: Array of daily returns R_t (1-D array, clean of NaN).
        lookback: Rolling lookback window size (default 1250).
        
    Returns:
        (v_realized, sigma_sq, scores_norm, scores_unnorm)
        Each array has length T - lookback.
    """
    T = len(returns)
    if T <= lookback:
        raise ValueError(f"Returns length ({T}) must be greater than lookback ({lookback})")
        
    num_predictions = T - lookback
    sigma_sq = np.zeros(num_predictions, dtype=float)
    v_realized = np.zeros(num_predictions, dtype=float)
    scores_norm = np.zeros(num_predictions, dtype=float)
    scores_unnorm = np.zeros(num_predictions, dtype=float)
    
    prev_params = None
    
    for i in range(num_predictions):
        t = lookback + i
        window = returns[t - lookback : t]
        curr_return = returns[t]
        v_curr = curr_return ** 2
        
        # Fit zero-mean GARCH(1, 1)
        am = arch_model(window, mean="Zero", vol="Garch", p=1, q=1, dist="normal", rescale=False)
        try:
            res = am.fit(disp="off", show_warning=False, starting_values=prev_params)
            fc = res.forecast(horizon=1)
            var_pred = float(fc.variance.values[-1, 0])
            prev_params = res.params
        except Exception:
            # Fallback if convergence with warm start fails
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1)
            var_pred = float(fc.variance.values[-1, 0])
            prev_params = res.params
            
        # Avoid division by zero or negative variance
        var_pred = max(var_pred, 1e-12)
        
        diff = abs(v_curr - var_pred)
        sigma_sq[i] = var_pred
        v_realized[i] = v_curr
        scores_norm[i] = diff / var_pred
        scores_unnorm[i] = diff
        
    return v_realized, sigma_sq, scores_norm, scores_unnorm


def find_beta(recent_scores: np.ndarray, cur_score: float, epsilon: float = 0.001) -> float:
    """Find beta_t via binary search, matching author's findBeta implementation.
    
    Finds beta such that Quantile(1 - beta, recent_scores) approx cur_score.
    """
    top = 1.0
    bot = 0.0
    mid = (top + bot) / 2.0
    
    while (top - bot) > epsilon:
        # np.quantile with method='linear' matches R's default type 7 quantile
        q = np.quantile(recent_scores, 1.0 - mid, method="linear")
        if q > cur_score:
            bot = mid
        else:
            top = mid
        mid = (top + bot) / 2.0
        
    return mid


def compute_betas(scores: np.ndarray, lookback: int = 1250, epsilon: float = 0.001) -> np.ndarray:
    """Compute beta_t sequence from scores using rolling recentScores window.
    
    Args:
        scores: Sequence of conformity scores of length M.
        lookback: Maximum number of recent scores to look back (default 1250).
        epsilon: Precision for binary search (default 0.001).
        
    Returns:
        Array of beta_t of length M - 1 (starts at t=1, using scores[0] as past reference).
    """
    M = len(scores)
    beta_seq = np.zeros(M - 1, dtype=float)
    
    for t in range(1, M):
        start_idx = max(0, t - lookback)
        recent_scores = scores[start_idx:t]
        beta_seq[t - 1] = find_beta(recent_scores, scores[t], epsilon=epsilon)
        
    return beta_seq


def process_stock(stock_name: str, force_recompute: bool = False) -> Dict[str, np.ndarray]:
    """Compute (or load from cache) GARCH forecasts, conformity scores, and betas for a stock.
    
    Cache file: cache/{stock_name}_garch.npz
    """
    cache_file = CACHE / f"{stock_name.replace(' ', '_')}_garch.npz"
    
    if cache_file.exists() and not force_recompute:
        data = np.load(cache_file, allow_pickle=True)
        return {key: data[key] for key in data.files}
        
    df = load_stock_data(stock_name)
    # Exclude initial NaN return
    clean_df = df.iloc[1:].reset_index(drop=True)
    returns = clean_df["Return"].to_numpy(dtype=float)
    dates = clean_df["Date"].to_numpy()
    
    lookback = 1250
    v_realized, sigma_sq, scores_norm, scores_unnorm = compute_garch_scores(returns, lookback=lookback)
    
    # Dates corresponding to scores (from index lookback onwards)
    score_dates = dates[lookback:]
    
    # Betas are computed for scores[1:]
    betas_norm = compute_betas(scores_norm, lookback=lookback)
    betas_unnorm = compute_betas(scores_unnorm, lookback=lookback)
    
    # Dates corresponding to betas (from score_dates[1:] onwards)
    beta_dates = score_dates[1:]
    
    # Realized variance and sigma_sq aligned with betas
    v_aligned = v_realized[1:]
    sigma_sq_aligned = sigma_sq[1:]
    scores_norm_aligned = scores_norm[1:]
    scores_unnorm_aligned = scores_unnorm[1:]
    
    result = {
        "stock": np.array(stock_name),
        "beta_dates": beta_dates,
        "v_realized": v_aligned,
        "sigma_sq": sigma_sq_aligned,
        "scores_norm": scores_norm_aligned,
        "scores_unnorm": scores_unnorm_aligned,
        "betas_norm": betas_norm,
        "betas_unnorm": betas_unnorm,
    }
    
    np.savez_compressed(cache_file, **result)
    print(f"[{stock_name}] Successfully computed and cached ({len(betas_norm)} points).")
    return result


def _process_wrapper(stock_name: str) -> Tuple[str, Dict[str, np.ndarray]]:
    res = process_stock(stock_name)
    return stock_name, res


def process_all_stocks(parallel: bool = True, force_recompute: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute and cache GARCH results for all 4 stocks."""
    results = {}
    stocks = list(STOCK_FILES.keys())
    
    if parallel:
        with ProcessPoolExecutor(max_workers=min(4, len(stocks))) as executor:
            futures = [executor.submit(process_stock, s, force_recompute) for s in stocks]
            for s, fut in zip(stocks, futures):
                results[s] = fut.result()
    else:
        for s in stocks:
            results[s] = process_stock(s, force_recompute=force_recompute)
            
    return results


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("Starting rolling GARCH pipeline for all stocks...")
    res = process_all_stocks(parallel=True)
    t1 = time.time()
    print(f"Done processing all stocks in {t1 - t0:.2f} seconds.")
