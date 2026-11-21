"""Synthetic-only checks for aci_repro.covid (no downloads)."""

import numpy as np

from aci_repro.conformal_methods import global_coverage, run_method
from aci_repro.covid import build_betas, county_betas, make_synthetic


def test_betas_are_valid_probabilities():
    p = make_synthetic(n=40, T=300)
    s = build_betas(p.co, p.fa)
    b = s.beta[~np.isnan(s.beta)]
    assert b.size > 0 and b.min() >= 0 and b.max() <= 1
    assert abs(b.mean() - 0.5) < 0.1  # roughly uniform when scores are exchangeable


def test_no_lookahead_in_forecast():
    """Changing data after the forecast origin t0 = u - 7 must not change co_hat[u]."""
    p = make_synthetic(n=30, T=300)
    u = 200
    base = build_betas(p.co, p.fa).co_hat[u]
    co2, fa2 = p.co.copy(), p.fa.copy()
    co2[u - 7 + 1:] += 1e3  # corrupt everything strictly after the origin
    fa2[u - 7 + 1:] += 1e3
    assert np.allclose(base, build_betas(co2, fa2).co_hat[u])


def test_dtaci_covers_on_covid_betas():
    p = make_synthetic(n=40, T=450)
    s = build_betas(p.co, p.fa)
    res = run_method("dtaci", county_betas(s, 0).tolist(), 0.1)
    assert abs(global_coverage(res["err"]) - 0.9) < 0.05
