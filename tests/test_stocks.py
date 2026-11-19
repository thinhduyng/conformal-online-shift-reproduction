"""Synthetic checks for the stock experiment wrappers (no data or GARCH needed)."""

import numpy as np

from aci_repro.experiment2 import run_all_methods


def test_run_all_methods_on_uniform_betas():
    betas = np.random.default_rng(123).uniform(size=2000)
    res = run_all_methods(betas, "synthetic")
    assert set(res) == {"DtACI", "AgACI", "MVP", "Constant"}
    for name, r in res.items():
        assert r["err"].shape == r["alphas"].shape == (2000,)
        assert set(np.unique(r["err"])) <= {0.0, 1.0}
        # every method should be close to 90% coverage on exchangeable scores
        assert abs(1 - r["err"].mean() - 0.9) < 0.03, name
    assert np.array_equal(res["Constant"]["err"], (betas < 0.1).astype(float))
