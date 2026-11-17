# conformal-online-shift-reproduction
Reproduction &amp; analysis of "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts". Link: https://jmlr.org/papers/v25/22-1218.html

## Setup

```bash
uv sync                        # creates .venv with the exact locked versions
uv run nbstripout --install    # once per clone: strips notebook outputs on commit
uv run pytest                  # sanity checks
```

Code lives in `src/aci_repro/` (import as `aci_repro`); notebooks in `notebooks/` should stay thin and call into it.
Use `aci_repro.paths` (`DATA`, `CACHE`, `FIGURES`) instead of relative paths, and `aci_repro.plotting` (`use_style`, `METHOD_COLORS`, `savefig`) for figures.
`data/` and `cache/` are gitignored; `figures/` is committed.
