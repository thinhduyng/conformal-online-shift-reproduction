## Experiment 4 & 5b: COVID-19 case counts (Sec. 4.2.3)

### Prerequisites
- Git and [uv](https://docs.astral.sh/uv/). `uv` downloads Python 3.12 automatically.
  - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
    then **close and reopen the terminal**.
- A free [Weights & Biases](https://wandb.ai) account (only for the logging step).

```bash
git clone <repo-url>
cd conformal-online-shift-reproduction
git checkout feat/covid19-experiments   # skip once merged into main
```

### Step 0: install the environment
```bash
uv sync
uv run pytest        # sanity check, expect: all tests passed
```

### Step 1: log in to W&B
```bash
uv run wandb login   # paste your API key from https://wandb.ai/authorize
```

### Step 2: run the experiment (data is downloaded from Hugging Face)
```bash
uv run python -m aci_repro.experiment4 --source hf --hf-repo lochithanh/covid-county-panel --wandb
```
