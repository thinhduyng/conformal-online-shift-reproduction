# Summary of Experiments: "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts" (Gibbs & Candès, 2024)

## Common Hyperparameters Across All Experiments

- **Target coverage level ($1 - \alpha$):** $0.9$ ($\alpha = 0.1$)
- **Candidate step-sizes ($\{\gamma_i\}_{i=1}^k$ for DtACI & AgACI):** $\{0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128\}$ ($k = 8$)
- **DtACI Parameters (Section 3.1):** Interval size $|I| = 500$, $\sigma = \frac{1}{2|I|} = 0.001$, and fixed heuristic:
  $$\eta = \sqrt{\frac{3}{500} \frac{\log(2k \cdot 500) + 1}{(1 - \alpha)^2 \alpha^2}} \approx 2.72$$
  (using the deterministic variant in Algorithm 2).
- **MVP Parameters (Appendix A, Algorithm 3):** $m = 40$ bins, $\eta = \sqrt{\frac{\log(m)}{4.2m}}$, $r = 800000$.

---

## Experiment 1: Simulated Distribution Shifts (Section 4.1)

Evaluates how well the algorithms track a known ground-truth optimal parameter $\alpha_t^*$ under controlled regimes (Figures 4.1, 4.2, and 4.3).

### 1. Dataset / Required Input
- Synthetic sequence of independent observations $Y_t \sim \mathcal{N}(\mu_t, 1)$ for $T = 6000$ time steps (averaged over $100$ independent trials with fixed $\mu_t$ trajectories).
- Three settings for the mean trajectory $\{\mu_t\}_{t=1}^T$:
  - **Stationary:** $\mu_t = 0$ for all $t$.
  - **Smooth shift:** $\mu_0 = 0$, $\mu_{t+1} = \mu_t + \frac{1}{2}(\mu_t - \mu_{t-1}) + \frac{1}{2}\varepsilon_t$, where $\varepsilon_t \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 0.006)$.
  - **Jump shift:** Divided into $3$ equal intervals of $2000$ steps. In intervals $1$ and $3$, $\mu_t$ oscillates between $-0.075$ and $0.075$; in interval $2$, $\mu_t$ undergoes large oscillations between $-1.5$ and $1.5$.

### 2. Algorithms
- **Quantile Estimator:** Standard normal quantile, $\hat{C}_t(\alpha_t) = \{y : y \le \operatorname{Quantile}(1 - \alpha_t, \mathcal{N}(0, 1))\}$.
- **Conformal Methods:** DtACI (Algorithm 2), AgACI (Section 2.4), and MVP (Algorithm 3).

### 3. Desired Output / Performance
- **Trajectories of $\alpha_t$ vs. true $\alpha_t^*$ (Figure 4.2):** DtACI and AgACI track $\alpha_t^*$ closely in all $3$ settings; MVP only succeeds in the stationary setting and fails to adapt to shifts.
- **Instantaneous Coverage Gap Difference $\text{CG}_{\text{comp}} - \text{CG}_{\text{DtACI}}$ & Step-Size $\bar{\gamma}_t$ (Figure 4.3):** where $\text{CG}_x = |\mathbb{P}(Y_t \in \hat{C}(\alpha_t^x) \mid \alpha_t^x) - (1 - \alpha)|$.
  - **Stationary:** AgACI and MVP slightly outperform DtACI (difference slightly $< 0$).
  - **Smooth:** DtACI and AgACI perform almost identically (difference $\approx 0$); MVP has large positive gaps.
  - **Jump:** DtACI adapts its mean step-size $\bar{\gamma}_t = \sum_i p_t^i \gamma_i$ up in phase $2$ and back down in phase $3$. AgACI lags behind in phase $2$ and fails to decay $\bar{\gamma}_t$ in phase $3$, causing higher coverage gaps.

---

## Experiment 2: Online Prediction in the Stock Market (Section 4.2.1 & Appendix D)

Tests adaptability to real-world volatility shifts under two conformity score scales (Figures 4.4, 4.5, 4.6, D.4, D.6).

### 1. Dataset / Required Input
- Historical daily open stock prices $\{P_t\}$ (roughly 2000–2021) for 4 stocks: Nvidia, AMD, BlackBerry, and Fannie Mae.
- Daily return $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ and realized volatility $V_t = R_t^2$.
- Rolling lookback window of the most recent $r = 1250$ days.

### 2. Algorithms
- **Baseline Forecaster:** $\text{GARCH}(1, 1)$ model ($\sigma_t^2 = \omega + \tau V_{t-1} + \lambda \sigma_{t-1}^2$) refit at each step $t$ on $\{R_s\}_{t-1250 \le s < t}$ to produce a one-step-ahead volatility prediction $(\hat{\sigma}_t^t)^2$.
- **Two Conformity Score Variants:**
  - **Normalized score (Figures 4.4 & 4.6):** $S_t(v) = \frac{|v - (\hat{\sigma}_t^t)^2|}{(\hat{\sigma}_t^t)^2}$ (smaller distribution shift).
  - **Unnormalized score (Figures 4.5 & D.4):** $S_t(v) = |v - (\hat{\sigma}_t^t)^2|$ (much larger distribution shift).
- **Conformal Methods:** DtACI, AgACI, MVP, and a Constant Baseline ($\alpha_t = \alpha = 0.1$ fixed).

### 3. Desired Output / Performance
- **Local Coverage Plots (Figures 4.4 & 4.5):** Moving $500$-day average $\text{LocalCov}_t = 1 - \frac{1}{500}\sum_{s=t-249}^{t+250} \text{err}_s$ and global coverage (dashed lines). DtACI and AgACI maintain local coverage tightly around $0.90$ across all stocks and both normalized and unnormalized scores without tuning, whereas MVP and the Constant Baseline suffer severe under/over-coverage dips (dropping as low as $0.4$ on unnormalized Fannie Mae during the 2008 financial crisis).
- **Q-Q Plots of Coverage Error Gaps (Figures 4.6 & D.4):** Empirical quantiles of $|\text{LocalCov}_t - (1 - \alpha)|$ for DtACI plotted against an idealized i.i.d. $\operatorname{Bernoulli}(\alpha)$ sequence. Points lie tightly along the $y = x$ identity line (ranging from $0.00$ to $0.03$), showing DtACI achieves near-optimal local coverage tightness.

---

## Experiment 3: Evaluating Reactivity / Conditional Coverage of DtACI (Section 4.2.2 & Appendix B)

Verifies that DtACI actually learns $\alpha_t^*$ rather than merely oscillating reactively between over- and under-coverage (Figures 4.7 & D.5).

### 1. Dataset / Required Input
- Same 4 stock return series $\{R_t\}$ and $\text{GARCH}(1, 1)$ setup from Experiment 2 (both normalized and unnormalized conformity scores).
- Partition of $[0, 1]$ into $m$ evenly sized sub-intervals $B_1, \dots, B_m$.

### 2. Algorithms
- **DtACI (Algorithm 2)** to produce sequence $\{\bar{\alpha}_t\}$ and $\{\text{err}_t\}$.
- **Block Bootstrap (Algorithm 4):** Resample $M = 100$ bootstrap series using block size $b = 100$ from $\{R_t\}$ to compute $0.025$ and $0.975$ confidence quantiles.

### 3. Desired Output / Performance
- **Empirical Conditional Coverage Plots (Figures 4.7 & D.5):** Plots of $1 - \text{CondCoverage}_i$ (where $\text{CondCoverage}_i = \frac{1}{|\{t : \bar{\alpha}_t \in B_i\}|} \sum_{t:\bar{\alpha}_t \in B_i} \text{err}_t$) across bins of $\bar{\alpha}_t$.
- **Performance:** Conditional coverage stays flat around the target $1 - \alpha = 0.9$ across all bins of $\bar{\alpha}_t$ (with bootstrap error bars covering $0.9$), proving DtACI is calibrated conditionally on $\bar{\alpha}_t$ and not just canceling out past errors.

---

## Experiment 4: Predicting COVID-19 Case Counts (Section 4.2.3)

Evaluates adaptation to pandemic waves across US counties (Figures 4.8 & D.7).

### 1. Dataset / Required Input
- Public DELPHI repository data (Carnegie Mellon) from roughly mid-2020 to early 2022 across US counties (filtered to counties with no missing data):
  - $\text{CO}_{t,i}$: 7-day moving average of confirmed COVID-19 cases in county $i$ at day $t$.
  - $F_{t,i}$: 7-day moving average of Facebook survey responses (proportion of community knowing someone with COVID-19).
- Highlighted evaluation on 4 urban counties: San Francisco (CA), New York (NY), Miami-Dade (FL), and Dallas (TX).

### 2. Algorithms
- **Baseline Forecaster:** 7-day-ahead least-squares linear regression refit at each time step $t$ using the past 14 days ($s = t - 14, \dots, t$) across all available counties:
  $$\widehat{\text{CO}}_{t+7,i} = \hat{\beta}_0^t + \sum_{j=1}^3 \hat{\lambda}_j^t \, \text{CO}_{t-7j,i} + \sum_{j=1}^3 \hat{\kappa}_j^t \, F_{t-7j,i}$$
- **Conformity Score & Quantile Reference:** Scaled residual:
  $$S_{t,i} = \frac{|\widehat{\text{CO}}_{t,i} - \text{CO}_{t,i}|}{|\text{CO}_{t-7,i} - \text{CO}_{t,i}|}$$
  compared against the spatial empirical distribution of scores across all $n_t$ available counties at time $t - 1$:
  $$\frac{1}{n_t} \sum_{i} \delta_{S_{t-1,i}}$$
- **Conformal Methods:** DtACI, AgACI, MVP, and Constant Baseline ($\alpha_t = 0.1$), run separately per county to get county-specific trajectories $\{\alpha_{t,i}\}$.

### 3. Desired Output / Performance
- **Local Coverage Plots (Figure 4.8):** Moving 200-step local coverage average ($\frac{1}{200}\sum_{s=t-99}^{t+100}(1 - \text{err}_s)$) and global coverage (dashed lines).
- **Performance:** DtACI and AgACI stay tightly near $0.90$ local and global coverage throughout multiple COVID-19 waves, whereas MVP and the Constant Baseline exhibit severe coverage drops ($< 0.6$ initially) and large oscillations ($0.7$ to $1.0$).

---

## Experiment 5: Ablation — Fixed vs. Variable $\eta_t$ in DtACI (Section 3.1 & Appendix D)

Compares the fixed heuristic learning rate $\eta$ against an online adaptive learning rate $\eta_t$ (Figures D.1, D.2, and D.3).

### 1. Dataset / Required Input
- The Stock Market Volatility datasets (both normalized and unnormalized scores from Experiment 2) and the COVID-19 dataset (from Experiment 4).

### 2. Algorithms
- **DtACI with Fixed $\eta$:** $\eta \approx 2.72$.
- **DtACI with Variable $\eta_t$:** Updated online over the last $|I| = 500$ steps via:
  $$\eta_t = \sqrt{\frac{\log(2k \cdot 500) + 1}{\sum_{s=t-500}^{t-1} \ell(\beta_s, \alpha_s)^2}}$$

### 3. Desired Output / Performance
- **Local & Global Coverage Comparison (Figures D.1, D.2, D.3):** The blue (fixed $\eta$) and purple (variable $\eta_t$) local coverage trajectories overlap almost completely around $0.90$, confirming that the theoretical uniform approximation $\frac{1}{|I|} \sum \mathbb{E}[\ell(\beta_t, \alpha_t)^2] \approx \frac{(1 - \alpha)^2 \alpha^2}{3}$ is highly accurate in practice.

---

## Work Split

| Role | Assigned Experiments | Figures to Reproduce | Coding Effort | Other Responsibilities |
|---|---|---|---|---|
| Person 1 (Doc Lead) | Shared Core (DtACI, AgACI, MVP) + Exp 1 (Simulated) | Figs 4.1, 4.2, 4.3 (3 figures) | ~20% (Low) | Report writing, theoretical summary, integrating plots |
| Person 2 (Stocks) | Exp 2 (Stocks + Q-Q), Exp 3 (Bootstrap), Exp 5a (Var $\eta$) | Figs 4.4, 4.5, 4.6, 4.7, D.1, D.2, D.4, D.5, D.6 (9 figures) | ~40% (Medium-High) | Documenting Section 4.2.1 & 4.2.2 results |
| Person 3 (COVID-19) | Exp 4 (COVID-19), Exp 5b (Var $\eta$) | Figs 4.8, D.3, D.7 (3 multi-panel figures, heavy data prep) | ~40% (Medium-High) | Documenting Section 4.2.3 results |

### Why This Split

#### Person 1 (Lighter Coding + Documentation Lead): Core Algorithms + Simulated Experiments
- **Coding Tasks:**
  1. Implement the shared `conformal_methods.py` module containing DtACI (Algorithm 2, both fixed $\eta$ and variable $\eta_t$), AgACI (Section 2.4), and MVP (Algorithm 3) with a clean interface: `run_method(betas, alpha=0.1) -> alphas, gammas`.
  2. Reproduce Experiment 1 (Section 4.1): Generate the 3 synthetic $\mu_t$ regimes (Stationary, Smooth, Jump), run 100 trials, and plot Figures 4.1, 4.2, and 4.3.
- **Why it's lighter:** No external datasets to clean, no regression/GARCH models to fit, and fast execution (seconds). Once Person 1 writes the 3 core conformal functions on Day 1, Person 2 and Person 3 can import them directly (or Person 1 can even develop Exp 1 in parallel if everyone implements/uses the shared 80-line algorithm file).
- **Documentation Tasks:** Working on the slide, with results/overview/visualization from 4 other experiments, keep placeholders there (but still make sure to do as complete as possible).

#### Person 2 (Heavy Pipeline 1): Stock Market Volatility Suite (Experiments 2, 3, & 5a)
- **Coding Tasks:**
  1. Load the 4 stock price series (Nvidia, AMD, BlackBerry, Fannie Mae) and plot Figure D.6.
  2. Implement the rolling 1250-day $\text{GARCH}(1, 1)$ forecaster and compute both normalized and unnormalized conformity scores ($\beta_t$).
  3. Experiment 2: Run DtACI, AgACI, MVP, and Constant Baseline to produce the local coverage plots (Figures 4.4 & 4.5) and the Bernoulli Q-Q plots (Figures 4.6 & D.4).
  4. Experiment 3: Run the Block Bootstrap (Algorithm 4) on the stock data to produce the conditional coverage plots (Figures 4.7 & D.5).
  5. Experiment 5a: Run DtACI with variable $\eta_t$ on the stock scores (Figures D.1 & D.2).
- **Why keep all of these together:** Once Person 2 has the GARCH scores cached from Exp 2, generating the plots for Exp 3, Q-Q plots, and Exp 5a is just a few extra plotting functions on the exact same data.

#### Person 3 (Heavy Pipeline 2): COVID-19 Case Count Suite (Experiments 4 & 5b)
- **Coding Tasks:**
  1. Download/load and preprocess the CMU DELPHI COVID-19 case counts ($\text{CO}_{t,i}$) and Facebook survey responses ($F_{t,i}$) across US counties, handling missing data filtering, and plot Figure D.7 (case counts for San Francisco, New York, Miami-Dade, Dallas).
  2. Implement the 14-day sliding window least-squares regression model to predict 7-day-ahead county case counts $\widehat{\text{CO}}_{t+7,i}$.
  3. Compute the county-level scaled conformity scores $S_{t,i}$ and cross-county empirical quantiles $\beta_{t,i}$ at each timestep.
  4. Experiment 4: Run DtACI, AgACI, MVP, and Constant Baseline per county and plot the 200-step local and global coverage for the 4 urban counties (Figure 4.8).
  5. Experiment 5b: Run DtACI with variable $\eta_t$ on the COVID-19 data (Figure D.3).
- **Why this is a full-sized role:** While COVID-19 has fewer figures than the Stock Market section, data wrangling/aligning multi-county panel time series with lag features ($t - 7, t - 14, t - 21$) and cross-sectional quantiles takes more data-engineering effort than loading 4 single-stock CSVs.
