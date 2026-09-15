# Machine-Learning-Enhanced Statistical Arbitrage

This system modernizes the classic Avellaneda & Lee (2010) framework by replacing static regressions with dynamic Kalman Filters, introducing non-linear manifold clustering via Parametric UMAP and DBSCAN, and applying a causal Temporal Convolutional Network (TCN) to dynamically control tail-risk exposure.

---
## Introduction.

![Statistical arbitrage pipeline](stat_arb_pipeline_detailed_topdown.svg)

Arbitrage pricing theory framework dictates that the return of an asset is driven by an arbitrary number of macroeconomics factors, and the idiosyncratic error is a martingale difference sequence. The core objective of this project is to exploit the arbitrage opportunities from these mispricing anomalies. We extract the systematic risk factors by PCA decomposition, in potentially high dimensional PCA space. So, we performed a parametric UMAP algorithm to compress the factor exposures into a 2D nonlinear-manifold. To extract the clean mispricing without lookback bias, we continuously track the time-varying betas using a recursive Kalman Filter updating model on a 2-year rolling window, which is more appropriate for noise reduction. And for many other reasons a longer window in comparison to the original paper works in our favor, for example, the stability in ADF stationary test and correlation, the noise of sample covariance matrix due to the dimensionality, the estimation noise of the UMAP and DBSCAN combination, and training efficacy of ML components. Then we applied the mean-reverting processes to convert spread deviations into tradable normalized scores and signals. And to account for the portfolio risk factors, we trained a causal dilated TCN model directly from pinball loss for quantiles to forecast the 1% and 5% VaR, validated against Kupiec POF.

---


## Theoretical Framework & Architecture. 

### 1. Market Modeling & Dimensionality Reduction
PCA Factor Extraction: Compresses the multi-asset variance of a highly correlated equities universe into orthogonal systematic risk factors (eigenvectors) via eigendecomposition. The data matrix $X$ is standardized, and the sample covariance matrix $\Sigma$ is computed to map asset relationships:

$$\Sigma = \frac{1}{n-1} X^T X$$

Eigendecomposition: The covariance matrix is decomposed to extract its eigenvalues $\lambda_i$ and eigenvectors $v_i$:

$$\Sigma v_i = \lambda_i v_i$$

Parametric UMAP & DBSCAN Clustering: Projects linear PCA factor loadings into a dense, non-linear latent manifold using a neural network encoder. The UMAP algorithm optimizes the fuzzy set cross-entropy loss to contract cohesive assets into dense topological clusters:

$$\mathcal{L}_{\text{UMAP}} = \sum_{e \in E} \left[ w_h(e) \log\left(\frac{w_h(e)}{w_l(e)}\right) + (1 - w_h(e)) \log\left(\frac{1 - w_h(e)}{1 - w_l(e)}\right) \right]$$

Dynamic Selection: DBSCAN evaluates spatial distances on these UMAP embeddings to isolate highly cohesive, cointegrated asset clusters while filtering out erratic assets as noise.


### 2. Dynamic Residual Extraction
Multi-Factor Kalman Filter: Replaces traditional Ordinary Least Squares (OLS) regression to prevent stale hedge ratios. The State-Space Model continuously updates unobserved factor betas as new daily observations arrive.

Unobserved State Equation: Models the dynamic hedge ratio $\beta_t$ as a random walk, where $w_t$ represents the process noise:

$$\beta_t = \beta_{t-1} + w_t$$

Observation Equation: Models the actual market data, where $y_t$ is the real asset return, $x_t$ represents the PCA factor returns, and $v_t$ is the measurement noise:

$$y_t = \beta_t x_t + v_t$$

Pure Idiosyncratic Spreads: The innovation error ($v_t$) of the Kalman Filter isolates the pure, adaptive idiosyncratic residual spread of each asset, cleanly stripped of broad market influence.


### 3. Mean-Reversion & Signal Generation
Before spread modeling, the cumulative idiosyncratic residuals extracted from the Kalman Filter undergo two automated validation filters to prune non-convergent assets:

* **Augmented Dickey-Fuller (ADF) Test:** Tests the null hypothesis ($H_0$) that the spread contains a unit root. Spreads must achieve $p < 0.05$ to reject non-stationarity and confirm mean-reverting bounds.
* **Lag-1 Autocorrelation Screening:** Calculates the first-order serial correlation $\rho_1 = \frac{\text{Cov}(e_t, e_{t-1})}{\text{Var}(e_t)}$. The spread must exhibit negative autocorrelation ($\rho_1 < 0$) to verify that daily shocks experience a mean-reverting pull rather than trending momentum.

Spreads failing either criterion are discarded before parameter estimation, preventing the model from fitting invalid parameters to random walks.

Once the systematic PCA factors are hedged out, the remaining idiosyncratic residual spread is modeled using the Ornstein-Uhlenbeck (OU) process. This stochastic differential equation (SDE) is governed by two competing forces: a deterministic "drift" that pulls the asset back to its historical mean, and a stochastic "diffusion" representing random market noise.

The continuous-time SDE is defined as:

$$dx_t = \kappa(\theta - x_t)dt + \sigma dW_t$$

Where:
* **$\theta$ (Long-Term Equilibrium Mean):** The historical gravitational center of the trade. Because we are trading hedged residuals derived from zero-mean Kalman innovations, $\theta$ typically centers around zero.
* **$\kappa$ (Mean Reversion Speed):** The deterministic pull or "rubber band" effect. A high $\kappa$ indicates the spread violently snaps back to $\theta$, while a low $\kappa$ indicates sluggish convergence.
* **$\sigma dW_t$ (Stochastic Diffusion):** The unpredictable market noise that continuously perturbs the spread away from equilibrium.

### Discrete-Time AR(1) Calibration
Because our market data is sampled at discrete daily intervals ($\Delta t = 1/252$) rather than continuously, the SDE is mathematically mapped to an exact Autoregressive AR(1) process for calibration:

$$x_n = a + b x_{n-1} + \zeta_n$$

By fitting the cumulative daily residuals via Ordinary Least Squares (OLS) to this AR(1) structure, we extract the continuous-time parameters:
* **Mean-Reversion Speed:** $\kappa = -\frac{\ln(b)}{\Delta t}$

* **Equilibrium Mean:** $\theta = \frac{a}{1 - b}$

* **Equilibrium Volatility:** $\sigma_{\text{eq}} = \sqrt{\frac{\text{Var}(\zeta)}{1 - b^2}}$

### Automated Trade Execution
These calibrated parameters create a rigorous mathematical boundary for execution. We transform the raw spread into a dimensionless $s$-score:

$$s_t = \frac{x_t - \theta}{\sigma_{\text{eq}}}$$

When the stochastic diffusion pushes the $s$-score significantly far from zero, the deterministic drift term $\kappa(\theta - x_t)dt$ mathematically overpowers the random noise. This triggers automated entry signals, betting on high-probability convergence back to the historical mean.

* **Open Long Spread ($+1$):** $s_t < -s_{\text{open}}$ (Spread is oversold; buy asset, short factor basket).
* **Open Short Spread ($-1$):** $s_t > +s_{\text{open}}$ (Spread is overbought; short asset, buy factor basket).
* **Close Position ($0$):** $\vert{}s_t\vert{} \le s_{\text{close}}$ (Spread has reverted to equilibrium $\theta$).

### Dynamic Risk Parity Allocation

Capital is allocated across active cluster members using inverse equilibrium volatility weighting:

$$w_{i, t} \propto \frac{\text{Signal}_{i, t}}{\sigma_{\text{eq}, i}}$$

Portfolio weights are normalized row-wise to enforce target gross leverage limits ($\sum |w_{i,t}| \le L_{\text{max}}$). This ensures quieter spreads receive proportionally larger capital allocations while volatile spreads are scaled down, equalizing tail-risk contributions across the strategy.


### 4. Deep Learning Risk Overlay (TCN)
Temporal Convolutional Network: A PyTorch architecture processing sequential 3D tensors (combining portfolio PnL, squared variance proxies, and macro factors) to forecast next-day Value at Risk (VaR).

1D Causal Dilated Convolutions: Ensures the filter output at time $t$ is strictly derived from inputs at time $t$ and earlier, explicitly preventing future data leakage. For a 1D sequence $\mathbf{x} \in \mathbb{R}^T$ and a convolutional filter $f$ with dilation factor $d$, the operation expands the receptive field efficiently:

$$y_t = (\mathbf{x} *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot \mathbf{x}_{t - d \cdot i}$$

Multi-Quantile Pinball Loss: Optimizes directly for the 1% and 5% left-tail risk percentiles ($q \in \{0.01, 0.05\}$). The loss asymmetrically penalizes overestimation and underestimation to pinpoint the conditional quantile:

$$\mathcal{L}_q(y, \hat{y}_q) = \max\left(q(y - \hat{y}_q), (q - 1)(y - \hat{y}_q)\right) = (y - \hat{y}_q)\left(q - \mathbb{I}_{\{y < \hat{y}_q\}}\right)$$

Total Batch Loss: The aggregated loss across a batch of $N$ samples and target quantiles $Q$ is computed as:

$$\mathcal{L}_{\text{total}} = \sum_{q \in Q} \frac{1}{N} \sum_{i=1}^N \mathcal{L}_q\left(y_i, \hat{y}_{q, i}\right)$$

Statistical Validation: The network's unconditional coverage is formally evaluated against an EGARCH baseline using the Kupiec Proportion of Failures (POF) Likelihood Ratio test. Based on empirical failures $x$ over $N$ observations against target risk level $\alpha$, the statistic follows a $\chi^2(1)$ distribution:


$$\text{LR}_{\text{POF}} = -2 \left[ x \ln(\alpha) + (N - x) \ln(1 - \alpha) - x \ln\left(\frac{x}{N}\right) - (N - x) \ln\left(1 - \frac{x}{N}\right) \right]$$



---

## Repository Structure. 
```text
├── data/                                   # Local data cache (gitignored except .gitkeep)
│   └── .gitkeep
├── notebooks/                              # 4-Stage Execution Narrative
│   ├── 01_pca_decomposition.ipynb          # PCA, Parametric UMAP, and DBSCAN Clustering
│   ├── 02_KF_residuals_verification.ipynb  # Kalman Filter tracking vs OLS + OU Calibration
│   ├── 03_portfolio_backtest.ipynb         # Multi-Asset Execution & Transaction Cost Friction
│   └── 04_tcn_var_risk_overlay.ipynb       # PyTorch TCN VaR Forecasting & Kupiec POF Testing
├── results/
│   └── BACKTEST_FINDINGS.md                # Measured results and the bias analysis behind them
├── src/                                    # Core Modular Engine
│   ├── backtest.py                         # Portfolio aggregation, inverse-vol weighting, tearsheets
│   ├── causal_signal.py                    # Look-ahead-free s-score and OOS residual construction
│   ├── clustering.py                       # Parametric UMAP + DBSCAN density clustering
│   ├── data_loader.py                      # Request-keyed Parquet caching and yfinance ingestion
│   ├── diagnostics.py                      # ADF stationarity and autocorrelation screening
│   ├── ou_process.py                       # Ornstein-Uhlenbeck SDE modeling and s-score generation
│   ├── pca_model.py                        # Eigendecomposition and variance mapping
│   ├── residuals_KF.py                     # Multi-Factor Kalman Filter State-Space model
│   ├── residuals_OLS.py                    # Static OLS residual baseline
│   ├── rolling_engine.py                   # Walk-forward PCA re-estimation
│   ├── tcn_var.py                          # PyTorch TCN architecture and Pinball Loss
│   └── validation.py                       # Sign-shuffle null and autocorrelation sanity checks
├── test/                                   # Pytest unit testing suite
│   └── test_engine.py
├── scripts_generate_cache.py               # Populate data/ from yfinance for offline runs
├── .gitignore                              # Excludes data/, .pt weights, Jupyter checkpoints
└── requirements.txt                        # Python dependencies
```

---
## Results & Methodological Caveat.

The pipeline as originally written reported a Sharpe ratio of **8.59**
net of costs over 2023-2025. That number is an artifact, not a result,
and `results/BACKTEST_FINDINGS.md` documents the analysis in full.

The dominant cause: the strategy traded the cumulative sum of the Kalman
filter's own one-step innovations. The filter's update step,
`beta_t = beta_pred + K e_t`, leaves a `-H_{t+1} K e_t` term in the next
innovation, manufacturing negative autocorrelation in proportion to the
Kalman gain. Raw daily returns and static OLS residuals both show lag-1
autocorrelation near +0.016; the innovations showed -0.113. Sweeping the
process-noise parameter across five orders of magnitude moved the Sharpe
ratio from 1.93 to 9.03 -- performance tracked a tuning constant rather
than any market signal. Two further defects, full-sample OU calibration
and full-sample universe selection, contributed additional look-ahead.

`src/causal_signal.py` rebuilds the signal path along the lines
Avellaneda & Lee specify: betas fit on a trailing window ending strictly
before the scored day, residuals cumulated within that window, OU
calibrated on that window alone, and the traded return formed
out-of-sample. Under that construction the induced autocorrelation
disappears (+0.011, in line with the raw returns) and the honest result
over the same period is:

| Metric | Original | Causal construction |
|---|---|---|
| Sharpe ratio | 8.59 | **0.52** |
| CAGR | 255.7% | 4.1% |
| Annualized volatility | 14.9% | 8.4% |
| Max drawdown | -4.3% | -7.0% |

A 200-draw sign-shuffle null gives a mean Sharpe of -0.41 (sd 0.71)
against the observed 0.52, i.e. z = +1.30 and an empirical p-value of
**0.110**. On this 20-asset universe over this period, the strategy's
edge is *not* statistically significant, and the repository reports it
that way rather than quoting the artifact.

The components that stand independently of this defect are the
UMAP/DBSCAN sector recovery (notebook 01) and the TCN VaR overlay, whose
forecasts pass Kupiec unconditional-coverage tests out-of-sample
(5% VaR: 6.25% realized breach rate, p = 0.558).

---
## Sources.
---
## Sources.  
* **Avellaneda, M., & Lee, J.-H. (2010).** *Statistical arbitrage in the US equities market.* **Quantitative Finance**.
* **McInnes, L., Healy, J., & Melville, J. (2018).** *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* **arXiv:1802.03426**.
* **Sainburg, T., McInnes, L., & Gentner, T. Q. (2021).** *Parametric UMAP: learning embeddings with deep neural networks for representation and semi-supervised learning.* **Neural Computation**, (arXiv:2009.12981, 2020).
* **Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).** *A density-based algorithm for discovering clusters in large spatial databases with noise.* **Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)**, 226–231.
* **Kupiec, P. H. (1995).** *Techniques for verifying the accuracy of risk measurement models.* **The Journal of Derivatives**, 3(2), 73–84.
* **Koenker, R., & Bassett, G. (1978).** *Regression quantiles.* **Econometrica**, 46(1), 33–50.
* **Bai, S., Kolter, J. Z., & Koltun, V. (2018).** *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.* **arXiv:1803.01271**.
* **Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017).** *Temporal convolutional networks for action segmentation and detection.* **IEEE CVPR**, 156–165 (arXiv:1611.05267, 2016).
  * *Project Role:* The original 2016/2017 formulation of hierarchical causal convolutions and temporal receptive fields for streaming sequential signals.
* **Montana, G., Triantafyllopoulos, K., & Tsagaris, T. (2024).** *Dynamic modeling of mean-reverting for statistical arbitrage* **Statistical Finance**, (arXiv:0808.1710, 2009)
