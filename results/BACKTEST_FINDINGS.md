# Backtest Findings — 2023-01-01 to 2025-01-01

All four notebooks executed end-to-end on real Yahoo Finance data
(20 US large caps, 5 walk-forward PCA factors, 504-day estimation
window re-fit every 21 days, 5 bps transaction costs).

## Headline result, and why it is not usable

The pipeline as written reports:

| Metric | Gross | Net (5 bps) |
|---|---|---|
| Total return | 1106.6% | 976.7% |
| CAGR | 278.0% | 255.7% |
| Annualized vol | 15.1% | 14.9% |
| **Sharpe** | **8.93** | **8.59** |
| Max drawdown | -4.3% | -4.3% |
| Win rate | 75.0% | 74.8% |

A Sharpe near 9 on daily equity stat-arb is not a plausible research
result. It is an artifact. Three separate causes were isolated.

## 1. The mean reversion is manufactured by the Kalman filter

This is the dominant effect. The strategy trades the cumulative sum of
Kalman innovations, betting they mean-revert. They do — but the
reversion is created by the filter's own update step, not by the market.

The update is `beta_t = beta_pred + K * e_t`, so the next innovation
`e_{t+1} = y_{t+1} - H_{t+1} . beta_t` carries a `-H_{t+1} . K . e_t`
term. That induces negative autocorrelation proportional to the Kalman
gain K.

Measured mean lag-1 autocorrelation across the 20 assets:

| Series | Lag-1 autocorr |
|---|---|
| Raw daily returns | +0.0161 |
| Static OLS residuals | +0.0155 |
| KF residuals, Q=1e-8 (gain ~0.000) | +0.0214 |
| KF residuals, Q=1e-6 (gain ~0.001) | +0.0069 |
| KF residuals, Q=1e-5 (gain ~0.010) | -0.0260 |
| KF residuals, Q=1e-4 (gain ~0.091) — **notebook setting** | **-0.1132** |
| KF residuals, Q=1e-3 (gain ~0.500) | -0.2766 |

Real returns and OLS residuals show no negative autocorrelation. As the
gain goes to zero the KF residual converges to the OLS value, confirming
the signal is entirely filter-induced.

Sweeping the same parameter through the full backtest:

| Process noise Q | Kalman gain | Lag-1 autocorr | Sharpe | CAGR | Assets passing ADF |
|---|---|---|---|---|---|
| 1e-8 | 0.000 | +0.0214 | 1.93 | 19.6% | 1 / 20 |
| 1e-6 | 0.001 | +0.0069 | 2.40 | 62.0% | 4 / 20 |
| 1e-5 | 0.010 | -0.0260 | 6.05 | 122.0% | 11 / 20 |
| 1e-4 | 0.091 | -0.1132 | 9.03 | 254.6% | 17 / 20 |
| 1e-3 | 0.500 | -0.2766 | 7.39 | 264.7% | 4 / 20 |

Sharpe tracks a tuning constant, not a market signal. Note the ADF gate
too: at near-zero gain only 1 of 20 spreads passes the stationarity
test; at the notebook's setting 17 of 20 pass. The filter manufactures
the stationarity the gate then "discovers".

## 2. Look-ahead in the OU calibration

`OUProcessModel.compute_s_score` calls `fit_spread` on the entire spread
series, then scores that same series. On day 1 the s-score already knows
theta and sigma_eq of the whole future sample.

## 3. Look-ahead in universe selection

The ADF/autocorrelation gate runs on the full sample, then selects the
14-17 assets that mean-reverted over the whole window — including the
future portion that is then traded.

Removing 2 and 3 while leaving the filter untouched:

| Configuration | Sharpe | CAGR | Max DD |
|---|---|---|---|
| A. Baseline (as written) | 9.03 | 254.6% | -2.6% |
| B. + causal expanding-window OU calibration | 7.81 | 134.9% | -3.1% |
| C. + universe selected in-sample only | 6.72 | 97.2% | -2.7% |

Still implausible, because cause 1 dominates and is untouched by these
fixes.

## What does hold up

**Clustering (notebook 01).** Parametric UMAP + DBSCAN on PCA loadings
recovers sector structure with no sector labels supplied: mega-cap tech
(AAPL, MSFT), internet (GOOGL, AMZN, META), semis (NVDA, TSLA, AMD,
QCOM), banks (BAC, WFC, C, GS), integrated energy (XOM, COP), oil
services (SLB, EOG); INTC, JPM, CVX fall out as noise. This is a real,
reproducible result and is independent of the backtest defect.

**Kalman vs OLS residual quality (notebook 02).** On NVDA, the KF spread
is stationary (ADF -6.51, p = 1.1e-08) while the static OLS spread is
not (ADF -1.85, p = 0.356). The KF does adapt to drifting factor
exposure as claimed. The caveat from cause 1 applies: part of that
stationarity is filter-induced.

**TCN VaR overlay (notebook 04).** The tail-risk model is sound and its
validation is independent of the alpha defect. Out-of-sample over 112
days:

| Quantile | Breaches | Rate | Target | Kupiec LR | p | Accepted |
|---|---|---|---|---|---|---|
| 5% VaR | 7 | 6.25% | 5% | 0.343 | 0.558 | Yes |
| 1% VaR | 0 | 0.00% | 1% | 2.251 | 0.134 | Yes |

Applied as a de-leveraging overlay it cut annualized volatility 14.9% ->
12.7% (-15%) and max drawdown -4.34% -> -3.48% (-20%) while leaving
Sharpe essentially unchanged (8.59 -> 8.49). The overlay does what it
claims; it is riding on an inflated base strategy.

## Remediation (implemented)

`src/causal_signal.py` rebuilds the signal path as Avellaneda & Lee
specify it. For each day, betas are fit on a trailing 60-day window that
ends strictly before that day, residuals are cumulated *within* that
window to form a bounded spread, OU parameters are calibrated on that
window alone, and the traded return is the out-of-sample idiosyncratic
return obtained by applying those past betas to the realized factor
returns of the scored day.

An intermediate attempt, keeping the Kalman filter but applying its
betas with a purge gap, was rejected: it replaces one artifact with
another. Purging flips the mean lag-1 autocorrelation from -0.119 to
+0.098, because stale betas leave a persistent mis-hedge in the residual.

Result of the causal construction:

| Series | Mean lag-1 autocorr |
|---|---|
| Raw daily returns | +0.0161 |
| Kalman innovations (original signal) | -0.1132 |
| Kalman betas applied with a purge gap (rejected) | +0.0980 |
| **Causal OOS residuals (implemented)** | **+0.0112** |

The induced autocorrelation is gone; the traded series now behaves like
the returns it came from.

| Metric | Original | Causal construction |
|---|---|---|
| Sharpe ratio | 8.59 | **0.52** |
| CAGR | 255.7% | 4.1% |
| Annualized volatility | 14.9% | 8.4% |
| Max drawdown | -4.3% | -7.0% |
| Win rate | 74.8% | 45.0% |

## Null test

`src/validation.py` provides `sign_shuffle_null`, which flips the sign of
each traded return and re-runs the backtest. This preserves volatility
and tail shape while destroying any genuine signal-to-outcome link.

Over 200 draws: null Sharpe mean **-0.405** (sd 0.711, 95th percentile
+0.713) against an observed **+0.521**, giving z = +1.30 and a one-sided
empirical **p = 0.110**.

**Conclusion: on this 20-asset universe over 2023-2025, the strategy's
edge is not statistically significant.** The correct reportable outcome
is a null result with a well-specified, look-ahead-free pipeline -- not
the Sharpe of 8.59 the original construction produced.

## Regression guards

`test/test_engine.py` now pins both defects so they cannot return:

* `test_causal_signal_has_no_lookahead` perturbs a future observation
  and asserts every earlier s-score and residual is bit-identical.
* `test_causal_signal_does_not_induce_autocorrelation` asserts the
  construction leaves white-noise input white, within 0.05.
* `test_sign_shuffle_null_rejects_a_zero_edge_strategy` asserts a
  no-edge strategy fails the null.
