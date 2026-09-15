"""Null-hypothesis tests for backtest results.

A backtest number means nothing on its own: the construction itself can
manufacture apparent skill, as this engine's earlier Kalman-innovation
signal did. These tests re-run the same machinery against data whose
tradeable structure has been destroyed, so the strategy's result can be
read against the distribution it would produce by luck alone.
"""

import numpy as np
import pandas as pd


def sign_shuffle_null(
    backtest_fn,
    traded_returns: pd.DataFrame,
    n_draws: int = 200,
    seed: int = 0,
) -> dict:
    """Randomly flips the sign of each traded return and re-runs the backtest.

    Flipping signs preserves every return's magnitude, and so the
    volatility and fat tails, while destroying any genuine relationship
    between the signal and what follows it. A strategy with real edge
    scores far above this distribution; one trading an estimation
    artifact sits inside it.

    :param backtest_fn: Callable taking a returns frame and giving back
        the realized Sharpe ratio.
    :param traded_returns: The per-asset returns the strategy earns.
    :param n_draws: Number of null draws.
    :param seed: Seed, so a reported p-value can be reproduced.
    :return: Summary with the null distribution's moments, the observed
        Sharpe, its z-score, and a one-sided empirical p-value.
    """
    rng = np.random.default_rng(seed)
    observed = float(backtest_fn(traded_returns))

    draws = np.empty(n_draws)
    for i in range(n_draws):
        flips = rng.choice([-1.0, 1.0], size=traded_returns.shape)
        draws[i] = backtest_fn(
            traded_returns * pd.DataFrame(
                flips, index=traded_returns.index, columns=traded_returns.columns
            )
        )

    mean, sd = float(draws.mean()), float(draws.std())
    return {
        "observed_sharpe": observed,
        "null_mean": mean,
        "null_sd": sd,
        "null_p95": float(np.percentile(draws, 95)),
        "z_score": (observed - mean) / sd if sd > 0 else np.nan,
        "p_value": float((draws >= observed).mean()),
        "significant": bool((draws >= observed).mean() < 0.05),
        "n_draws": n_draws,
    }


def residual_autocorrelation_check(
    traded_returns: pd.DataFrame, raw_returns: pd.DataFrame, tolerance: float = 0.05
) -> dict:
    """Checks that residual construction has not invented autocorrelation.

    Daily equity returns are close to serially uncorrelated. A residual
    series showing markedly more lag-1 autocorrelation than the raw
    returns it came from has had that structure introduced by the
    estimator -- which is exactly how the Kalman innovations in this
    engine produced a Sharpe near 9 out of nothing.

    :return: Both mean autocorrelations, their gap, and whether the gap
        stays inside `tolerance`.
    """
    traded = float(traded_returns.apply(lambda s: s.dropna().autocorr(1)).mean())
    raw = float(raw_returns.apply(lambda s: s.dropna().autocorr(1)).mean())
    return {
        "traded_lag1_autocorr": traded,
        "raw_lag1_autocorr": raw,
        "excess": traded - raw,
        "within_tolerance": bool(abs(traded - raw) < tolerance),
    }
