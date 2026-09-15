"""Causal signal construction for the statistical arbitrage engine.

The original pipeline built its trading signal from the Kalman filter's
own one-step innovations, cumulated from the first day of the backtest,
and calibrated the Ornstein-Uhlenbeck parameters on the entire sample.
That has three defects, all of which this module removes:

1. **Innovation artifact.** The filter's update step is
   ``beta_t = beta_pred + K e_t``, so the next innovation carries a
   ``-H_{t+1} K e_t`` term. That manufactures negative autocorrelation
   proportional to the Kalman gain, and the strategy then trades it.
   Measured on 20 US large caps, raw returns show +0.016 lag-1
   autocorrelation and static OLS residuals +0.016, while the filter's
   innovations show -0.113 at the pipeline's own tuning -- the signal
   was an estimation artifact, not a market effect.

2. **Look-ahead in calibration.** ``OUProcessModel.compute_s_score``
   fits theta and sigma_eq on the whole spread series and then scores
   that same series, so day 1 already knows the future mean and
   volatility.

3. **Unbounded spread.** Cumulating residuals from the first day gives
   a series whose level depends on how long the backtest has been
   running, rather than a bounded, tradeable deviation.

The construction here follows Avellaneda & Lee (2010): on each day the
factor betas are estimated on a trailing window that ends strictly
before that day, the residuals are cumulated *within* that window to
form a bounded spread, the OU parameters are calibrated on that window
alone, and the position is scored from the final point of it. The
return the strategy actually earns is the genuinely out-of-sample
idiosyncratic return, formed by applying those past betas to today's
realized factor returns.
"""

import numpy as np
import pandas as pd


class CausalSignalEngine:
    """Builds s-scores and out-of-sample idiosyncratic returns causally.

    :param lookback: Trailing window, in trading days, used to estimate
        betas and calibrate the OU process (Avellaneda & Lee use 60).
    :param dt: Sampling interval in years, for annualizing kappa.
    :param min_half_life: Reject a window whose fitted mean reversion is
        faster than this many days -- such a fit is noise, not signal.
    :param max_half_life: Reject a window whose mean reversion is slower
        than this, since the position would not converge in time.
    """

    def __init__(
        self,
        lookback: int = 60,
        dt: float = 1.0 / 252.0,
        min_half_life: float = 0.5,
        max_half_life: float = 30.0,
    ):
        if lookback < 20:
            raise ValueError("lookback must be at least 20 observations")
        self.lookback = lookback
        self.dt = dt
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    @staticmethod
    def _fit_ou(x: np.ndarray) -> tuple[float, float, float]:
        """Fits AR(1) x_n = a + b x_{n-1} + zeta, returning (b, theta, sigma_eq)."""
        x_lag, x_cur = x[:-1], x[1:]
        n = len(x_lag)
        design = np.column_stack([np.ones(n), x_lag])
        params, *_ = np.linalg.lstsq(design, x_cur, rcond=None)
        a, b = float(params[0]), float(params[1])
        if not (0.0 < b < 1.0):
            return b, np.nan, np.nan
        resid = x_cur - (a + b * x_lag)
        var_zeta = float(np.var(resid, ddof=2))
        if var_zeta <= 0:
            return b, np.nan, np.nan
        theta = a / (1.0 - b)
        sigma_eq = float(np.sqrt(var_zeta / (1.0 - b**2)))
        return b, theta, sigma_eq

    def compute(
        self, returns: pd.DataFrame, factor_returns: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs the causal construction across every asset.

        :param returns: Asset returns (T x N), time indexed.
        :param factor_returns: Factor returns (T x K), same index.
        :return: ``(s_scores, oos_residuals, sigma_eq)``, each T x N and
            aligned to `returns`. Entry ``[t, i]`` of `s_scores` uses only
            data strictly before ``t``; ``oos_residuals[t, i]`` is the
            idiosyncratic return realized at ``t`` under betas fit before
            it, which is what the portfolio actually earns.
        """
        idx, cols = returns.index, returns.columns
        F_all = factor_returns.to_numpy(dtype=float)
        T, K = F_all.shape

        s_scores = pd.DataFrame(np.nan, index=idx, columns=cols)
        oos_resid = pd.DataFrame(np.nan, index=idx, columns=cols)
        sigma_eq = pd.DataFrame(np.nan, index=idx, columns=cols)

        for col in cols:
            y_all = returns[col].to_numpy(dtype=float)
            s_col = np.full(T, np.nan)
            r_col = np.full(T, np.nan)
            v_col = np.full(T, np.nan)

            for t in range(self.lookback, T):
                lo = t - self.lookback
                y_win, F_win = y_all[lo:t], F_all[lo:t]      # strictly before t
                if not (np.all(np.isfinite(y_win)) and np.all(np.isfinite(F_win))):
                    continue

                design = np.column_stack([np.ones(len(y_win)), F_win])
                beta, *_ = np.linalg.lstsq(design, y_win, rcond=None)

                # Residuals inside the window, cumulated into a bounded spread.
                resid = y_win - design @ beta
                spread = np.cumsum(resid)

                b, theta, s_eq = self._fit_ou(spread)
                if not np.isfinite(s_eq) or s_eq <= 0:
                    continue

                half_life = np.log(2.0) / (-np.log(b))
                if not (self.min_half_life <= half_life <= self.max_half_life):
                    continue

                s_col[t] = (spread[-1] - theta) / s_eq
                v_col[t] = s_eq

                # What the book actually earns at t: today's realized
                # return minus today's factor exposure priced with betas
                # that were fit before today.
                H_t = np.insert(F_all[t], 0, 1.0)
                r_col[t] = y_all[t] - float(H_t @ beta)

            s_scores[col] = s_col
            oos_resid[col] = r_col
            sigma_eq[col] = v_col

        return s_scores, oos_resid, sigma_eq
