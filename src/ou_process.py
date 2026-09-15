import numpy as np
import pandas as pd


class OUProcessModel:

    def __init__(self, dt: float = 1.0 / 252.0):
        
        """Initializes the Ornstein-Uhlenbeck calibration engine.

        :param dt: Time-step size (default: 1/252 for daily data)
        """
        
        self.dt = dt

    def fit_spread(self, spread_series: pd.Series) -> dict[str, float]:
        
        """Fits an AR(1) specification to the cumulative spread to extract
        kappa, theta, sigma, and the equilibrium standard deviation.

        Units: `kappa` is annualized (per year, given dt = 1/252) while
        `half_life` is expressed in sampling periods, i.e. trading days.
        """
        
        x = np.asarray(spread_series.to_numpy(dtype=float), dtype=float)
        x_lag = x[:-1]
        x_curr = x[1:]

        # OLS fit: x_curr = a + b * x_lag + eps
        N = len(x_lag)
        X_design = np.column_stack([np.ones(N), x_lag])
        params, residuals, _, _ = np.linalg.lstsq(X_design, x_curr, rcond=None)

        a, b = params[0], params[1]

        # Valid mean-reversion requires 0 < b < 1
        if b <= 0 or b >= 1:
            return {
                "kappa": np.nan,
                "theta": np.nan,
                "sigma": np.nan,
                "sigma_eq": np.nan,
                "half_life": np.nan,
            }

        kappa = -np.log(b) / self.dt
        theta = a / (1.0 - b)
        var_zeta = np.var(x_curr - (a + b * x_lag), ddof=2)
        sigma = np.sqrt(var_zeta * (-2.0 * np.log(b)) / (self.dt * (1.0 - b**2)))
        sigma_eq = np.sqrt(var_zeta / (1.0 - b**2))
        # kappa is annualized (dt = 1/252), so ln(2)/kappa is in YEARS.
        # Report the half-life in sampling periods (trading days) instead,
        # which is ln(2) / -ln(b) -- the same quantity divided by dt.
        half_life = np.log(2.0) / (-np.log(b))

        return {
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "sigma_eq": sigma_eq,
            "half_life": half_life,
        }

    def compute_s_score(self, spread_series: pd.Series) -> pd.Series:
        
        """Computes standardized s-score series: s_t = (x_t - theta) / sigma_eq."""
        
        params = self.fit_spread(spread_series)
        if np.isnan(params["sigma_eq"]) or params["sigma_eq"] == 0:
            return pd.Series(index=spread_series.index, dtype=float)

        s_scores = (spread_series - params["theta"]) / params["sigma_eq"]
        return s_scores
