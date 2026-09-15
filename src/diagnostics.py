import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import cast


class SpreadDiagnostics:

    def __init__(self, significance_level: float = 0.05):
        """Initializes the statistical diagnostics engine.

        :param significance_level: Threshold p-value for stationarity (default:
        0.05)
        """
        self.significance_level = significance_level

    def adf_test(self, series: pd.Series) -> dict:

        """Runs the Augmented Dickey-Fuller (ADF) unit root test on a time series.

        :param series: Cumulative idiosyncratic spread or price series
        :return: Dictionary containing test statistic, p-value, and stationarity
        verdict
        """
        
        clean_series = series.dropna()
        if len(clean_series) < 30:
            return {
                "adf_stat": np.nan,
                "p_value": np.nan,
                "is_stationary": False,
                "critical_values": {},
            }

        # statsmodels 0.16 switches adfuller's default return type to an
        # ADFullerResult object. Pin the plain-tuple form explicitly where
        # the argument exists, and fall back for versions predating it, so
        # the unpacking below stays correct across both.
        try:
            result = adfuller(clean_series, autolag="AIC", result_object=False)
        except TypeError:
            result = adfuller(clean_series, autolag="AIC")
        adf_stat = result[0]
        p_value = result[1]
        critical_values = cast(
            tuple[float, float, int, int, dict[str, float], float], result
        )[4]

        return {
            "adf_stat": float(adf_stat),
            "p_value": float(p_value),
            "is_stationary": bool(p_value < self.significance_level),
            "critical_values": critical_values,
        }

    def compute_autocorrelation(
        self, series: pd.Series, max_lag: int = 5
    ) -> pd.Series:
        
        """Computes autocorrelation coefficients rho_k up to max_lag.

        :param series: Residual or spread series
        :param max_lag: Number of lags to evaluate
        :return: pd.Series of autocorrelation values per lag
        """
        
        clean_series = series.dropna()
        autocorr_values = {}

        for lag in range(1, max_lag + 1):
            # rho_k = Cov(Y_t, Y_{t-k}) / Var(Y_t)
            autocorr_values[f"lag_{lag}"] = clean_series.autocorr(lag=lag)

        return pd.Series(autocorr_values)

    def filter_tradeable_spreads(
        self, cumulative_spreads: pd.DataFrame, daily_residuals: pd.DataFrame
    ) -> pd.DataFrame:
        
        """Screens all assets in the universe, filtering out non-stationary spreads

        and ranking remaining spreads by mean-reversion strength.
        """
        
        diagnostic_records = []

        for ticker in cumulative_spreads.columns:
            spread = cumulative_spreads[ticker]
            residuals = daily_residuals[ticker]

            adf_res = self.adf_test(spread)
            autocorr_res = self.compute_autocorrelation(residuals, max_lag=1)
            lag1_corr = autocorr_res.get("lag_1", np.nan)

            diagnostic_records.append(
                {
                    "ticker": ticker,
                    "adf_stat": adf_res["adf_stat"],
                    "p_value": adf_res["p_value"],
                    "is_stationary": adf_res["is_stationary"],
                    "lag1_autocorr": lag1_corr,
                    "tradeable": bool(
                        adf_res["is_stationary"] and lag1_corr < 0
                    ),
                }
            )

        df_diagnostics = pd.DataFrame(diagnostic_records).set_index("ticker")
        return df_diagnostics
