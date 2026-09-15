import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "QCOM",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "XOM", "CVX", "COP", "SLB",
]


# ==============================================================================
# 1. Architectural Layers & Causal Convolutions
# ==============================================================================
class Chomp1d(nn.Module):
    """Safely slices trailing right-side padding to enforce strict temporal causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, : -self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    """Residual dilated causal convolutional block with projection alignment."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # 1x1 projection for residual dimension alignment
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCNVaRForecaster(nn.Module):
    """Multi-channel TCN mapping sequential portfolio & macro features to tail quantiles."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int] = [16, 32, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        quantiles: list[float] = [0.01, 0.05],
    ):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.quantiles = quantiles
        self.fc = nn.Linear(num_channels[-1], len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, num_features, seq_len)
        output = self.tcn(x)
        # Extract representation at final time step t
        last_step = output[:, :, -1]
        return self.fc(last_step)


# ==============================================================================
# 2. Vectorized Quantile (Pinball) Loss
# ==============================================================================
class PinballLoss(nn.Module):
    """Vectorized multi-quantile pinball loss function."""

    def __init__(self, quantiles: list[float] = [0.01, 0.05]):
        super().__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        # y_pred: (batch, n_q), y_true: (batch, 1)
        q = self.quantiles.unsqueeze(0)  # (1, n_q)
        error = y_true - y_pred  # (batch, n_q)
        loss = torch.max(q * error, (q - 1.0) * error)
        return torch.mean(loss)


# ==============================================================================
# 3. Portfolio-Level Feature Engineering for the Risk Overlay
# ==============================================================================
def prepare_stat_arb_features(
    positions: pd.DataFrame,
    residuals: pd.DataFrame,
    factor_returns: pd.DataFrame,
    sigma_eq_dict: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the multi-channel feature tensor feeding the TCN VaR overlay:
    [R_p, R_p^2, F_1, ..., F_K], day-aligned with the realized portfolio
    return R_p, which doubles as the TCN's forecasting target.

    Portfolio return uses the same T+1 causal convention as
    PortfolioBacktester.run_backtest: positions decided at t-1 are applied
    to idiosyncratic residual returns realized at t, so no future
    information leaks into the feature the TCN is trained to predict.

    :param positions: Portfolio weights (T x N) -- e.g. the unhedged
        base_weights from PortfolioBacktester.compute_portfolio_weights.
    :param residuals: Idiosyncratic daily residual returns (T x N),
        same asset universe and date index as `positions`.
    :param factor_returns: Systematic PCA factor returns (T x K), same
        date index.
    :param sigma_eq_dict: Per-asset equilibrium volatility from OU
        calibration. Not used in this feature set directly -- positions
        already come in inverse-equilibrium-vol weighted (see
        PortfolioBacktester.compute_portfolio_weights), so the scaling
        it captures is already embedded in `positions`. Kept as a
        parameter so callers can pass the same dict used upstream and
        so future feature variants (e.g. per-asset risk contribution)
        can use it without changing the call signature.
    :return: (feature_matrix of shape (T, 2 + K), portfolio_returns of shape (T,))
    """
    common_cols = [c for c in positions.columns if c in residuals.columns]
    positions = positions[common_cols]
    residuals = residuals[common_cols]

    # T+1 causal execution: yesterday's decided position times today's
    # realized residual return.
    lagged_positions = positions.shift(1).fillna(0.0)
    portfolio_returns = (lagged_positions * residuals).sum(axis=1)

    factor_returns = factor_returns.loc[portfolio_returns.index]

    r_p = portfolio_returns.to_numpy(dtype=float)
    r_p_squared = r_p**2  # realized-variance proxy channel

    feature_matrix = np.column_stack(
        [r_p, r_p_squared, factor_returns.to_numpy(dtype=float)]
    )

    return feature_matrix, r_p


# ==============================================================================
# 4. Generalized Portfolio Dataset Pipeline
# ==============================================================================
class PortfolioSequenceDataset(Dataset):
    """Constructs rolling temporal 3D tensors: [N_samples, n_features, seq_len]."""

    def __init__(
        self, features: np.ndarray, targets: np.ndarray, seq_len: int = 30
    ):
        self.seq_len = seq_len
        X_list, y_list = [], []

        for i in range(len(features) - seq_len):
            X_list.append(features[i : i + seq_len])
            y_list.append(targets[i + seq_len])

        # Transpose from (N, seq_len, features) to (N, features, seq_len) for Conv1d
        self.X = torch.tensor(np.array(X_list), dtype=torch.float32).transpose(
            1, 2
        )
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(
            1
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ==============================================================================
# 5. Statistical Validation: Kupiec POF Test
# ==============================================================================
def kupiec_pof_test(
    actual_returns: np.ndarray,
    predicted_var: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Evaluates the unconditional coverage of VaR forecasts via the Kupiec POF Likelihood Ratio test."""
    N = len(actual_returns)
    failures = np.sum(actual_returns < predicted_var)
    failure_rate = failures / N if N > 0 else 0.0

    if failures == 0 or failures == N:
        return {
            "alpha": alpha,
            "failures": int(failures),
            "failure_rate": float(failure_rate),
            "lr_stat": 0.0,
            "p_value": 1.0,
            "model_accepted": True,
        }

    # Likelihood Ratio: -2 * ln( ( (1-p)^(N-x) * p^x ) / ( (1 - x/N)^(N-x) * (x/N)^x ) )
    lr_stat = -2.0 * (
        (N - failures) * np.log(1.0 - alpha)
        + failures * np.log(alpha)
        - (N - failures) * np.log(1.0 - failure_rate)
        - failures * np.log(failure_rate)
    )
    p_val = 1.0 - chi2.cdf(lr_stat, df=1)

    return {
        "alpha": alpha,
        "failures": int(failures),
        "failure_rate": float(failure_rate),
        "lr_stat": float(lr_stat),
        "p_value": float(p_val),
        "model_accepted": bool(p_val > 0.05),
    }


# ==============================================================================
# 6. Dynamic Risk Overlay: VaR-Conditioned Leverage Scaling
# ==============================================================================
def apply_var_risk_overlay(
    positions: pd.DataFrame,
    var_forecasts: np.ndarray,
    target_risk_limit: float = 0.015,
    var_index_offset: int = 0,
    var_quantile_index: int = 0,
) -> pd.DataFrame:
    """De-levers the portfolio ahead of forecasted tail risk.

    For every day with a TCN VaR forecast, scales that day's position
    down (never up) so the expected tail-quantile loss stays within
    `target_risk_limit`. Days before the first forecast is available
    (the initial `var_index_offset` burn-in, i.e. `seq_len`, during
    which the TCN has no sequence history yet) pass through at full
    size.

    :param positions: Full-sample portfolio weights (T x N), the same
        index used to build the TCN's feature/target arrays via
        `prepare_stat_arb_features`.
    :param var_forecasts: TCN quantile forecasts, shape
        (T - var_index_offset, n_quantiles).
    :param target_risk_limit: Maximum tolerable magnitude of the tail
        VaR (e.g. 0.015 caps expected tail-quantile daily loss at 1.5%).
    :param var_index_offset: Row offset between `positions` and
        `var_forecasts` (the burn-in length, `seq_len`, consumed by
        PortfolioSequenceDataset before its first prediction).
    :param var_quantile_index: Which column of `var_forecasts` drives
        sizing -- defaults to column 0, the most extreme/lowest
        quantile the model was trained on (e.g. the 1% VaR).
    :return: De-levered weights, same shape and index as `positions`.
    """
    hedged = positions.astype(float)
    tail_var = np.asarray(var_forecasts)[:, var_quantile_index]

    # VaR forecasts are a loss quantile and typically negative; compare
    # magnitudes so the risk-limit check is sign-safe.
    var_magnitude = np.abs(tail_var)

    # Scale down (never up) whenever forecasted tail loss exceeds budget.
    scale = np.where(
        var_magnitude > target_risk_limit,
        target_risk_limit / np.maximum(var_magnitude, 1e-12),
        1.0,
    )
    scale = np.clip(scale, 0.0, 1.0)

    n_forecasts = len(scale)
    scale_series = pd.Series(1.0, index=positions.index)
    scale_series.iloc[var_index_offset : var_index_offset + n_forecasts] = scale

    return hedged.mul(scale_series, axis=0)


# ==============================================================================
# 7. End-to-End Pipeline Runner
# ==============================================================================
def train_and_evaluate_tcn(
    returns: pd.DataFrame | None = None,
    tickers: list | None = None,
    start_date: str = "2019-01-01",
    end_date: str = "2025-01-01",
    backtest_start: str = "2023-01-01",
    n_components: int = 5,
    pca_window: int = 504,
    pca_rebalance_freq: int = 21,
    seq_len: int = 30,
    epochs: int = 35,
    quantiles: list | None = None,
    target_risk_limit: float = 0.015,
    verbose: bool = True,
) -> dict:
    """Assembles and runs the full strategy pipeline end-to-end: rolling PCA,
    UMAP/DBSCAN clustering, Kalman residual extraction, stationarity
    diagnostics, OU calibration, an unhedged backtest, TCN VaR training,
    and the dynamic risk overlay -- then returns the trained model plus
    both backtests for comparison.

    Pass `returns` directly (asset returns, T x N, time-indexed) to run
    against injected/offline data instead of hitting the network via
    yfinance -- useful for tests and reproducibility.

    :return: dict with keys: model, base_results, hedged_results,
        base_metrics, hedged_metrics, kupiec_1pct, kupiec_5pct,
        active_universe, clusters.
    """
    # Local imports to avoid import-time cycles with modules that don't
    # need torch just to be imported elsewhere in the package.
    from src.backtest import PortfolioBacktester
    from src.clustering import FactorClusterer
    from src.diagnostics import SpreadDiagnostics
    from src.ou_process import OUProcessModel
    from src.residuals_KF import extract_idiosyncratic_residuals
    from src.rolling_engine import RollingPCAEngine

    quantiles = quantiles if quantiles is not None else [0.01, 0.05]

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Data
    if returns is None:
        from src.data_loader import fetch_equity_returns

        returns = fetch_equity_returns(
            tickers or DEFAULT_TICKERS, start_date=start_date, end_date=end_date
        )
    _log(f"Returns matrix shape: {returns.shape}")

    # 2. Walk-forward PCA factor extraction
    rolling_pca = RollingPCAEngine(
        n_components=n_components, window=pca_window, rebalance_freq=pca_rebalance_freq
    )
    factor_returns = rolling_pca.fit_transform(returns)

    returns = returns.loc[backtest_start:]
    factor_returns = factor_returns.loc[backtest_start:]
    if len(returns) == 0:
        raise ValueError(
            f"No data on/after backtest_start={backtest_start!r}; check the "
            "PCA burn-in window leaves a non-empty backtest period."
        )

    # 3. Clustering (latest loadings snapshot as of the backtest start)
    clusterer = FactorClusterer(n_components=2, eps=0.4, min_samples=2)
    clusterer.fit(rolling_pca.loadings_as_of(returns.index[0]))
    clusters = clusterer.get_clusters()
    noise_assets = set(clusters.get(-1, []))

    # 4. Kalman-filtered idiosyncratic residuals
    residuals, cumulative_spreads, _ = extract_idiosyncratic_residuals(
        returns, factor_returns, process_noise=1e-4, measurement_noise=1e-3
    )
    burn_in = 30
    clean_residuals = residuals.iloc[burn_in:]
    clean_spreads = cumulative_spreads.iloc[burn_in:]
    clean_factors = factor_returns.loc[clean_spreads.index]

    # 5. Stationarity/gatekeeper diagnostics
    diag = SpreadDiagnostics(significance_level=0.05)
    diagnostic_summary = diag.filter_tradeable_spreads(clean_spreads, clean_residuals)
    tradeable_mask = diagnostic_summary["tradeable"] & (
        ~diagnostic_summary.index.isin(noise_assets)
    )
    active_universe = diagnostic_summary[tradeable_mask].index.tolist()
    _log(f"Active tradeable universe: {len(active_universe)} assets")
    if len(active_universe) == 0:
        raise ValueError(
            "No tradeable assets survived the ADF/autocorrelation screen; "
            "cannot build s-scores or train the TCN on an empty universe."
        )

    # 6. OU calibration & s-scores
    ou_engine = OUProcessModel()
    s_scores = pd.DataFrame(index=clean_spreads.index, columns=active_universe)
    sigma_eq_dict = {}
    for ticker in active_universe:
        params = ou_engine.fit_spread(clean_spreads[ticker])
        if not np.isnan(params["sigma_eq"]):
            sigma_eq_dict[ticker] = params["sigma_eq"]
            s_scores[ticker] = ou_engine.compute_s_score(clean_spreads[ticker])
    s_scores = s_scores.dropna(how="all", axis=1).astype(float)

    # 7. Unhedged baseline backtest
    backtester = PortfolioBacktester(
        s_open=1.25, s_close=0.5, transaction_cost_bps=5.0, max_gross_leverage=1.0
    )
    signals = backtester.generate_signals(s_scores)
    base_weights = backtester.compute_portfolio_weights(signals, sigma_eq_dict)
    base_results = backtester.run_backtest(
        base_weights, clean_residuals[s_scores.columns]
    )

    # 8. TCN feature tensor
    feature_matrix, portfolio_returns = prepare_stat_arb_features(
        positions=base_weights,
        residuals=clean_residuals[s_scores.columns],
        factor_returns=clean_factors,
        sigma_eq_dict=sigma_eq_dict,
    )

    if len(feature_matrix) <= seq_len * 2:
        raise ValueError(
            f"Only {len(feature_matrix)} feature rows available for "
            f"seq_len={seq_len}; need enough history for a meaningful "
            "train/test split. Widen the date range or shrink seq_len."
        )

    split_idx = int(0.70 * len(feature_matrix))
    train_dataset = PortfolioSequenceDataset(
        feature_matrix[:split_idx], portfolio_returns[:split_idx], seq_len=seq_len
    )
    test_dataset = PortfolioSequenceDataset(
        feature_matrix[split_idx:], portfolio_returns[split_idx:], seq_len=seq_len
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 9. TCN training
    n_features = feature_matrix.shape[1]
    model = TCNVaRForecaster(
        num_inputs=n_features,
        num_channels=[16, 32, 64],
        kernel_size=3,
        dropout=0.15,
        quantiles=quantiles,
    )
    criterion = PinballLoss(quantiles=quantiles)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    model.train()
    loss_history = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(train_dataset)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            _log(f"Epoch {epoch:02d}/{epochs} | Pinball Loss: {avg_loss:.6f}")

    # 10. Out-of-sample evaluation
    model.eval()
    test_preds, test_actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            test_preds.append(model(batch_x))
            test_actuals.append(batch_y)

    var_preds = torch.cat(test_preds, dim=0).numpy()
    actual_pnl = torch.cat(test_actuals, dim=0).numpy().flatten()

    kupiec_1pct = kupiec_pof_test(actual_pnl, var_preds[:, 0], alpha=quantiles[0])
    kupiec_5pct = kupiec_pof_test(actual_pnl, var_preds[:, 1], alpha=quantiles[1])
    _log(f"Kupiec POF ({quantiles[0]:.0%} VaR): {kupiec_1pct}")
    _log(f"Kupiec POF ({quantiles[1]:.0%} VaR): {kupiec_5pct}")

    # 11. Full-sample risk overlay & hedged backtest
    full_dataset = PortfolioSequenceDataset(
        feature_matrix, portfolio_returns, seq_len=seq_len
    )
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    full_preds = []
    with torch.no_grad():
        for batch_x, _ in full_loader:
            full_preds.append(model(batch_x))
    full_var_forecasts = torch.cat(full_preds, dim=0).numpy()

    hedged_weights = apply_var_risk_overlay(
        positions=base_weights,
        var_forecasts=full_var_forecasts,
        target_risk_limit=target_risk_limit,
        var_index_offset=seq_len,
    )
    hedged_results = backtester.run_backtest(
        hedged_weights, clean_residuals[s_scores.columns]
    )

    base_metrics = backtester.calculate_metrics(base_results["net_returns"])
    hedged_metrics = backtester.calculate_metrics(hedged_results["net_returns"])
    _log(f"Base Sharpe: {base_metrics.get('Sharpe Ratio', float('nan')):.3f} | "
         f"Hedged Sharpe: {hedged_metrics.get('Sharpe Ratio', float('nan')):.3f}")

    return {
        "model": model,
        "base_results": base_results,
        "hedged_results": hedged_results,
        "base_metrics": base_metrics,
        "hedged_metrics": hedged_metrics,
        "kupiec_1pct": kupiec_1pct,
        "kupiec_5pct": kupiec_5pct,
        "active_universe": active_universe,
        "clusters": clusters,
        "loss_history": loss_history,
    }


if __name__ == "__main__":
    train_and_evaluate_tcn()
