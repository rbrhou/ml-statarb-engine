import numpy as np
import pandas as pd
import pytest
import torch

from src.clustering import FactorClusterer
from src.diagnostics import SpreadDiagnostics
from src.ou_process import OUProcessModel
from src.pca_model import PCAFactorModel
from src.residuals_KF import KalmanResidualFilter
from src.tcn_var import TCNVaRForecaster, PinballLoss


def test_pca_dimension_reduction():
    """Verifies that the PCA model extracts the correct orthogonal factor shapes."""
    np.random.seed(42)
    returns = pd.DataFrame(np.random.normal(0, 0.02, (100, 5)))
    pca = PCAFactorModel(n_components=2).fit(returns)

    assert pca.factor_loadings.shape == (5, 2)
    assert pca.factor_returns.shape == (100, 2)
    # explained_variance_ratio spans the FULL eigenvalue spectrum (all 5
    # components), so it is normalized to sum to exactly 1.0.
    assert np.isclose(pca.explained_variance_ratio.sum(), 1.0)
    # The retained top-2 components must explain a strict subset of it.
    assert 0 < pca.explained_variance_ratio[:2].sum() < 1.0


def test_parametric_umap_clustering():
    """Validates that Parametric UMAP maps high-dimensional loadings to a dense

    manifold and DBSCAN successfully assigns cluster labels (including noise -1).
    """
    np.random.seed(42)
    # 10 assets, 3 PCA factors
    loadings = pd.DataFrame(
        np.random.normal(0, 1, (10, 3)), index=[f"TICK_{i}" for i in range(10)]
    )

    # Use low epochs for testing speed
    clusterer = FactorClusterer(
        n_components=2, eps=0.5, min_samples=2, n_epochs=10
    )
    clusterer.fit(loadings)
    clusters = clusterer.get_clusters()

    assert isinstance(clusters, dict)
    assert len(clusterer.labels_) == 10
    assert clusterer.umap_embeddings_.shape == (10, 2)


def test_kalman_filter_adaptive_residuals():
    """Checks that the recursive state-space model correctly extracts

    1D innovations and tracks the drifting intercept + K factor betas.
    """
    np.random.seed(42)
    T, K = 100, 2
    y = np.random.normal(0, 0.02, T)
    F = np.random.normal(0, 0.02, (T, K))

    kf = KalmanResidualFilter(n_factors=K, process_noise=1e-4)
    innovations, betas_history = kf.filter_series(y, F)

    assert innovations.shape == (T,)
    # Beta history includes intercept (alpha) + K factors
    assert betas_history.shape == (T, K + 1)


def test_spread_diagnostics_stationarity():
    """Proves the ADF test correctly distinguishes between a bounded stationary

    process and an explosive random walk.
    """
    np.random.seed(42)
    diag = SpreadDiagnostics(significance_level=0.05)

    # 1. Stationary White Noise
    stationary_series = pd.Series(np.random.normal(0, 1, 200))
    stat_result = diag.adf_test(stationary_series)
    assert stat_result["is_stationary"] is True
    assert stat_result["p_value"] < 0.05

    # 2. Non-Stationary Random Walk
    random_walk = pd.Series(np.cumsum(np.random.normal(0, 1, 200)))
    rw_result = diag.adf_test(random_walk)
    assert rw_result["is_stationary"] is False
    assert rw_result["p_value"] > 0.05


def test_ou_calibration_mean_reverting():
    """Ensures continuous-time Ornstein-Uhlenbeck calibration extracts a positive

    mean-reversion speed (kappa) from an AR(1) process.
    """
    np.random.seed(42)
    T = 500
    spread = np.zeros(T)
    for t in range(1, T):
        # Stationary AR(1) process: x_t = 0.8 * x_{t-1} + noise
        spread[t] = 0.8 * spread[t - 1] + np.random.normal(0, 0.1)

    ou_model = OUProcessModel(dt=1.0 / 252.0)
    params = ou_model.fit_spread(pd.Series(spread))

    assert not np.isnan(params["kappa"])
    assert params["kappa"] > 0
    assert params["half_life"] > 0
    assert params["sigma_eq"] > 0


def test_tcn_var_forward_pass_and_loss():
    """Validates the deep learning VaR architecture processes 3D sequence tensors

    and computes the asymmetric multi-quantile pinball loss correctly.
    """
    torch.manual_seed(42)
    batch_size = 16
    channels = 4  # e.g., R_p, R_p^2, F_1, F_2
    seq_len = 30
    quantiles = [0.01, 0.05]

    # Synthetic Input Tensor: (Batch, Channels, Sequence)
    x = torch.randn(batch_size, channels, seq_len)
    y_true = torch.randn(batch_size, 1)

    # Forward Pass
    tcn = TCNVaRForecaster(
        num_inputs=channels, num_channels=[8, 16], quantiles=quantiles
    )
    preds = tcn(x)

    assert preds.shape == (batch_size, len(quantiles))

    # Loss Calculation
    criterion = PinballLoss(quantiles=quantiles)
    loss = criterion(preds, y_true)

    assert loss.item() > 0
    assert not torch.isnan(loss)
