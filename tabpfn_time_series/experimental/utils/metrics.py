import numpy as np
from sklearn.metrics import mean_pinball_loss


def compute_wql(
    y_test: np.ndarray,
    pred_quantiles: np.ndarray,
) -> float:
    """
    Compute weighted quantile (pinball) loss by averaging over α levels.

    pred_quantiles: array of shape (9, n_timesteps), quantile forecasts for α=0.1…0.9
    y_test:          array of shape (n_timesteps,), true values
    """
    alphas = np.arange(0.1, 1.0, 0.1)
    losses = [
        mean_pinball_loss(y_true=y_test, y_pred=q, alpha=alpha)
        for alpha, q in zip(alphas, pred_quantiles)
    ]
    return float(np.mean(losses))


def compute_mase(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Compute Mean Absolute Scaled Error (MASE).

    MASE normalizes MAE by the MAE of a naive seasonal forecast.
    For seasonality=1, uses naive forecast (previous value).

    Args:
        y_test: True test values
        y_pred: Predicted values
        y_train: Training values (used to compute naive forecast error)
        seasonality: Seasonal period for naive forecast (1 for non-seasonal)

    Returns:
        MASE value (lower is better, <1 means better than naive)
    """
    # Forecast error
    forecast_error = np.abs(y_test - y_pred)

    # Naive forecast error from training data
    if seasonality == 1:
        # Simple naive: use previous value
        naive_error = np.abs(y_train[1:] - y_train[:-1])
    else:
        # Seasonal naive: use value from previous season
        naive_error = np.abs(y_train[seasonality:] - y_train[:-seasonality])

    # Avoid division by zero
    naive_mae = np.mean(naive_error)
    if naive_mae == 0:
        return np.inf if np.mean(forecast_error) > 0 else 0.0

    mase = np.mean(forecast_error) / naive_mae
    return float(mase)


def compute_sql(
    y_test: np.ndarray,
    pred_quantiles: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Compute Scaled Quantile Loss (SQL).

    SQL normalizes the weighted quantile loss by the naive forecast error,
    similar to how MASE scales MAE.

    Args:
        y_test: True test values
        pred_quantiles: Quantile predictions (9, n_timesteps)
        y_train: Training values (used to compute naive forecast error)
        seasonality: Seasonal period for naive forecast (1 for non-seasonal)

    Returns:
        SQL value (lower is better, <1 means better than naive)
    """
    # Compute WQL
    wql = compute_wql(y_test, pred_quantiles)

    # Compute naive forecast error (same as in MASE)
    if seasonality == 1:
        naive_error = np.abs(y_train[1:] - y_train[:-1])
    else:
        naive_error = np.abs(y_train[seasonality:] - y_train[:-seasonality])

    # Avoid division by zero
    naive_mae = np.mean(naive_error)
    if naive_mae == 0:
        return np.inf if wql > 0 else 0.0

    sql = wql / naive_mae
    return float(sql)
