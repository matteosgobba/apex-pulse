"""Conservative gradient-boosted regression for checkpoint-safe features."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from f1_prediction.config import HistGradientBoostingConfig

MODEL_NAME = "hist_gradient_boosting"


def build_hist_gradient_boosting_regressor(
    config: HistGradientBoostingConfig,
) -> HistGradientBoostingRegressor:
    """Build the configured sklearn histogram gradient boosting regressor."""
    return HistGradientBoostingRegressor(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
        random_state=config.random_state,
    )
