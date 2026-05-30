"""
model_global.py — Tier 1: Global Marketplace Model.

Trains a single LightGBM model on the entire historical dataset.
Acts as a marketplace-wide price predictor that generalises across
all shops, items, and categories.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os

from src.config import (
    LGBM_PARAMS_GLOBAL,
    LOG_TARGET,
    MODEL_DIR,
    RANDOM_SEED,
    TARGET,
)
from src.feature_engineering import get_feature_columns


def train_global_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    save: bool = True,
) -> lgb.LGBMRegressor:
    """
    Train the global marketplace model (Tier 1).

    Args:
        train_df: Training data with all features computed.
        val_df: Optional validation data for early stopping.
        save: Whether to save the model to disk.

    Returns:
        Trained LGBMRegressor.
    """
    feature_cols = get_feature_columns(for_product_model=False)
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df[LOG_TARGET]

    params = LGBM_PARAMS_GLOBAL.copy()
    n_estimators = params.pop("n_estimators", 2000)
    early_stopping = params.pop("early_stopping_rounds", 100)

    model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)

    fit_params = {}
    if val_df is not None:
        X_val = val_df[feature_cols]
        y_val = val_df[LOG_TARGET]
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["callbacks"] = [
            lgb.early_stopping(stopping_rounds=early_stopping),
            lgb.log_evaluation(period=200),
        ]

    model.fit(X_train, y_train, **fit_params)

    if save:
        path = os.path.join(MODEL_DIR, "global_model.pkl")
        joblib.dump(model, path)
        print(f"[Tier 1] Global model saved to {path}")

    return model


def predict_global(
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions using the global model.

    Returns predictions in original price scale (not log).
    """
    feature_cols = get_feature_columns(for_product_model=False)
    feature_cols = [c for c in feature_cols if c in df.columns]

    log_pred = model.predict(df[feature_cols])
    pred = np.expm1(log_pred)  # Inverse of log1p
    pred = np.maximum(pred, 0)  # Ensure non-negative
    return pred


def load_global_model(path: str | None = None) -> lgb.LGBMRegressor:
    """Load a previously saved global model."""
    if path is None:
        path = os.path.join(MODEL_DIR, "global_model.pkl")
    return joblib.load(path)


def get_feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract feature importance from the trained model.

    Returns a DataFrame sorted by importance.
    """
    if feature_names is None:
        feature_names = model.feature_name_

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance
