"""
model_product.py — Tier 2: Shop / Product Level Model.

Uses a single LightGBM model enhanced with entity-specific features
(per shopId, itemId, modelId) to capture product-specific pricing logic.
Implements a hierarchical fallback for cold-start entities.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os

from src.config import (
    LGBM_PARAMS_PRODUCT,
    LOG_TARGET,
    MIN_ENTITY_OBS,
    MODEL_DIR,
    TARGET,
)
from src.feature_engineering import get_feature_columns


def train_product_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    save: bool = True,
) -> lgb.LGBMRegressor:
    """
    Train the product-level model (Tier 2).

    This model is a single LightGBM trained on the full dataset but
    enriched with entity-specific features (model_price_last, model_price_mean, etc.)
    that effectively condition it on each product's pricing history.

    Args:
        train_df: Training data with all features (including entity stats).
        val_df: Optional validation data for early stopping.
        save: Whether to save the model.

    Returns:
        Trained LGBMRegressor.
    """
    feature_cols = get_feature_columns(for_product_model=True)
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df[LOG_TARGET]

    params = LGBM_PARAMS_PRODUCT.copy()
    n_estimators = params.pop("n_estimators", 3000)
    early_stopping = params.pop("early_stopping_rounds", 150)

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
        path = os.path.join(MODEL_DIR, "product_model.pkl")
        joblib.dump(model, path)
        print(f"[Tier 2] Product model saved to {path}")

    return model


def predict_product(
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions using the product-level model.

    Returns predictions in original price scale.
    """
    feature_cols = get_feature_columns(for_product_model=True)
    feature_cols = [c for c in feature_cols if c in df.columns]

    log_pred = model.predict(df[feature_cols])
    pred = np.expm1(log_pred)
    pred = np.maximum(pred, 0)
    return pred


def apply_hierarchical_fallback(
    df: pd.DataFrame,
    product_pred: np.ndarray,
    global_pred: np.ndarray,
    entity_stats: dict,
) -> np.ndarray:
    """
    Apply hierarchical fallback for entities with insufficient history.

    Priority:
    1. Product model prediction (if model has >= MIN_ENTITY_OBS history)
    2. Item-level last price (if item has history)
    3. Global model prediction (ultimate fallback)

    Args:
        df: Input dataframe with entity IDs.
        product_pred: Tier 2 predictions.
        global_pred: Tier 1 predictions.
        entity_stats: Dict of entity-level stats DataFrames.

    Returns:
        Final blended predictions.
    """
    final_pred = product_pred.copy()

    if "model" in entity_stats:
        model_counts = entity_stats["model"]["model_price_count"]

        for i, row in df.iterrows():
            model_id = row["modelId"]
            count = model_counts.get(model_id, 0)

            if count < MIN_ENTITY_OBS:
                # Check item-level fallback
                item_id = row["itemId"]
                if "item" in entity_stats:
                    item_count = entity_stats["item"]["item_price_count"].get(
                        item_id, 0
                    )
                    if item_count >= MIN_ENTITY_OBS:
                        # Use item-level last price as anchor
                        item_last = entity_stats["item"]["item_price_last"].get(
                            item_id
                        )
                        if pd.notna(item_last):
                            idx = df.index.get_loc(i)
                            final_pred[idx] = item_last
                            continue

                # Ultimate fallback: global model
                idx = df.index.get_loc(i)
                final_pred[idx] = global_pred[idx]

    return final_pred


def load_product_model(path: str | None = None) -> lgb.LGBMRegressor:
    """Load a previously saved product model."""
    if path is None:
        path = os.path.join(MODEL_DIR, "product_model.pkl")
    return joblib.load(path)
