"""
anchor_calibration.py — Multi-level anchor set calibration.

Uses the 100 anchor samples per day to calibrate model predictions.
Implements a hierarchical calibration strategy:
  1. Per-model exact match (if anchor exists for same modelId)
  2. Per-item calibration (aggregate anchors at item level)
  3. Per-category calibration (aggregate anchors at category level)
  4. Global bias correction (fallback)
"""
import numpy as np
import pandas as pd

from src.config import TARGET


def calibrate_predictions(
    predictions: np.ndarray,
    pred_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    model,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Apply multi-level anchor calibration to model predictions.

    Strategy:
    1. Compute model predictions on anchor samples.
    2. Calculate prediction errors at multiple granularity levels.
    3. Apply corrections from most specific to most general.

    Args:
        predictions: Raw model predictions (original price scale).
        pred_df: DataFrame of rows being predicted (with IDs).
        anchor_df: Anchor samples with known prices.
        model: The trained model (to predict anchor set).
        feature_cols: Feature columns used by the model.

    Returns:
        Calibrated predictions.
    """
    if len(anchor_df) == 0:
        print("[Calibration] No anchor samples available, skipping.")
        return predictions

    # Get available feature columns
    available_cols = [c for c in feature_cols if c in anchor_df.columns]

    # Predict anchors using the model
    anchor_pred = np.expm1(model.predict(anchor_df[available_cols]))
    anchor_actual = anchor_df[TARGET].values
    anchor_errors = anchor_actual - anchor_pred  # positive = model underestimates

    # Compute calibration factors at different levels
    calibrations = _compute_calibration_factors(anchor_df, anchor_actual, anchor_pred)

    # Apply calibration hierarchically
    calibrated = predictions.copy()
    for i in range(len(pred_df)):
        row = pred_df.iloc[i]
        correction = _get_correction(row, calibrations)
        calibrated[i] = predictions[i] * correction

    # Ensure non-negative
    calibrated = np.maximum(calibrated, 0)

    _print_calibration_summary(calibrations)

    return calibrated


def _compute_calibration_factors(
    anchor_df: pd.DataFrame,
    anchor_actual: np.ndarray,
    anchor_pred: np.ndarray,
) -> dict:
    """Compute calibration ratio factors at multiple levels."""
    factors = {}

    # Ratio-based calibration: actual / predicted
    ratios = np.where(anchor_pred > 0, anchor_actual / anchor_pred, 1.0)

    # 1. Global ratio
    factors["global_ratio"] = np.median(ratios)

    # 2. Per-model ratio
    model_ratios = {}
    for model_id in anchor_df["modelId"].unique():
        mask = anchor_df["modelId"].values == model_id
        if mask.sum() > 0:
            model_ratios[model_id] = np.median(ratios[mask])
    factors["model_ratios"] = model_ratios

    # 3. Per-item ratio
    item_ratios = {}
    for item_id in anchor_df["itemId"].unique():
        mask = anchor_df["itemId"].values == item_id
        if mask.sum() > 0:
            item_ratios[item_id] = np.median(ratios[mask])
    factors["item_ratios"] = item_ratios

    # 4. Per-category ratio
    if "cat_id" in anchor_df.columns:
        cat_ratios = {}
        for cat_id in anchor_df["cat_id"].unique():
            mask = anchor_df["cat_id"].values == cat_id
            if mask.sum() >= 2:  # Need at least 2 for reliable estimate
                cat_ratios[cat_id] = np.median(ratios[mask])
        factors["cat_ratios"] = cat_ratios
    else:
        factors["cat_ratios"] = {}

    # 5. Per-shop ratio
    shop_ratios = {}
    for shop_id in anchor_df["shopId"].unique():
        mask = anchor_df["shopId"].values == shop_id
        if mask.sum() > 0:
            shop_ratios[shop_id] = np.median(ratios[mask])
    factors["shop_ratios"] = shop_ratios

    return factors


def _get_correction(row: pd.Series, calibrations: dict) -> float:
    """
    Get the most specific calibration correction for a given row.

    Priority: model > item > shop > category > global
    """
    # 1. Exact model match
    model_id = row.get("modelId")
    if model_id in calibrations.get("model_ratios", {}):
        return calibrations["model_ratios"][model_id]

    # 2. Item match
    item_id = row.get("itemId")
    if item_id in calibrations.get("item_ratios", {}):
        return calibrations["item_ratios"][item_id]

    # 3. Shop match
    shop_id = row.get("shopId")
    if shop_id in calibrations.get("shop_ratios", {}):
        return calibrations["shop_ratios"][shop_id]

    # 4. Category match
    cat_id = row.get("cat_id")
    if cat_id in calibrations.get("cat_ratios", {}):
        return calibrations["cat_ratios"][cat_id]

    # 5. Global fallback
    return calibrations.get("global_ratio", 1.0)


def _print_calibration_summary(calibrations: dict) -> None:
    """Print a summary of calibration factors."""
    print("\n" + "=" * 60)
    print("ANCHOR CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"  Global ratio:    {calibrations['global_ratio']:.4f}")
    print(f"  Model-level:     {len(calibrations.get('model_ratios', {}))} entities")
    print(f"  Item-level:      {len(calibrations.get('item_ratios', {}))} entities")
    print(f"  Shop-level:      {len(calibrations.get('shop_ratios', {}))} entities")
    print(f"  Category-level:  {len(calibrations.get('cat_ratios', {}))} categories")
    print("=" * 60 + "\n")


def simple_global_bias_correction(
    predictions: np.ndarray,
    anchor_df: pd.DataFrame,
    model,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Simple global bias correction — baseline calibration.

    Computes the mean additive error on anchor set and applies it
    uniformly to all predictions. Useful for comparison.
    """
    available_cols = [c for c in feature_cols if c in anchor_df.columns]
    anchor_pred = np.expm1(model.predict(anchor_df[available_cols]))
    anchor_actual = anchor_df[TARGET].values

    bias = np.mean(anchor_actual - anchor_pred)
    print(f"[Simple Calibration] Global bias correction: {bias:,.0f} IDR")

    calibrated = predictions + bias
    return np.maximum(calibrated, 0)
