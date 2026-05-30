"""
inference.py — Prediction pipeline for test data.

Handles:
  - Loading test data and splitting anchors vs prediction rows
  - Reconstructing features for prediction rows using train history
  - Running both Global and Product models
  - Applying anchor calibration per day
  - Producing the final output CSV
"""
import numpy as np
import pandas as pd
import os

from src.config import (
    DATETIME_COL,
    LOG_TARGET,
    OUTPUT_DIR,
    TARGET,
    TEST_CSV,
)
from src.data_preprocessing import load_test, load_train
from src.feature_engineering import (
    add_derived_features,
    add_temporal_features,
    build_entity_price_stats,
    encode_brand,
    get_feature_columns,
    merge_entity_features,
)
from src.anchor_calibration import calibrate_predictions
from src.model_global import load_global_model, predict_global
from src.model_product import load_product_model, predict_product


def run_inference(
    test_path: str = TEST_CSV,
    output_name: str = "predictions_final.csv",
) -> pd.DataFrame:
    """
    Full inference pipeline.

    1. Load train data (for entity stats)
    2. Load test data (split anchors / to_predict)
    3. Build features for prediction rows
    4. Predict with both models
    5. Calibrate using anchors
    6. Save results

    Returns:
        DataFrame with filled predictions.
    """
    print("=" * 60)
    print("INFERENCE PIPELINE")
    print("=" * 60)

    # --- Step 1: Load training data for feature stats ---
    print("\n[1/6] Loading training data for entity statistics...")
    train_df = load_train()
    train_df = add_temporal_features(train_df)
    entity_stats = build_entity_price_stats(train_df)
    print(f"  Train data loaded: {len(train_df)} rows")

    # --- Step 2: Load test data ---
    print("\n[2/6] Loading test data...")
    anchors_raw, to_predict_raw = load_test(test_path)
    print(f"  Anchors: {len(anchors_raw)} rows")
    print(f"  To predict: {len(to_predict_raw)} rows")

    # --- Step 3: Feature engineering for anchors ---
    print("\n[3/6] Engineering features for anchor samples...")
    anchors = _prepare_features(anchors_raw, train_df, entity_stats)

    # --- Step 4: Feature engineering for prediction rows ---
    print("\n[4/6] Engineering features for prediction rows...")
    to_predict = _prepare_prediction_features(to_predict_raw, train_df, entity_stats)

    # --- Step 5: Load models and predict ---
    print("\n[5/6] Loading models and generating predictions...")
    global_model = load_global_model()
    product_model = load_product_model()

    # Process per day for proper per-day anchor calibration
    dates = sorted(to_predict["date"].unique())
    all_results = []

    for date in dates:
        print(f"\n  --- Processing date: {date} ---")
        day_mask = to_predict["date"] == date
        day_df = to_predict[day_mask].copy()

        day_anchor_mask = anchors["date"] == date
        day_anchors = anchors[day_anchor_mask].copy()
        print(f"  Predictions: {len(day_df)}, Anchors: {len(day_anchors)}")

        # Raw predictions
        global_pred = predict_global(global_model, day_df)
        product_pred = predict_product(product_model, day_df)

        # Calibrate using day-specific anchors
        feature_cols_global = get_feature_columns(for_product_model=False)
        feature_cols_product = get_feature_columns(for_product_model=True)

        global_calibrated = calibrate_predictions(
            global_pred, day_df, day_anchors, global_model, feature_cols_global
        )
        product_calibrated = calibrate_predictions(
            product_pred, day_df, day_anchors, product_model, feature_cols_product
        )

        # Use product model (Tier 2) as primary — generally more accurate
        day_df["pred_global"] = global_calibrated
        day_df["pred_product"] = product_calibrated
        day_df["pred_final"] = product_calibrated  # Tier 2 is primary

        all_results.append(day_df)

    results = pd.concat(all_results, ignore_index=True)

    # --- Step 6: Build output CSV ---
    print("\n[6/6] Building output CSV...")
    output_df = _build_output(to_predict_raw, results, test_path)

    output_path = os.path.join(OUTPUT_DIR, output_name)
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"   Total rows: {len(output_df)}")
    print(f"   Filled predictions: {output_df[TARGET].notna().sum()}")
    print(f"   Remaining nulls: {output_df[TARGET].isna().sum()}")

    # Also save separate model outputs
    _save_separate_outputs(results)

    return output_df


def _prepare_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    entity_stats: dict,
) -> pd.DataFrame:
    """Prepare features for a dataframe that has full columns (anchors)."""
    df = add_temporal_features(df)
    df = merge_entity_features(df, entity_stats)
    df = add_derived_features(df)

    # Brand encoding
    _, df = encode_brand(train_df, df)

    return df


def _prepare_prediction_features(
    pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    entity_stats: dict,
) -> pd.DataFrame:
    """
    Prepare features for prediction rows.

    Since prediction rows only have (capturedAt, shopId, itemId, modelId),
    we need to fill in all other features from training history.
    """
    # Temporal features
    pred_df = add_temporal_features(pred_df)

    # Get the latest observation for each modelId from training data
    latest_obs = (
        train_df.sort_values(DATETIME_COL)
        .groupby("modelId")
        .last()
        .reset_index()
    )

    # Merge latest known features onto prediction rows
    feature_cols_from_train = [
        "cat_id",
        "priceBeforeDiscount",
        "promotionId",
        "raw_discount",
        "show_discount",
        "brand",
        "is_free_shipping",
        "is_pre_order",
        "item_price_min",
        "item_price_max",
        "review_rating",
        "total_rating_count",
        "cmt_count",
        "shop_rating",
        "shop_response_rate",
        "shop_follower_count",
        "is_official_shop",
        "is_verified",
        "is_preferred_plus_seller",
    ]
    
    # Drop these columns from pred_df so they don't get _x/_y suffixes during merge
    cols_to_drop = [c for c in feature_cols_from_train if c in pred_df.columns]
    pred_df = pred_df.drop(columns=cols_to_drop)
    
    available_cols = ["modelId"] + [c for c in feature_cols_from_train if c in latest_obs.columns]
    pred_df = pred_df.merge(latest_obs[available_cols], on="modelId", how="left")

    # Fill any remaining brand NaN
    if "brand" in pred_df.columns:
        pred_df["brand"] = pred_df["brand"].fillna("unknown")

    # Entity stats
    pred_df = merge_entity_features(pred_df, entity_stats)

    # Derived features
    pred_df = add_derived_features(pred_df)

    # Brand encoding
    _, pred_df = encode_brand(train_df, pred_df)

    return pred_df


def _build_output(
    original_test: pd.DataFrame,
    results: pd.DataFrame,
    test_path: str,
) -> pd.DataFrame:
    """
    Build the final output CSV matching the original test format.

    Fills in the predicted prices while keeping anchor prices intact.
    """
    # Reload original test to preserve format
    output_df = pd.read_csv(test_path)

    # Create a mapping from (capturedAt, shopId, itemId, modelId) → predicted price
    pred_map = {}
    for _, row in results.iterrows():
        key = (
            str(row.get("capturedAt_orig", row[DATETIME_COL])),
            row["shopId"],
            row["itemId"],
            row["modelId"],
        )
        pred_map[key] = row["pred_final"]

    # Fill missing prices
    filled = 0
    for idx, row in output_df.iterrows():
        if pd.isna(row[TARGET]):
            key = (row[DATETIME_COL], row["shopId"], row["itemId"], row["modelId"])
            if key in pred_map:
                output_df.at[idx, TARGET] = pred_map[key]
                filled += 1

    # For any remaining nulls, try matching on just modelId
    # (handles duplicate entries)
    remaining_nulls = output_df[output_df[TARGET].isna()]
    if len(remaining_nulls) > 0:
        model_prices = results.groupby("modelId")["pred_final"].median().to_dict()
        for idx, row in remaining_nulls.iterrows():
            if row["modelId"] in model_prices:
                output_df.at[idx, TARGET] = model_prices[row["modelId"]]
                filled += 1

    # Ultimate fallback: global median
    global_median = results["pred_final"].median()
    output_df[TARGET] = output_df[TARGET].fillna(global_median)

    print(f"  Filled {filled} predictions, fallback applied to rest")
    return output_df


def _save_separate_outputs(results: pd.DataFrame) -> None:
    """Save separate output files for each model tier."""
    for col, name in [
        ("pred_global", "predictions_global.csv"),
        ("pred_product", "predictions_product.csv"),
    ]:
        if col in results.columns:
            out = results[["shopId", "itemId", "modelId", "date", col]].copy()
            out = out.rename(columns={col: TARGET})
            path = os.path.join(OUTPUT_DIR, name)
            out.to_csv(path, index=False)
            print(f"  Saved {name}")
