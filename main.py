"""
main.py — Entry point for the MrScraper Price Intelligence system.

Usage:
    python main.py --mode train      # Train + validate both models
    python main.py --mode predict    # Run inference on test data
    python main.py --mode all        # Train + validate + predict

This script orchestrates the full pipeline:
  1. Data loading & preprocessing
  2. Feature engineering
  3. Validation (simulated outage scenario)
  4. Training both Tier 1 (Global) and Tier 2 (Product) models
  5. Inference on test data with anchor calibration
"""
import argparse
import warnings
import time
import sys
import os

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    ANCHOR_SAMPLE_SIZE,
    LOG_TARGET,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET,
)
from src.data_preprocessing import load_train, handle_outliers
from src.feature_engineering import (
    add_derived_features,
    add_temporal_features,
    build_entity_price_stats,
    encode_brand,
    get_feature_columns,
    merge_entity_features,
)
from src.model_global import train_global_model, predict_global, get_feature_importance
from src.model_product import train_product_model, predict_product
from src.anchor_calibration import calibrate_predictions, simple_global_bias_correction
from src.evaluate import compute_metrics, compare_approaches, per_category_metrics
from src.inference import run_inference

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


def train_and_validate():
    """
    Full training + validation pipeline.

    Simulates the outage scenario by holding out the last day's data
    as validation, with 100 random samples as the simulated anchor set.
    """
    print("=" * 70)
    print("  MrScraper — Price Intelligence & Anomaly Detection")
    print("  Training & Validation Pipeline")
    print("=" * 70)

    start_time = time.time()

    # ============================================================
    # Step 1: Load and preprocess data
    # ============================================================
    print("\n📦 [Step 1/7] Loading and preprocessing data...")
    df = load_train()
    print(f"  Loaded {len(df):,} rows")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Unique dates: {df['date'].nunique()}")

    # Handle outliers
    df = handle_outliers(df)
    print("  Outliers capped at 1st/99th percentile")

    # ============================================================
    # Step 2: Time-based validation split
    # ============================================================
    print("\n📅 [Step 2/7] Creating time-based validation split...")

    # Sort dates and pick the last full day as validation
    all_dates = sorted(df["date"].unique())
    val_date = all_dates[-1]  # Last day = 2025-03-22

    # Use the second-to-last day as an additional check
    train_mask = df["date"] < val_date
    val_mask = df["date"] == val_date

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    print(f"  Training:   {len(train_df):,} rows (up to {all_dates[-2]})")
    print(f"  Validation: {len(val_df):,} rows (date = {val_date})")

    # Simulate anchor set: random 100 samples from validation day
    anchor_indices = np.random.choice(val_df.index, size=ANCHOR_SAMPLE_SIZE, replace=False)
    simulated_anchors = val_df.loc[anchor_indices].copy()
    print(f"  Simulated anchor set: {len(simulated_anchors)} samples")

    # ============================================================
    # Step 3: Feature engineering
    # ============================================================
    print("\n🔧 [Step 3/7] Engineering features...")

    # Temporal features
    train_df = add_temporal_features(train_df)
    val_df = add_temporal_features(val_df)

    # Entity price stats (computed from training data only — no leakage!)
    entity_stats = build_entity_price_stats(train_df)
    print(f"  Entity stats computed:")
    for level, stat_df in entity_stats.items():
        print(f"    {level}: {len(stat_df)} entities")

    # Merge entity features
    train_df = merge_entity_features(train_df, entity_stats)
    val_df = merge_entity_features(val_df, entity_stats)

    # Derived features
    train_df = add_derived_features(train_df)
    val_df = add_derived_features(val_df)

    # Brand encoding
    train_df, val_df = encode_brand(train_df, val_df)

    # Re-index simulated anchors with new features
    simulated_anchors = val_df.loc[val_df.index.isin(anchor_indices)].copy()

    print(f"  Feature columns: {len(get_feature_columns())}")

    # ============================================================
    # Step 4: Train Tier 1 — Global Model
    # ============================================================
    print("\n🌐 [Step 4/7] Training Tier 1 — Global Marketplace Model...")

    # Use a small portion of training data as early stopping set
    early_stop_idx = np.random.choice(
        train_df.index, size=min(10000, len(train_df) // 5), replace=False
    )
    early_stop_df = train_df.loc[early_stop_idx]
    train_for_model = train_df.drop(early_stop_idx)

    global_model = train_global_model(
        train_for_model, val_df=early_stop_df, save=True
    )

    # Feature importance
    fi = get_feature_importance(global_model)
    print("\n  Top 15 features (Global Model):")
    print(fi.head(15).to_string(index=False))

    # ============================================================
    # Step 5: Train Tier 2 — Product Model
    # ============================================================
    print("\n🎯 [Step 5/7] Training Tier 2 — Product Level Model...")

    product_model = train_product_model(
        train_for_model, val_df=early_stop_df, save=True
    )

    # Feature importance
    fi2 = get_feature_importance(product_model)
    print("\n  Top 15 features (Product Model):")
    print(fi2.head(15).to_string(index=False))

    # ============================================================
    # Step 6: Validation — Predict & Evaluate
    # ============================================================
    print("\n📊 [Step 6/7] Validating on held-out day...")

    y_true = val_df[TARGET].values

    # --- Tier 1: Global Model ---
    global_pred = predict_global(global_model, val_df)

    # --- Tier 2: Product Model ---
    product_pred = predict_product(product_model, val_df)

    # Metrics (before calibration)
    print("\n── Before Anchor Calibration ──")
    compute_metrics(y_true, global_pred, "Tier 1 — Global (no calibration)")
    compute_metrics(y_true, product_pred, "Tier 2 — Product (no calibration)")

    # --- Anchor Calibration ---
    print("\n── Applying Anchor Calibration ──")
    feature_cols_global = get_feature_columns(for_product_model=False)
    feature_cols_product = get_feature_columns(for_product_model=True)

    # Calibrate global model
    global_calibrated = calibrate_predictions(
        global_pred, val_df, simulated_anchors, global_model, feature_cols_global
    )

    # Calibrate product model
    product_calibrated = calibrate_predictions(
        product_pred, val_df, simulated_anchors, product_model, feature_cols_product
    )

    # Simple global bias (for comparison)
    global_simple_bias = simple_global_bias_correction(
        global_pred, simulated_anchors, global_model, feature_cols_global
    )

    # Metrics (after calibration)
    print("\n── After Anchor Calibration ──")
    compute_metrics(y_true, global_calibrated, "Tier 1 — Global (multi-level calib)")
    compute_metrics(y_true, global_simple_bias, "Tier 1 — Global (simple bias)")
    compute_metrics(y_true, product_calibrated, "Tier 2 — Product (multi-level calib)")

    # ============================================================
    # Step 7: Comparison Summary
    # ============================================================
    print("\n📈 [Step 7/7] Generating comparison summary...")

    comparison = compare_approaches(
        y_true,
        global_pred,
        product_pred,
        global_calibrated=global_calibrated,
        product_calibrated=product_calibrated,
    )

    # Per-category breakdown for the best model
    if "cat_id" in val_df.columns:
        per_category_metrics(
            y_true,
            product_calibrated,
            val_df["cat_id"].values,
            "Tier 2 — Product (calibrated)",
        )

    # Save comparison
    comparison.to_csv(os.path.join(OUTPUT_DIR, "validation_comparison.csv"))
    print(f"\n  Comparison saved to {OUTPUT_DIR}/validation_comparison.csv")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("✅ Training & validation complete!")

    return global_model, product_model


def predict():
    """Run inference on test data."""
    print("\n" + "=" * 70)
    print("  Running Inference Pipeline")
    print("=" * 70)

    output_df = run_inference()
    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="MrScraper — Price Intelligence & Anomaly Detection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "predict", "all"],
        help="Mode: train (train+validate), predict (inference), all (both)",
    )
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        train_and_validate()

    if args.mode in ("predict", "all"):
        predict()


if __name__ == "__main__":
    main()
