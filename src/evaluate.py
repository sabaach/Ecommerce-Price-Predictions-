"""
evaluate.py — Validation metrics and reporting.

Computes MAE, RMSE, MAPE and generates comparison tables
between Tier 1 and Tier 2 approaches.
"""
import numpy as np
import pandas as pd

from src.config import TARGET


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = ""
) -> dict[str, float]:
    """
    Compute regression metrics.

    Returns:
        Dict with MAE, RMSE, MAPE, and Median Absolute Error.
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))

    # MAPE: avoid division by zero
    nonzero_mask = y_true > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(errors[nonzero_mask]) / y_true[nonzero_mask]) * 100
    else:
        mape = float("nan")

    median_ae = np.median(abs_errors)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Median AE": median_ae,
    }

    if label:
        _print_metrics(metrics, label)

    return metrics


def _print_metrics(metrics: dict, label: str) -> None:
    """Pretty-print metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    for name, value in metrics.items():
        if "MAPE" in name:
            print(f"  {name:15s}: {value:.2f}%")
        else:
            print(f"  {name:15s}: {value:,.0f} IDR")
    print(f"{'=' * 50}\n")


def compare_approaches(
    y_true: np.ndarray,
    global_pred: np.ndarray,
    product_pred: np.ndarray,
    global_calibrated: np.ndarray | None = None,
    product_calibrated: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Create a comparison table of metrics across approaches.
    """
    rows = []

    # Global (uncalibrated)
    m = compute_metrics(y_true, global_pred, "Tier 1 — Global (no calibration)")
    rows.append({"Approach": "Tier 1 — Global (no calibration)", **m})

    # Global (calibrated)
    if global_calibrated is not None:
        m = compute_metrics(y_true, global_calibrated, "Tier 1 — Global (calibrated)")
        rows.append({"Approach": "Tier 1 — Global (calibrated)", **m})

    # Product (uncalibrated)
    m = compute_metrics(y_true, product_pred, "Tier 2 — Product (no calibration)")
    rows.append({"Approach": "Tier 2 — Product (no calibration)", **m})

    # Product (calibrated)
    if product_calibrated is not None:
        m = compute_metrics(
            y_true, product_calibrated, "Tier 2 — Product (calibrated)"
        )
        rows.append({"Approach": "Tier 2 — Product (calibrated)", **m})

    comparison = pd.DataFrame(rows).set_index("Approach")
    
    # Round metrics for cleaner output
    for col in ["MAE", "RMSE", "Median AE"]:
        if col in comparison.columns:
            comparison[col] = comparison[col].round(0).astype(int)
    if "MAPE (%)" in comparison.columns:
        comparison["MAPE (%)"] = comparison["MAPE (%)"].round(2)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison.to_string())
    print("=" * 80 + "\n")

    return comparison


def per_category_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cat_ids: np.ndarray,
    label: str = "",
) -> pd.DataFrame:
    """Compute metrics broken down by category."""
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "cat_id": cat_ids,
        }
    )

    rows = []
    for cat_id, group in df.groupby("cat_id"):
        m = compute_metrics(group["y_true"].values, group["y_pred"].values)
        m["cat_id"] = cat_id
        m["count"] = len(group)
        rows.append(m)

    result = pd.DataFrame(rows).sort_values("MAE", ascending=False)
    if label:
        print(f"\n[{label}] Per-Category Metrics:")
        print(result.to_string(index=False))
    return result
