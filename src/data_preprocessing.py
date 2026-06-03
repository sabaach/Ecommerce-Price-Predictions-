"""
data_preprocessing.py — Data loading, cleaning, and type conversion.

Handles:
  - CSV loading for train and test datasets
  - Datetime parsing
  - Dropping high-null columns (stock, normal_stock)
  - Boolean encoding (t/f → 1/0)
  - Missing value imputation
  - Log-transform of price target
  - Outlier handling
"""
import numpy as np
import pandas as pd

from src.config import (
    BOOL_COLS,
    DATETIME_COL,
    DROP_COLS,
    LOG_TARGET,
    TARGET,
    TRAIN_CSV,
    TEST_CSV,
)


def load_train(path: str = TRAIN_CSV) -> pd.DataFrame:
    """Load and preprocess the training CSV."""
    df = pd.read_csv(path)
    df = _common_preprocessing(df)

    # Log-transform price
    df[LOG_TARGET] = np.log1p(df[TARGET])

    return df


def load_test(path: str = TEST_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the test CSV and split into anchor rows and prediction rows.

    Returns:
        anchors: rows where price is not null (anchor samples)
        to_predict: rows where price is null (need prediction)
    """
    df = pd.read_csv(path)

    # Separate anchors (price not null) vs rows to predict
    anchors = df[df[TARGET].notna()].copy()
    to_predict = df[df[TARGET].isna()].copy()

    # Preprocess anchors (they have full data)
    anchors = _common_preprocessing(anchors)
    anchors[LOG_TARGET] = np.log1p(anchors[TARGET])

    # For prediction rows, only capturedAt and ID cols are available
    to_predict["capturedAt_orig"] = to_predict[DATETIME_COL].copy()
    to_predict[DATETIME_COL] = pd.to_datetime(to_predict[DATETIME_COL])
    to_predict["date"] = to_predict[DATETIME_COL].dt.date

    return anchors, to_predict


def _common_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Shared preprocessing for train and anchor data."""

    # 1. Parse datetime
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df["date"] = df[DATETIME_COL].dt.date

    # 2. Drop high-null columns
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 3. Boolean encoding: t/f → 1/0
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = (df[col] == "t").astype(int)

    # 4. Fill missing brand with "unknown"
    if "brand" in df.columns:
        df["brand"] = df["brand"].fillna("unknown")

    # 5. Fill missing shop_response_rate with median
    if "shop_response_rate" in df.columns:
        median_rate = df["shop_response_rate"].median()
        df["shop_response_rate"] = df["shop_response_rate"].fillna(median_rate)

    return df


def handle_outliers(
    df: pd.DataFrame,
    col: str = TARGET,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Cap extreme outliers in the target column.
    Uses percentile-based capping to handle data entry errors and flash sales.
    """
    lower = df[col].quantile(lower_pct)
    upper = df[col].quantile(upper_pct)
    df[col] = df[col].clip(lower, upper)

    # Recompute log target if applicable
    if LOG_TARGET in df.columns:
        df[LOG_TARGET] = np.log1p(df[col])

    return df


def get_date_range(df: pd.DataFrame) -> dict:
    """Return summary statistics about the date range in the dataset."""
    dates = pd.to_datetime(df["date"])
    return {
        "min_date": dates.min(),
        "max_date": dates.max(),
        "n_unique_dates": dates.nunique(),
        "total_rows": len(df),
    }
