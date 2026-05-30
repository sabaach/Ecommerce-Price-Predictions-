"""
feature_engineering.py — Feature creation pipeline.

Creates features in multiple categories:
  - Temporal features from capturedAt
  - Price-derived historical features (per entity)
  - Shop-level aggregated features
  - Category-level aggregated features
  - Interaction / relative features
  - Brand encoding
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import DATETIME_COL, ID_COLS, LOG_TARGET, MIN_ENTITY_OBS, TARGET


# ============================================================
# Temporal Features
# ============================================================
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from capturedAt."""
    dt = df[DATETIME_COL]

    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["hour"] = dt.dt.hour

    # Days since the earliest date in the dataset
    min_date = dt.min()
    df["days_since_start"] = (dt - min_date).dt.total_seconds() / 86400

    return df


# ============================================================
# Historical Price Features (per entity)
# ============================================================
def build_entity_price_stats(
    train_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Compute historical price statistics for each entity level:
      - modelId level
      - itemId level
      - shopId level
      - (shopId, cat_id) level

    Returns a dict of DataFrames keyed by level name.
    """
    stats = {}

    # --- Model level (most granular) ---
    model_grp = train_df.groupby("modelId")[TARGET]
    stats["model"] = pd.DataFrame(
        {
            "model_price_mean": model_grp.mean(),
            "model_price_std": model_grp.std().fillna(0),
            "model_price_median": model_grp.median(),
            "model_price_min": model_grp.min(),
            "model_price_max": model_grp.max(),
            "model_price_count": model_grp.count(),
        }
    )

    # Last known price per model (most recent observation)
    last_price = (
        train_df.sort_values(DATETIME_COL)
        .groupby("modelId")
        .last()[[TARGET]]
        .rename(columns={TARGET: "model_price_last"})
    )
    stats["model"] = stats["model"].join(last_price)

    # Price momentum (slope): fit linear trend on log_price over time
    def _compute_momentum(group):
        if len(group) < 3:
            return 0.0
        x = group["days_since_start"].values
        y = group[LOG_TARGET].values
        if x.std() == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return slope

    if "days_since_start" in train_df.columns and LOG_TARGET in train_df.columns:
        momentum = train_df.groupby("modelId").apply(
            _compute_momentum, include_groups=False
        )
        stats["model"]["model_price_momentum"] = momentum

    # Price volatility (coefficient of variation)
    stats["model"]["model_price_cv"] = (
        stats["model"]["model_price_std"] / stats["model"]["model_price_mean"]
    ).fillna(0)

    # --- Item level ---
    item_grp = train_df.groupby("itemId")[TARGET]
    stats["item"] = pd.DataFrame(
        {
            "item_price_mean": item_grp.mean(),
            "item_price_std": item_grp.std().fillna(0),
            "item_price_median": item_grp.median(),
            "item_price_count": item_grp.count(),
        }
    )

    last_item_price = (
        train_df.sort_values(DATETIME_COL)
        .groupby("itemId")
        .last()[[TARGET]]
        .rename(columns={TARGET: "item_price_last"})
    )
    stats["item"] = stats["item"].join(last_item_price)

    # --- Shop level ---
    shop_grp = train_df.groupby("shopId")[TARGET]
    stats["shop"] = pd.DataFrame(
        {
            "shop_avg_price": shop_grp.mean(),
            "shop_price_std": shop_grp.std().fillna(0),
            "shop_product_count": train_df.groupby("shopId")["modelId"].nunique(),
        }
    )

    # --- Category level ---
    cat_grp = train_df.groupby("cat_id")[TARGET]
    stats["category"] = pd.DataFrame(
        {
            "cat_avg_price": cat_grp.mean(),
            "cat_price_std": cat_grp.std().fillna(0),
        }
    )

    # --- Shop × Category level ---
    shop_cat_grp = train_df.groupby(["shopId", "cat_id"])[TARGET]
    stats["shop_cat"] = pd.DataFrame(
        {
            "shop_cat_avg_price": shop_cat_grp.mean(),
            "shop_cat_price_std": shop_cat_grp.std().fillna(0),
            "shop_cat_count": shop_cat_grp.count(),
        }
    )

    return stats


def merge_entity_features(
    df: pd.DataFrame, stats: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Merge pre-computed entity stats onto a dataframe."""

    # Model-level
    if "model" in stats:
        df = df.merge(
            stats["model"], left_on="modelId", right_index=True, how="left"
        )

    # Item-level (fallback for cold-start models)
    if "item" in stats:
        df = df.merge(
            stats["item"], left_on="itemId", right_index=True, how="left"
        )

    # Shop-level
    if "shop" in stats:
        df = df.merge(
            stats["shop"], left_on="shopId", right_index=True, how="left"
        )

    # Category-level
    if "category" in stats and "cat_id" in df.columns:
        df = df.merge(
            stats["category"], left_on="cat_id", right_index=True, how="left"
        )

    # Shop × Category level
    if "shop_cat" in stats and "cat_id" in df.columns:
        df = df.merge(
            stats["shop_cat"],
            left_on=["shopId", "cat_id"],
            right_index=True,
            how="left",
        )

    return df


# ============================================================
# Derived Features
# ============================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features from existing columns."""

    # Discount depth
    if "priceBeforeDiscount" in df.columns:
        df["discount_depth"] = np.where(
            df["priceBeforeDiscount"] > 0,
            (df["priceBeforeDiscount"] - df.get(TARGET, 0))
            / df["priceBeforeDiscount"],
            0,
        )
        df["discount_depth"] = df["discount_depth"].clip(0, 1).fillna(0)

    # Has promotion
    if "promotionId" in df.columns:
        df["has_promotion"] = (df["promotionId"] > 0).astype(int)

    # Shop engagement score
    if "shop_rating" in df.columns and "shop_follower_count" in df.columns:
        df["shop_engagement_score"] = df["shop_rating"] * np.log1p(
            df["shop_follower_count"]
        )

    # Price relative to category average
    if "model_price_mean" in df.columns and "cat_avg_price" in df.columns:
        df["price_vs_cat_ratio"] = (
            df["model_price_mean"] / df["cat_avg_price"]
        ).fillna(1)

    # Price relative to shop average
    if "model_price_mean" in df.columns and "shop_avg_price" in df.columns:
        df["price_vs_shop_ratio"] = (
            df["model_price_mean"] / df["shop_avg_price"]
        ).fillna(1)

    # Item price range
    if "item_price_min" in df.columns and "item_price_max" in df.columns:
        df["item_price_range"] = df["item_price_max"] - df["item_price_min"]
        df["item_price_range_ratio"] = np.where(
            df["item_price_min"] > 0,
            df["item_price_max"] / df["item_price_min"],
            1,
        )

    # Log transforms for skewed columns
    for col in ["shop_follower_count", "total_rating_count", "cmt_count"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    return df


# ============================================================
# Brand Encoding
# ============================================================
def encode_brand(
    train_df: pd.DataFrame, target_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Frequency-encode brand column.
    Rare brands (< 10 occurrences) are grouped as 'rare'.
    """
    if "brand" not in train_df.columns:
        return train_df, target_df

    # Compute brand frequencies from training data
    brand_counts = train_df["brand"].value_counts()
    rare_brands = brand_counts[brand_counts < 10].index

    train_df["brand_clean"] = train_df["brand"].replace(rare_brands, "rare")
    target_df["brand_clean"] = target_df["brand"].apply(
        lambda x: x if x in brand_counts.index and x not in rare_brands else "rare"
        if pd.notna(x) else "rare"
    )

    # Frequency encoding
    brand_freq = train_df["brand_clean"].value_counts(normalize=True).to_dict()
    train_df["brand_freq"] = train_df["brand_clean"].map(brand_freq).fillna(0)
    target_df["brand_freq"] = target_df["brand_clean"].map(brand_freq).fillna(0)

    # Label encoding for LightGBM categorical
    le = LabelEncoder()
    train_df["brand_encoded"] = le.fit_transform(train_df["brand_clean"])

    # Handle unseen brands in target
    brand_map = dict(zip(le.classes_, le.transform(le.classes_)))
    target_df["brand_encoded"] = (
        target_df["brand_clean"]
        .map(brand_map)
        .fillna(brand_map.get("rare", 0))
        .astype(int)
    )

    return train_df, target_df


# ============================================================
# Full Pipeline
# ============================================================
def get_feature_columns(for_product_model: bool = False) -> list[str]:
    """Return the list of feature columns used for modelling."""
    base_features = [
        # Temporal
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "hour",
        "days_since_start",
        # Shop metadata
        "shop_rating",
        "shop_response_rate",
        "shop_follower_count",
        "log_shop_follower_count",
        "shop_engagement_score",
        # Item metadata
        "review_rating",
        "total_rating_count",
        "log_total_rating_count",
        "cmt_count",
        "log_cmt_count",
        "item_price_min",
        "item_price_max",
        "item_price_range",
        "item_price_range_ratio",
        # Discount
        "raw_discount",
        "show_discount",
        "discount_depth",
        "has_promotion",
        "priceBeforeDiscount",
        # Boolean
        "is_free_shipping",
        "is_pre_order",
        "is_official_shop",
        "is_verified",
        "is_preferred_plus_seller",
        # Brand
        "brand_encoded",
        "brand_freq",
        # Category
        "cat_id",
        "cat_avg_price",
        "cat_price_std",
    ]

    entity_features = [
        # Model-level
        "model_price_mean",
        "model_price_std",
        "model_price_median",
        "model_price_min",
        "model_price_max",
        "model_price_last",
        "model_price_count",
        "model_price_momentum",
        "model_price_cv",
        # Item-level
        "item_price_mean",
        "item_price_std",
        "item_price_median",
        "item_price_last",
        "item_price_count",
        # Shop-level
        "shop_avg_price",
        "shop_price_std",
        "shop_product_count",
        # Shop × Category
        "shop_cat_avg_price",
        "shop_cat_price_std",
        "shop_cat_count",
        # Relative
        "price_vs_cat_ratio",
        "price_vs_shop_ratio",
    ]

    if for_product_model:
        return base_features + entity_features
    return base_features + entity_features  # Both use entity features
