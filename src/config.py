"""
config.py — Global configuration, paths, and constants.
"""
import os

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Files")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

TRAIN_CSV = os.path.join(DATA_DIR, "ecommerce_price_prediction-train.csv")
TEST_CSV = os.path.join(DATA_DIR, "ecommerce_price_prediction-test-3-days.csv")

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Column definitions
# ============================================================
ID_COLS = ["shopId", "itemId", "modelId"]
TARGET = "price"
LOG_TARGET = "log_price"
DATETIME_COL = "capturedAt"

BOOL_COLS = [
    "is_free_shipping",
    "is_pre_order",
    "is_official_shop",
    "is_verified",
    "is_preferred_plus_seller",
]

DROP_COLS = ["stock", "normal_stock"]  # 99 % null

NUMERIC_COLS = [
    "priceBeforeDiscount",
    "promotionId",
    "cat_id",
    "raw_discount",
    "show_discount",
    "item_price_min",
    "item_price_max",
    "review_rating",
    "total_rating_count",
    "cmt_count",
    "shop_rating",
    "shop_response_rate",
    "shop_follower_count",
]

# ============================================================
# LightGBM hyperparameters (reasonable defaults)
# ============================================================
LGBM_PARAMS_GLOBAL = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LGBM_PARAMS_PRODUCT = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 3000,
    "early_stopping_rounds": 150,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "min_child_samples": 10,
}

# ============================================================
# Validation
# ============================================================
# Number of simulated anchor samples during validation
ANCHOR_SAMPLE_SIZE = 100
# Minimum observations to use entity-level stats
MIN_ENTITY_OBS = 5
