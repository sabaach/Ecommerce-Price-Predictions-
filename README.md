# MrScraper — Price Intelligence & Anomaly Detection

## 📋 Overview

A machine learning system that **reconstructs e-commerce product prices** during scraping outages using historical data and a small set of manually collected anchor prices.

### Problem Statement
When scraping infrastructure goes down, we lose visibility into marketplace pricing. This system predicts missing prices using:
- **Historical pricing data** (306K observations across 59 days)
- **100 anchor samples per day** (manually collected ground truth)

### Two-Tier Approach
| Tier | Model | Description |
|------|-------|-------------|
| **Tier 1** | Global Marketplace Model | Single LightGBM trained on all data, generalises across shops/items |
| **Tier 2** | Product Level Model | LightGBM with entity-specific features per (shopId, itemId, modelId) |

Both models use **multi-level anchor calibration** (per-model → per-item → per-shop → per-category → global) to adjust for day-specific price shifts.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train & Validate
```bash
python main.py --mode train
```
This will:
- Load and preprocess training data
- Hold out the last day as validation (simulating an outage)
- Train both Tier 1 and Tier 2 models
- Report metrics before/after anchor calibration
- Save models to `models/`

### 3. Run Inference
```bash
python main.py --mode predict
```
Generates predictions for the test file and saves to `outputs/predictions_final.csv`.

### 4. Full Pipeline
```bash
python main.py --mode all
```
Runs both training + inference.

---

## 📁 Project Structure

```
MrScrapperChallenge/
├── Files/                              # Data files
│   ├── ecommerce_price_prediction-train.csv
│   └── ecommerce_price_prediction-test-3-days.csv
├── src/
│   ├── config.py                       # Constants, paths, hyperparameters
│   ├── data_preprocessing.py           # Data loading & cleaning
│   ├── feature_engineering.py          # Feature creation pipeline
│   ├── model_global.py                 # Tier 1 — Global Model
│   ├── model_product.py               # Tier 2 — Product Model
│   ├── anchor_calibration.py           # Multi-level anchor calibration
│   ├── evaluate.py                     # Metrics & reporting
│   └── inference.py                    # Prediction pipeline
├── models/                             # Saved model artifacts
├── outputs/                            # Prediction CSVs
├── main.py                             # CLI entry point
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🔧 Feature Engineering

### Temporal Features
- `day_of_week`, `day_of_month`, `month`, `is_weekend`, `hour`
- `days_since_start` — numeric progression

### Entity-Level Historical Features
- **Model-level**: `price_mean`, `price_std`, `price_last`, `price_momentum`, `price_cv`
- **Item-level**: `price_mean`, `price_std`, `price_last`
- **Shop-level**: `avg_price`, `price_std`, `product_count`
- **Category-level**: `avg_price`, `price_std`

### Derived Features
- `discount_depth` = (priceBeforeDiscount - price) / priceBeforeDiscount
- `has_promotion` = promotionId > 0
- `shop_engagement_score` = shop_rating × log(shop_follower_count)
- `price_vs_cat_ratio`, `price_vs_shop_ratio` — relative positioning
- Log transforms for skewed columns

### Brand Encoding
- Frequency encoding + label encoding
- Rare brands (< 10 occurrences) grouped as "rare"

---

## 🎯 Anchor Calibration Strategy

The 100 anchor samples per day are used in a **hierarchical ratio-based calibration**:

1. **Per-model match**: If an anchor exists for the exact same `modelId`, use its ratio (actual/predicted)
2. **Per-item match**: Aggregate anchor ratios at `itemId` level
3. **Per-shop match**: Aggregate at `shopId` level
4. **Per-category match**: Aggregate at `cat_id` level (requires ≥2 anchors)
5. **Global ratio**: Median ratio across all anchors (ultimate fallback)

This approach allows the system to detect and correct for day-specific pricing shifts at the most granular level possible.

---

## 📊 Validation Methodology

- **Time-based split**: Hold out the last day entirely as validation set
- **Simulated anchor set**: Random 100 samples from the validation day
- **Metrics**: MAE, RMSE, MAPE reported before and after anchor calibration
- **Per-category breakdown**: Detailed metrics per product category

This mirrors the real test conditions where an entire day's data is missing.

---

## 🔍 Approach Comparison: When to Use Each

### Tier 1 — Global Model
✅ Better for:
- Products with sparse/no history (cold start)
- New shops or categories
- Quick baseline production deployment

### Tier 2 — Product Model
✅ Better for:
- Products with sufficient history (≥5 observations)
- Capturing shop-specific pricing strategies
- Maximum prediction accuracy on known products

### Production Recommendation
Use **Tier 2 as primary** with **Tier 1 as fallback** for cold-start entities. This is implemented via the hierarchical fallback in `model_product.py`.

---

## 🔁 Reproducibility

- **Random seed**: 42 (set in `src/config.py`)
- **Python**: 3.10+
- **Dependencies**: Pinned in `requirements.txt`
- All experiments are deterministic given the same seed

---

## 📝 Notes

- Prices are in IDR smallest unit (e.g., 41900000 = Rp 41,900,000)
- `stock` and `normal_stock` dropped (99% null)
- Target is log-transformed during training → expm1 for final predictions
- Outliers capped at 1st/99th percentile
