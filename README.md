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
*Note: All final price predictions are rounded to the nearest integer as IDR prices do not use decimals.*

### 4. Full Pipeline
```bash
python main.py --mode all
```
Runs both training + inference.

---

## 📈 Results & Findings

### Validation Metrics
Based on our simulated outage day validation (holding out the last day and using 100 anchor samples), the Product Model with calibration yields the best performance.

| Approach | MAE (IDR) | RMSE (IDR) | MAPE (%) | Median AE (IDR) |
|----------|-----------|------------|----------|-----------------|
| Tier 1 — Global (no calibration) | 151,635 | 1,251,789 | 0.48% | 10,785 |
| Tier 1 — Global (calibrated) | 136,939 | 1,241,876 | 0.44% | 1,499 |
| Tier 2 — Product (no calibration)| 121,412 | 1,240,025 | 0.39% | 4,829 |
| **Tier 2 — Product (calibrated)**| **116,330** | **1,240,285** | **0.36%** | **1,070** |

### Analysis & Insights
- **Tier 2 Outperforms Tier 1:** As expected, incorporating shop and product-specific history heavily improves predictions. The Product Model significantly drops the MAE (by ~30,000 IDR compared to the Global Model).
- **Anchor Calibration is Crucial:** For both tiers, using the 100 anchor samples significantly improves the Median Absolute Error (AE). For the Product Model, Median AE drops from 4,829 IDR down to 1,070 IDR after calibration.
- **When to Use Each Approach:**
  - **Tier 2 (Product Model)** should be the primary model in production for products that have sufficient historical data (e.g., ≥5 observations).
  - **Tier 1 (Global Model)** serves as a robust fallback for cold-start entities (new shops or products) where historical stats are unavailable.

---

## 🔧 Feature Engineering

### Temporal Features
- `day_of_week`, `day_of_month`, `month`, `is_weekend`, `hour`
- `days_since_start` — numeric progression to capture inflation or overall trend.

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

## 📝 Preprocessing & Output Formatting Notes

- Prices are in IDR smallest unit (e.g., 41900000 = Rp 41,900,000).
- **Rounding:** In the final output (`predictions_final.csv`, `predictions_global.csv`, `predictions_product.csv`), all predicted prices are rounded to the nearest whole integer (`int`). This is aligned with the original format of the e-commerce data to avoid confusing decimal outputs like `7485332.982379999`.
- `stock` and `normal_stock` dropped (99% null).
- Target is log-transformed during training (`np.log1p`) → exponentiated for final predictions (`np.expm1`).
- Outliers capped at 1st/99th percentile.

---

## 🔁 Reproducibility & Structure

- **Random seed**: 42 (set in `src/config.py`)
- **Python**: 3.10+
- **Dependencies**: Pinned in `requirements.txt`
- All experiments are deterministic given the same seed.

### 📁 Project Structure

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
├── outputs/                            # Prediction CSVs (predictions_final.csv)
├── main.py                             # CLI entry point
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```
