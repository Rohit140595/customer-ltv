# Customer LTV Prediction — Olist E-Commerce Dataset

End-to-end two-stage LTV pipeline built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Results

| Metric | Score |
|---|---|
| **Stage 2 Weighted F1** | **0.479** |
| Stage 1 PR-AUC | 0.0052 (1.7× random baseline) |
| Stage 1 ROC-AUC | 0.514 |

**Stage 1** (return classifier) is fundamentally constrained by a 0.3% return rate — 97% of Olist customers never place a second order. Delivery experience features (delay, speed, late rate) were the only signals that lifted Stage 1 above random, confirmed via ablation across feature sets up to 34 features.

**Stage 2** (spend tier) operates on confirmed returners only (~280 customers) and is the more actionable output — product teams target High/Mid/Low segments with differentiated retention offers.

## Project Structure

```
customer-ltv/
├── config.yaml                     # Central config: paths, model params, split settings
├── data/
│   ├── raw/                        # Olist CSVs (not tracked — download from Kaggle)
│   └── processed/                  # abt.parquet (not tracked — regenerable)
├── models/                         # Trained artifacts (not tracked — regenerable)
│   ├── ltv_model.pkl               # Both stage models + metadata
│   ├── shap_stage1.json            # SHAP feature selection cache (Stage 1)
│   └── shap_stage2.json            # SHAP feature selection cache (Stage 2)
├── notebooks/
│   └── model_training.ipynb        # End-to-end training pipeline with MLflow logging
├── src/
│   ├── data.py                     # Load Olist tables → ABT → targets → split
│   ├── features.py                 # Leak-free feature engineering (RFM + behavioral + temporal + delivery)
│   ├── model.py                    # Two-stage training: SHAP selection, Optuna tuning, evaluation
│   ├── predict.py                  # Inference pipeline (mirrors training exactly)
│   └── api.py                      # FastAPI serving layer
├── tests/
│   └── test_api.py                 # 14 integration tests (CI, no model file required)
├── Dockerfile                      # Multi-stage build
├── requirements.txt
└── requirements-dev.txt
```

## Two-Stage Architecture

```
Customer order history
        │
        ▼
┌───────────────────┐
│  Feature matrix   │  RFM + behavioral + temporal + delivery
│  (21 features,    │  computed as of cutoff_date — leak-free
│   leak-free)      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    Stage 1        │  XGBoost binary classifier
│  Return           │  Will this customer place another order
│  Classifier       │  in the next 90 days?
└────────┬──────────┘
         │
    prob < threshold ──→  Non-returner  (stop)
         │
    prob ≥ threshold
         │
         ▼
┌───────────────────┐
│    Stage 2        │  XGBoost 3-class classifier
│  Spend Tier       │  Low / Mid / High spend tier
│  Classifier       │  (tertiles of training-set returners)
└────────┬──────────┘
         │
         ▼
   customer_segment: High / Mid / Low
```

## Feature Engineering

All features are computed as of a `cutoff_date` — only orders strictly before the cutoff are used. Delivery experience features were identified via ablation as the key signal for return prediction.

| Family | Features |
|---|---|
| **RFM** | recency, frequency, monetary, avg_order_value |
| **Behavioral** | unique_categories, unique_sellers, avg_review_score, avg_items_per_order, avg_freight_ratio, pct_installments, has_voucher |
| **Temporal** | orders_last_30d, spend_last_30d, orders_last_60d, spend_last_60d, avg_order_gap_days, is_repeat_customer |
| **Delivery** | avg_delivery_delay_days, pct_late_deliveries, avg_delivery_speed_days, customer_state_freq |

## Training Pipeline

`notebooks/model_training.ipynb` orchestrates the full pipeline:

1. **Load** ABT from `data/processed/abt.parquet`
2. **Targets** — `will_return` (binary) and `spend_tier` (Low/Mid/High via `pd.qcut` tertiles)
3. **Features** — `build_feature_matrix` on all pre-cutoff orders
4. **Split** — chronological 80/20 by customer first-order date (no shuffling)
5. **Stage 1** — SHAP top-15 selection → Optuna (50 trials, TimeSeriesSplit) → XGBoost
6. **Stage 2** — SHAP top-10 selection → Optuna (50 trials, StratifiedKFold) → XGBoost
7. **Save** — both models + metadata to `models/ltv_model.pkl`
8. **Log** — all params, metrics, and artifacts to MLflow

## Setup

```bash
git clone https://github.com/Rohit140595/customer-ltv.git
cd customer-ltv
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name customer-ltv --display-name "Python (customer-ltv)"
```

Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSVs in `data/raw/`.

## Running

**Build the ABT (one-time):**
```bash
python -c "
from src.data import load_config, load_raw_tables, build_analytical_base_table
cfg = load_config()
abt = build_analytical_base_table(load_raw_tables(cfg['paths']['raw_dir']))
abt.to_parquet('data/processed/abt.parquet', index=False)
"
```

**Train the model:**
```bash
jupyter notebook notebooks/model_training.ipynb
# Select 'Python (customer-ltv)' kernel and run all cells (~10 min with n_trials=50)
```

**Serve the API:**
```bash
uvicorn src.api:app --reload
# POST to http://localhost:8000/predict
```

**Run tests:**
```bash
pytest tests/ -v
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_unique_id": "customer-123",
    "cutoff_date": "2018-07-01T00:00:00",
    "orders": [{
      "order_id": "order-001",
      "order_purchase_timestamp": "2018-03-15T10:30:00",
      "payment_value": 149.90,
      "payment_installments": 3,
      "order_item_count": 2,
      "freight_value": 12.50,
      "review_score": 5.0,
      "customer_state": "SP",
      "order_delivered_customer_date": "2018-03-22T14:00:00",
      "order_estimated_delivery_date": "2018-03-25T00:00:00"
    }]
  }'
```

Response:
```json
{
  "customer_unique_id": "customer-123",
  "will_return_probability": 0.0021,
  "customer_segment": "Non-returner",
  "cutoff_date": "2018-07-01T00:00:00"
}
```

## Key Design Decisions

- **Chronological split** — customers split by first-order date, not randomly. Behavioral features would leak future data into training with a random split.
- **TimeSeriesSplit for Stage 1** — each CV validation fold is strictly later than its training data, consistent with temporal ordering.
- **StratifiedKFold for Stage 2** — only 197 training returners; ensures all 3 tiers appear in each fold.
- **Seeded Optuna TPESampler** — `TPESampler(seed=42)` makes hyperparameter search fully reproducible.
- **SHAP feature selection** — top-N by mean |SHAP| value rather than split counts (biased toward high-cardinality features). Ablation showed top-15/10 outperforms using all features with this imbalance level.
- **Delivery features as key signal** — ablation across 17 → 34 features confirmed that delivery experience (delay, speed, late rate) is the only category that meaningfully lifts Stage 1 above random. Most customers have identical RFM profiles (frequency=1), so purchase-pattern features have near-zero variance.
