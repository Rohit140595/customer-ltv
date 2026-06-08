"""
Inference pipeline for customer LTV prediction.

Mirrors the training feature engineering pipeline so there is no train-serve skew.

Two-stage prediction:
  Stage 1 — return classifier: will this customer place another order?
  Stage 2 — spend tier classifier: if yes, which tier (Low / Mid / High)?

Note on customer_state_freq: in training this feature encodes how many customers
share the same state across the full dataset. At serving time only the single
customer's history is available, so the feature is set to NaN and handled natively
by XGBoost. In production this would be resolved via a feature store that holds
training-time state frequencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_feature_matrix
from src.model import load_model


def _orders_to_abt(orders: list[dict], customer_id: str) -> pd.DataFrame:
    """
    Convert a list of raw order dicts to a mini-ABT matching the training schema.

    Column names and types must match what build_feature_matrix expects.
    Fields not available at inference time (unique_categories, unique_sellers,
    has_voucher) are set to conservative defaults.

    Args:
        orders:      List of order dicts from the API request.
        customer_id: Stable customer identifier.

    Returns:
        DataFrame with one row per order, ready for build_feature_matrix.
    """
    rows = []
    for o in orders:
        rows.append({
            "customer_unique_id":            customer_id,
            "order_id":                      o.get("order_id", ""),
            "order_purchase_timestamp":      pd.Timestamp(o["order_purchase_timestamp"]),
            "payment_value":                 float(o.get("payment_value", 0)),
            "max_installments":              int(o.get("payment_installments", 1)),
            "item_count":                    int(o.get("order_item_count", 1)),
            "total_price":                   float(o.get("payment_value", 0)),
            "total_freight":                 float(o.get("freight_value", 0)),
            "unique_categories":             1,   # unknown at API time
            "unique_sellers":                1,   # unknown at API time
            "has_voucher":                   0,   # conservative default
            "review_score":                  float(o["review_score"]) if o.get("review_score") is not None else float("nan"),
            "order_delivered_customer_date": pd.Timestamp(o["order_delivered_customer_date"])
                                             if o.get("order_delivered_customer_date") else pd.NaT,
            "order_estimated_delivery_date": pd.Timestamp(o["order_estimated_delivery_date"])
                                             if o.get("order_estimated_delivery_date") else pd.NaT,
            "customer_state":                o.get("customer_state", "SP"),
        })
    return pd.DataFrame(rows)


def predict_single(
    customer_id: str,
    orders: list[dict],
    cutoff_date: pd.Timestamp,
    model_path: str = "models/ltv_model.pkl",
) -> dict:
    """
    Score a single customer's LTV from their order history.

    Pipeline:
      1. Convert order history to mini-ABT.
      2. Build feature matrix (same functions as training).
      3. Stage 1 — predict return probability.
      4. If probability >= threshold, Stage 2 — predict spend tier.

    Args:
        customer_id:  Stable customer identifier (passed through to response).
        orders:       List of raw order dicts (pre-cutoff only).
        cutoff_date:  Feature observation cutoff — same semantics as training.
        model_path:   Path to saved model artifact.

    Returns:
        Dict with will_return_probability and customer_segment.

    Raises:
        ValueError: If no orders are found before cutoff_date.
    """
    artifact = load_model(model_path)

    abt = _orders_to_abt(orders, customer_id)

    # build_feature_matrix filters to pre-cutoff internally
    features = build_feature_matrix(abt, cutoff_date)

    if features.empty:
        raise ValueError(f"No pre-cutoff orders found for customer '{customer_id}'.")

    # Stage 1 — return classifier
    s1_cols = artifact["stage1_features"]
    X1      = features.reindex(columns=s1_cols, fill_value=np.nan)
    prob    = float(artifact["stage1_model"].predict_proba(X1)[0, 1])
    threshold = artifact["threshold"]

    if prob < threshold:
        return {
            "will_return_probability": round(prob, 4),
            "customer_segment":        "Non-returner",
        }

    # Stage 2 — spend tier classifier (returners only)
    s2_cols  = artifact["stage2_features"]
    X2       = features.reindex(columns=s2_cols, fill_value=np.nan)
    tier     = int(artifact["stage2_model"].predict(X2)[0])
    segment  = artifact["tier_labels"][tier]

    return {
        "will_return_probability": round(prob, 4),
        "customer_segment":        segment,
    }
