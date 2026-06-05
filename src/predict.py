"""
Inference pipeline for customer LTV prediction.

Mirrors the training feature engineering pipeline exactly so there is no
train-serve skew. Any change to features.py must be reflected here.

Used by:
  - api.py  : real-time single-customer scoring
  - batch   : offline scoring of all active customers (e.g. nightly job)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import build_feature_matrix
from src.model import load_model


def predict_single(
    customer_orders: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    model_path: str = "models/ltv_model.pkl",
) -> dict:
    """
    Score a single customer's LTV from their order history.

    Args:
        customer_orders: DataFrame of all orders for this customer (pre-cutoff).
        cutoff_date:     Feature observation cutoff — same as training cutoff.
        model_path:      Path to saved model artifact.

    Returns:
        Dict with predicted_ltv_90d and customer_segment (High/Mid/Low).
    """
    raise NotImplementedError


def predict_batch(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    model_path: str = "models/ltv_model.pkl",
) -> pd.DataFrame:
    """
    Score all customers in the analytical base table.

    Used for offline batch scoring (e.g. nightly job to refresh LTV scores
    for the entire customer base).

    Args:
        abt:         Full analytical base table.
        cutoff_date: Feature observation cutoff.
        model_path:  Path to saved model artifact.

    Returns:
        DataFrame with customer_unique_id, predicted_ltv_90d, customer_segment.
    """
    raise NotImplementedError


def assign_segment(predicted_ltv: float, thresholds: dict) -> str:
    """
    Assign a customer to High / Mid / Low LTV segment.

    Thresholds are computed from training set percentiles (e.g. top 20% = High).

    Args:
        predicted_ltv: Model output for a single customer.
        thresholds:    Dict with 'high' and 'low' cutoff values.

    Returns:
        'High', 'Mid', or 'Low'.
    """
    raise NotImplementedError
