"""
Feature engineering for customer lifetime value prediction.

Three core feature families:
  - RFM          : Recency, Frequency, Monetary — the canonical LTV features.
  - Behavioral   : product diversity, payment patterns, review sentiment.
  - Temporal     : order timing patterns, purchase velocity across windows.

All features are computed as of a cutoff date — only orders strictly before
the cutoff are used, making every feature leak-free by construction.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_rfm_features(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    customer_col: str = "customer_unique_id",
    date_col: str = "order_purchase_timestamp",
    value_col: str = "payment_value",
) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary features per customer.

    - Recency  : days since the customer's most recent order before cutoff.
    - Frequency: total number of orders before cutoff.
    - Monetary : total spend before cutoff (sum of payment_value).

    Args:
        abt:         Analytical base table (one row per order).
        cutoff_date: Only orders strictly before this date are used.
        customer_col, date_col, value_col: Column name overrides.

    Returns:
        DataFrame with one row per customer and recency, frequency,
        monetary columns.
    """
    raise NotImplementedError


def compute_behavioral_features(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    customer_col: str = "customer_unique_id",
) -> pd.DataFrame:
    """
    Compute behavioral features capturing purchase diversity and quality signals.

    Adds:
        unique_categories    : distinct product categories purchased before cutoff.
        unique_sellers       : distinct sellers transacted with before cutoff.
        avg_review_score     : mean review score across all orders before cutoff.
        pct_installments     : share of orders paid in installments.
        avg_items_per_order  : average number of items per order.
        avg_freight_ratio    : average freight / payment value ratio.

    Args:
        abt:         Analytical base table (one row per order).
        cutoff_date: Only orders strictly before this date are used.
        customer_col: Column identifying the customer.

    Returns:
        DataFrame with one row per customer and behavioral feature columns.
    """
    raise NotImplementedError


def compute_temporal_features(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    customer_col: str = "customer_unique_id",
    date_col: str = "order_purchase_timestamp",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute time-windowed purchase velocity features.

    For each window (default: 30, 60, 90 days before cutoff), counts:
        orders_last_{n}d  : number of orders in the last n days.
        spend_last_{n}d   : total spend in the last n days.

    Also adds:
        avg_order_gap_days : mean number of days between consecutive orders.
        is_repeat_customer : 1 if the customer has more than one order.

    Args:
        abt:         Analytical base table (one row per order).
        cutoff_date: Reference point for all window calculations.
        customer_col, date_col: Column name overrides.
        windows:     List of lookback windows in days (default [30, 60, 90]).

    Returns:
        DataFrame with one row per customer and temporal feature columns.
    """
    raise NotImplementedError


def build_feature_matrix(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix by joining all feature families.

    Steps:
      1. RFM features (recency, frequency, monetary)
      2. Behavioral features (diversity, review, payment patterns)
      3. Temporal features (velocity windows, order gap)

    All features are computed as of cutoff_date — leak-free by construction.

    Args:
        abt:         Analytical base table.
        cutoff_date: Feature observation cutoff date.

    Returns:
        DataFrame with one row per customer and all engineered features.
    """
    raise NotImplementedError
