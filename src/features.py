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
    - avg_order_value: monetary / frequency.

    Args:
        abt:         Analytical base table (one row per order).
        cutoff_date: Only orders strictly before this date are used.
        customer_col, date_col, value_col: Column name overrides.

    Returns:
        DataFrame with one row per customer: recency, frequency, monetary,
        avg_order_value.
    """
    pre = abt[abt[date_col] < cutoff_date]

    rfm = pre.groupby(customer_col).agg(
        last_order_date = (date_col,  "max"),
        frequency       = (date_col,  "count"),
        monetary        = (value_col, "sum"),
    ).reset_index()

    rfm["recency"]          = (cutoff_date - rfm["last_order_date"]).dt.days
    rfm["avg_order_value"]  = rfm["monetary"] / rfm["frequency"]
    rfm = rfm.drop(columns=["last_order_date"])

    return rfm


def compute_behavioral_features(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    customer_col: str = "customer_unique_id",
) -> pd.DataFrame:
    """
    Compute behavioral features capturing purchase diversity and quality signals.

    Adds:
        unique_categories   : distinct product categories purchased before cutoff.
        unique_sellers      : distinct sellers transacted with before cutoff.
        avg_review_score    : mean review score across all orders before cutoff.
        pct_installments    : share of orders paid in more than 1 installment.
        avg_items_per_order : average number of items per order.
        avg_freight_ratio   : average (freight / total_price) per order — proxy
                              for purchase distance / item size.
        has_voucher         : 1 if the customer ever used a voucher.

    Args:
        abt:         Analytical base table (one row per order).
        cutoff_date: Only orders strictly before this date are used.
        customer_col: Column identifying the customer.

    Returns:
        DataFrame with one row per customer and behavioral feature columns.
    """
    pre = abt[abt["order_purchase_timestamp"] < cutoff_date].copy()

    # Freight ratio per order — avoid division by zero
    pre["freight_ratio"] = pre["total_freight"] / pre["total_price"].replace(0, np.nan)

    behavioral = pre.groupby(customer_col).agg(
        unique_categories   = ("unique_categories",  "sum"),
        unique_sellers      = ("unique_sellers",      "sum"),
        avg_review_score    = ("review_score",        "mean"),
        avg_items_per_order = ("item_count",          "mean"),
        avg_freight_ratio   = ("freight_ratio",       "mean"),
        pct_installments    = ("max_installments",    lambda x: (x > 1).mean()),
        has_voucher         = ("has_voucher",         "max"),
    ).reset_index()

    return behavioral


def compute_temporal_features(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    customer_col: str = "customer_unique_id",
    date_col: str = "order_purchase_timestamp",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute time-windowed purchase velocity and timing features.

    For each window (default: 30, 60 days before cutoff), counts:
        orders_last_{n}d : number of orders in the last n days.
        spend_last_{n}d  : total spend in the last n days.

    Also adds:
        avg_order_gap_days  : mean days between consecutive orders.
                              NaN for one-time buyers (only one order).
        is_repeat_customer  : 1 if the customer has more than one order.

    Args:
        abt:         Analytical base table (one row per order).
        cutoff_date: Reference point for all window calculations.
        customer_col, date_col: Column name overrides.
        windows:     Lookback windows in days (default [30, 60]).

    Returns:
        DataFrame with one row per customer and temporal feature columns.
    """
    if windows is None:
        windows = [30, 60]

    pre = abt[abt[date_col] < cutoff_date]

    # Base: all unique customers in pre-cutoff data
    customers = pre[[customer_col]].drop_duplicates()

    # Velocity features per window
    for n in windows:
        window_start = cutoff_date - pd.Timedelta(days=n)
        recent = pre[pre[date_col] >= window_start]
        agg = recent.groupby(customer_col).agg(
            **{f"orders_last_{n}d": ("order_id", "count"),
               f"spend_last_{n}d":  ("payment_value", "sum")}
        ).reset_index()
        customers = customers.merge(agg, on=customer_col, how="left")

    # Fill customers with no orders in a window with 0
    for n in windows:
        customers[f"orders_last_{n}d"] = customers[f"orders_last_{n}d"].fillna(0).astype(int)
        customers[f"spend_last_{n}d"]  = customers[f"spend_last_{n}d"].fillna(0.0)

    # Average gap between consecutive orders (NaN for one-time buyers)
    sorted_pre = pre.sort_values([customer_col, date_col])
    sorted_pre["prev_order_date"] = sorted_pre.groupby(customer_col)[date_col].shift(1)
    sorted_pre["order_gap_days"]  = (
        sorted_pre[date_col] - sorted_pre["prev_order_date"]
    ).dt.days

    gap = (
        sorted_pre.dropna(subset=["order_gap_days"])
        .groupby(customer_col)["order_gap_days"]
        .mean()
        .reset_index()
        .rename(columns={"order_gap_days": "avg_order_gap_days"})
    )
    customers = customers.merge(gap, on=customer_col, how="left")

    # is_repeat_customer
    freq = pre.groupby(customer_col)["order_id"].count().reset_index()
    freq.columns = [customer_col, "_freq"]
    customers = customers.merge(freq, on=customer_col, how="left")
    customers["is_repeat_customer"] = (customers["_freq"] > 1).astype(int)
    customers = customers.drop(columns=["_freq"])

    return customers


def build_feature_matrix(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix by joining all feature families.

    Steps:
      1. RFM features (recency, frequency, monetary, avg_order_value)
      2. Behavioral features (diversity, review, payment patterns)
      3. Temporal features (velocity windows, order gap, repeat flag)

    All features are computed as of cutoff_date — leak-free by construction.

    Args:
        abt:         Analytical base table (one row per order).
        cutoff_date: Feature observation cutoff date.

    Returns:
        DataFrame with one row per customer and all engineered features.
    """
    rfm        = compute_rfm_features(abt, cutoff_date)
    behavioral = compute_behavioral_features(abt, cutoff_date)
    temporal   = compute_temporal_features(abt, cutoff_date)

    features = (
        rfm
        .merge(behavioral, on="customer_unique_id", how="left")
        .merge(temporal,   on="customer_unique_id", how="left")
    )

    print(f"Feature matrix: {features.shape[0]:,} customers × {features.shape[1]} features")
    return features
