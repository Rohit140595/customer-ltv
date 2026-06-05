"""
Data loading and merging layer for the Olist Brazilian E-commerce dataset.

Responsibilities:
  - Load raw CSVs from data/raw/
  - Merge the 9 Olist tables into a single analytical base table (ABT)
  - Compute the LTV target: total spend per customer in the next `horizon_days`
  - Split into train/test using a chronological cutoff (no random shuffling)

This module owns the raw data — nothing above this layer should read CSVs directly.
Feature engineering happens in features.py, not here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the central YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_tables(raw_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all 9 Olist CSV files from raw_dir.

    Returns a dict keyed by table name:
        orders, order_items, order_payments, order_reviews,
        customers, products, product_category, sellers, geolocation
    """
    raise NotImplementedError


def build_analytical_base_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge Olist tables into one row-per-order analytical base table.

    Joins:
      orders ── order_items ── products ── product_category
        └────── order_payments
        └────── order_reviews
        └────── customers

    Returns:
        DataFrame with one row per order, all relevant columns joined.
    """
    raise NotImplementedError


def compute_ltv_target(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Compute the LTV target: total spend per customer in the
    `horizon_days` window after `cutoff_date`.

    Customers with no orders in the horizon window get target = 0.

    Args:
        abt:          Analytical base table (one row per order).
        cutoff_date:  Feature observation cutoff — only orders before this
                      date are used for features; orders after define the target.
        horizon_days: Length of the prediction window in days.

    Returns:
        DataFrame with one row per customer and a `ltv_target` column.
    """
    raise NotImplementedError


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    date_col: str = "order_purchase_timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split customers into train/test by their first order date.

    The earliest `train_frac` of customers (by first order date) go to train;
    the rest go to test. No random shuffling — preserves temporal ordering.

    Returns:
        (train_df, test_df)
    """
    raise NotImplementedError
