"""
Unit tests for feature engineering functions.

Tests focus on three properties critical for production LTV features:
  1. Leak-free : no future orders appear in any feature value
  2. Correct aggregation : RFM values match hand-calculated expectations
  3. Schema stability : output always has expected columns and types
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features import compute_rfm_features, compute_temporal_features


@pytest.fixture
def sample_orders() -> pd.DataFrame:
    """Minimal order history for two customers across 6 months."""
    cutoff = datetime(2018, 6, 1)
    return pd.DataFrame([
        # customer_a — 3 orders before cutoff, 1 after (should not count)
        {"customer_unique_id": "customer_a", "order_purchase_timestamp": cutoff - timedelta(days=90), "payment_value": 100.0},
        {"customer_unique_id": "customer_a", "order_purchase_timestamp": cutoff - timedelta(days=30), "payment_value": 200.0},
        {"customer_unique_id": "customer_a", "order_purchase_timestamp": cutoff - timedelta(days=5),  "payment_value": 50.0},
        {"customer_unique_id": "customer_a", "order_purchase_timestamp": cutoff + timedelta(days=10), "payment_value": 999.0},  # future — must not leak
        # customer_b — 1 order before cutoff
        {"customer_unique_id": "customer_b", "order_purchase_timestamp": cutoff - timedelta(days=60), "payment_value": 75.0},
    ])


def test_rfm_no_future_leak(sample_orders):
    """Future orders (after cutoff) must never appear in RFM features."""
    cutoff = pd.Timestamp("2018-06-01")
    result = compute_rfm_features(sample_orders, cutoff_date=cutoff)

    customer_a = result[result["customer_unique_id"] == "customer_a"].iloc[0]
    # monetary should be 100 + 200 + 50 = 350, NOT 350 + 999
    assert customer_a["monetary"] == pytest.approx(350.0)


def test_rfm_frequency(sample_orders):
    """Frequency should count orders strictly before cutoff."""
    cutoff = pd.Timestamp("2018-06-01")
    result = compute_rfm_features(sample_orders, cutoff_date=cutoff)

    customer_a = result[result["customer_unique_id"] == "customer_a"].iloc[0]
    assert customer_a["frequency"] == 3  # not 4


def test_rfm_recency(sample_orders):
    """Recency should be days since the most recent pre-cutoff order."""
    cutoff = pd.Timestamp("2018-06-01")
    result = compute_rfm_features(sample_orders, cutoff_date=cutoff)

    customer_a = result[result["customer_unique_id"] == "customer_a"].iloc[0]
    assert customer_a["recency"] == pytest.approx(5.0)  # last order was 5 days before cutoff


def test_rfm_output_schema(sample_orders):
    """Output must have one row per customer and expected columns."""
    cutoff = pd.Timestamp("2018-06-01")
    result = compute_rfm_features(sample_orders, cutoff_date=cutoff)

    assert set(result.columns) >= {"customer_unique_id", "recency", "frequency", "monetary"}
    assert len(result) == 2  # one row per customer
