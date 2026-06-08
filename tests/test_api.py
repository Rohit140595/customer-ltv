"""
Integration tests for the FastAPI serving layer.

Uses FastAPI's TestClient — no live server needed, runs in CI.
load_model is patched with a minimal fake artifact so tests run without
a trained model on disk (models/ is gitignored and not available in CI).

Tests cover:
  - /health returns 200 with correct schema
  - /predict returns valid segment for a well-formed request
  - /predict rejects malformed requests (missing fields, bad values)
  - /predict handles a customer with no pre-cutoff orders
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from src.api import app

# ── Fake model fixture ────────────────────────────────────────────────────────

FEATURES = [
    "recency", "frequency", "monetary", "avg_order_value",
    "unique_categories", "unique_sellers", "avg_review_score",
    "avg_items_per_order", "avg_freight_ratio", "pct_installments",
    "has_voucher", "orders_last_30d", "spend_last_30d", "orders_last_60d",
    "spend_last_60d", "avg_order_gap_days", "is_repeat_customer",
    "avg_delivery_delay_days", "pct_late_deliveries",
    "avg_delivery_speed_days", "customer_state_freq",
]


def _make_fake_artifact() -> dict:
    """Build a minimal XGBoost artifact that mirrors the real model schema."""
    X = np.zeros((10, len(FEATURES)))
    y_binary = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_multi  = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    stage1 = xgb.XGBClassifier(n_estimators=1, max_depth=1, verbosity=0)
    stage1.fit(X, y_binary)

    stage2 = xgb.XGBClassifier(
        n_estimators=1, max_depth=1,
        objective="multi:softprob", num_class=3, verbosity=0,
    )
    stage2.fit(X, y_multi)

    return {
        "stage1_model":    stage1,
        "stage2_model":    stage2,
        "stage1_features": FEATURES,
        "stage2_features": FEATURES,
        "threshold":       0.9,          # high threshold → most requests → Non-returner
        "tier_labels":     {0: "Low", 1: "Mid", 2: "High"},
    }


@pytest.fixture(scope="module")
def client():
    """TestClient with load_model patched in both api and predict — no model file needed."""
    fake = _make_fake_artifact()
    with patch("src.api.load_model", return_value=fake), \
         patch("src.predict.load_model", return_value=fake):
        with TestClient(app) as c:
            yield c


# ── Shared test data ──────────────────────────────────────────────────────────

VALID_ORDER = {
    "order_id":                      "order-001",
    "order_purchase_timestamp":      "2018-03-15T10:30:00",
    "payment_value":                 149.90,
    "payment_installments":          3,
    "order_item_count":              2,
    "freight_value":                 12.50,
    "review_score":                  5.0,
    "customer_state":                "SP",
    "order_delivered_customer_date": "2018-03-22T14:00:00",
    "order_estimated_delivery_date": "2018-03-25T00:00:00",
}

VALID_REQUEST = {
    "customer_unique_id": "test-customer-001",
    "cutoff_date":        "2018-07-01T00:00:00",
    "orders":             [VALID_ORDER],
}

VALID_SEGMENTS = {"Non-returner", "Low", "Mid", "High"}


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_schema(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["loaded"], bool)


# ── Predict — happy path ──────────────────────────────────────────────────────

def test_predict_returns_200(client):
    assert client.post("/predict", json=VALID_REQUEST).status_code == 200


def test_predict_response_schema(client):
    body = client.post("/predict", json=VALID_REQUEST).json()
    assert "customer_unique_id"      in body
    assert "will_return_probability" in body
    assert "customer_segment"        in body
    assert "cutoff_date"             in body


def test_predict_customer_id_passthrough(client):
    body = client.post("/predict", json=VALID_REQUEST).json()
    assert body["customer_unique_id"] == "test-customer-001"


def test_predict_probability_in_range(client):
    prob = client.post("/predict", json=VALID_REQUEST).json()["will_return_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_segment_is_valid(client):
    segment = client.post("/predict", json=VALID_REQUEST).json()["customer_segment"]
    assert segment in VALID_SEGMENTS


def test_predict_multiple_orders(client):
    """Customer with 3 orders should score without error."""
    second = {**VALID_ORDER, "order_id": "order-002",
              "order_purchase_timestamp": "2018-05-01T09:00:00",
              "payment_value": 89.90}
    third  = {**VALID_ORDER, "order_id": "order-003",
              "order_purchase_timestamp": "2018-06-10T15:00:00",
              "payment_value": 210.00, "review_score": 4.0}
    request  = {**VALID_REQUEST, "orders": [VALID_ORDER, second, third]}
    response = client.post("/predict", json=request)
    assert response.status_code == 200
    assert response.json()["customer_segment"] in VALID_SEGMENTS


# ── Predict — validation errors ───────────────────────────────────────────────

def test_predict_rejects_empty_orders(client):
    response = client.post("/predict", json={**VALID_REQUEST, "orders": []})
    assert response.status_code == 422


def test_predict_rejects_negative_payment(client):
    bad = {**VALID_ORDER, "payment_value": -10.0}
    assert client.post("/predict", json={**VALID_REQUEST, "orders": [bad]}).status_code == 422


def test_predict_rejects_invalid_review_score(client):
    bad = {**VALID_ORDER, "review_score": 6.0}
    assert client.post("/predict", json={**VALID_REQUEST, "orders": [bad]}).status_code == 422


def test_predict_rejects_missing_timestamp(client):
    """order_purchase_timestamp is required."""
    bad = {k: v for k, v in VALID_ORDER.items() if k != "order_purchase_timestamp"}
    assert client.post("/predict", json={**VALID_REQUEST, "orders": [bad]}).status_code == 422


# ── Predict — edge cases ──────────────────────────────────────────────────────

def test_predict_orders_after_cutoff_returns_422(client):
    """All orders after cutoff — no pre-cutoff history available."""
    future = {**VALID_ORDER, "order_purchase_timestamp": "2018-08-01T00:00:00"}
    response = client.post("/predict", json={**VALID_REQUEST, "orders": [future]})
    assert response.status_code == 422


def test_predict_optional_fields_can_be_null(client):
    """review_score, delivery dates, and customer_state are all optional."""
    minimal = {
        "order_id":                 "order-min",
        "order_purchase_timestamp": "2018-04-01T00:00:00",
        "payment_value":            50.0,
        "payment_installments":     1,
        "order_item_count":         1,
        "freight_value":            5.0,
    }
    response = client.post("/predict", json={**VALID_REQUEST, "orders": [minimal]})
    assert response.status_code == 200
    assert response.json()["customer_segment"] in VALID_SEGMENTS
