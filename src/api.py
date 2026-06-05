"""
FastAPI serving layer for real-time customer LTV scoring.

Endpoints:
  POST /predict   — score a single customer from their order history
  GET  /health    — liveness check (used by Docker / Kubernetes)

Design decisions:
  - Model loaded once at startup via lifespan context manager — avoids
    per-request disk I/O.
  - Pydantic request/response schemas enforce types at the boundary — bad
    inputs are rejected before they reach feature engineering.
  - Feature pipeline mirrors training exactly (build_feature_matrix) to
    prevent train-serve skew.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_single
from src.model import load_model


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class OrderRecord(BaseModel):
    """A single order record for a customer."""
    order_id:                  str
    order_purchase_timestamp:  datetime
    payment_value:             float = Field(..., ge=0)
    payment_installments:      int   = Field(..., ge=1)
    order_item_count:          int   = Field(..., ge=1)
    freight_value:             float = Field(..., ge=0)
    product_category:          str | None = None
    seller_id:                 str | None = None
    review_score:              float | None = Field(None, ge=1, le=5)


class PredictRequest(BaseModel):
    """Request body: customer ID + their full order history."""
    customer_unique_id: str
    orders:             list[OrderRecord]
    cutoff_date:        datetime


class PredictResponse(BaseModel):
    """Response body: predicted LTV and segment assignment."""
    customer_unique_id:  str
    predicted_ltv_90d:   float
    customer_segment:    str   # High / Mid / Low
    cutoff_date:         datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── App lifecycle ─────────────────────────────────────────────────────────────

_model_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    model, feature_names = load_model("models/ltv_model.pkl")
    _model_store["model"]         = model
    _model_store["feature_names"] = feature_names
    print("Model loaded and ready.")
    yield
    _model_store.clear()


app = FastAPI(
    title="Customer LTV Prediction API",
    description="Predicts 90-day customer lifetime value from order history.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — returns 200 if the API is up and model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded="model" in _model_store,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Score a single customer's 90-day LTV from their order history.

    The caller provides all orders up to cutoff_date. Feature engineering
    mirrors the training pipeline exactly to prevent train-serve skew.
    """
    if "model" not in _model_store:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    orders_df   = pd.DataFrame([o.model_dump() for o in request.orders])
    cutoff_date = pd.Timestamp(request.cutoff_date)

    result = predict_single(
        customer_orders=orders_df,
        cutoff_date=cutoff_date,
    )

    return PredictResponse(
        customer_unique_id=request.customer_unique_id,
        predicted_ltv_90d=result["predicted_ltv_90d"],
        customer_segment=result["customer_segment"],
        cutoff_date=request.cutoff_date,
    )
