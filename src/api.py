"""
FastAPI serving layer for real-time customer LTV scoring.

Endpoints:
  POST /predict   — score a single customer from their order history
  GET  /health    — liveness check (used by Docker / Kubernetes)

Design decisions:
  - Model loaded once at startup via lifespan context — avoids per-request disk I/O.
  - Pydantic schemas enforce types and ranges at the boundary — bad inputs are
    rejected before reaching feature engineering.
  - predict_single mirrors the training feature pipeline exactly to prevent
    train-serve skew.
  - customer_segment returns one of: Non-returner / Low / Mid / High.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import load_model
from src.predict import predict_single


# ── Request / response schemas ────────────────────────────────────────────────

class OrderRecord(BaseModel):
    """One order in a customer's purchase history."""
    order_id:                      str
    order_purchase_timestamp:      datetime
    payment_value:                 float   = Field(..., ge=0)
    payment_installments:          int     = Field(..., ge=1)
    order_item_count:              int     = Field(..., ge=1)
    freight_value:                 float   = Field(..., ge=0)
    review_score:                  Optional[float]    = Field(None, ge=1, le=5)
    customer_state:                Optional[str]      = None
    order_delivered_customer_date: Optional[datetime] = None
    order_estimated_delivery_date: Optional[datetime] = None


class PredictRequest(BaseModel):
    """Request body: stable customer ID + pre-cutoff order history."""
    customer_unique_id: str
    orders:             list[OrderRecord] = Field(..., min_length=1)
    cutoff_date:        datetime


class PredictResponse(BaseModel):
    """Response body: return probability and segment assignment."""
    customer_unique_id:      str
    will_return_probability: float
    customer_segment:        str     # Non-returner | Low | Mid | High
    cutoff_date:             datetime


class HealthResponse(BaseModel):
    status: str
    loaded: bool


# ── App lifecycle ─────────────────────────────────────────────────────────────

_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release resources on shutdown."""
    _store["artifact"] = load_model("models/ltv_model.pkl")
    print(f"Model loaded — threshold={_store['artifact']['threshold']:.4f}")
    yield
    _store.clear()


app = FastAPI(
    title="Customer LTV Prediction API",
    description="Two-stage LTV scoring: predicts return probability and spend tier.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — returns 200 when the API is up and model is loaded."""
    return HealthResponse(
        status="ok",
        loaded="artifact" in _store,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Score a single customer's 90-day LTV from their order history.

    The caller provides all pre-cutoff orders. Feature engineering mirrors
    the training pipeline exactly to prevent train-serve skew.

    Returns:
        customer_segment: one of Non-returner / Low / Mid / High.
        will_return_probability: Stage 1 model output (0–1).
    """
    if "artifact" not in _store:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    orders = [o.model_dump() for o in request.orders]
    cutoff = pd.Timestamp(request.cutoff_date)

    try:
        result = predict_single(
            customer_id=request.customer_unique_id,
            orders=orders,
            cutoff_date=cutoff,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        customer_unique_id=request.customer_unique_id,
        will_return_probability=result["will_return_probability"],
        customer_segment=result["customer_segment"],
        cutoff_date=request.cutoff_date,
    )
