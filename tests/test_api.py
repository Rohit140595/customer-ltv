"""
Integration tests for the FastAPI serving layer.

Tests cover:
  - /health endpoint returns 200
  - /predict rejects malformed requests (missing fields, negative values)
  - /predict returns expected response schema
"""

import pytest
from fastapi.testclient import TestClient

# Tests will be implemented once src/api.py and src/predict.py are complete
