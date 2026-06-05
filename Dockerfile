# Multi-stage build — keeps the final image lean by separating build from runtime.
#
# Stage 1 (builder): installs all dependencies into a virtual environment.
# Stage 2 (runtime): copies only the venv and source code — no build tools.
#
# Usage:
#   docker build -t customer-ltv .
#   docker run -p 8000:8000 customer-ltv

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into an isolated venv
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy venv from builder — no compiler or build tools in final image
COPY --from=builder /opt/venv /opt/venv

# Copy source code and config
COPY src/     src/
COPY config.yaml .

# Activate venv
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
