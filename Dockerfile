# syntax=docker/dockerfile:1

# ---- Builder stage: compile dependencies into a venv, discarded later ----
FROM python:3.11-slim AS builder

WORKDIR /app

# build-essential is only needed to compile a couple of native-extension
# wheels (e.g. tokenizers) if a prebuilt wheel isn't available for the
# target platform — it never ships in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---- Runtime stage: slim image, no compilers, no build cache ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# curl is only for the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY app/ ./app/

# Pre-populate the mock IT policy vector store at BUILD time. This makes
# the container work immediately on first `docker compose up` with no
# manual init step, and means the container needs no runtime network
# access just to download the embedding model. It DOES require the
# build machine to have internet access (to fetch all-MiniLM-L6-v2) —
# if you're building somewhere offline, build on a connected machine
# and push/load the resulting image instead.
RUN python -m app.db.init_chroma

# Run as a non-root user rather than root inside the container.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# MISTRAL_API_KEY is intentionally absent from this file. It is injected
# at `docker run` / `docker compose up` time (see docker-compose.yml's
# env_file), so it is never baked into an image layer or committed here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
