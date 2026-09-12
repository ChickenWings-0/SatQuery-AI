# SatQuery AI — the API.
#
# Two targets:
#
#   deps     the locked Python environment, no GPU stack. What CI builds to
#            catch Dockerfile rot, and the base for a CPU-only demo
#            (SATQUERY_VLM_BACKEND=none: templated answers, no weights).
#   runtime  deps + the ROCm serving stack. Large; built on the demo box.
#
# Weights, data/, runs/ and the SQLite trace DB are never baked in — they are
# volumes (see docker-compose.yml). The image is the code and its environment.
#
#   docker build --target deps -t satquery-api:deps .
#   docker build -t satquery-api:latest .

# --------------------------------------------------------------------- deps
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# rasterio/GDAL wheels are self-contained; libexpat and libgomp are what the
# scientific stack still dlopens at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Dependencies first, without the project itself, so this layer only changes
# when the lockfile does.
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY openapi.json README.md ./
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 satquery \
    && mkdir -p /app/data && chown -R satquery:satquery /app
USER satquery

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/v1/health || exit 1

CMD ["uvicorn", "satquery.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------ runtime
# The serving stack on top. ROCm wheels come from PyTorch's index, not PyPI —
# the lockfile pins the CUDA build, which is why this is a separate layer and
# why `uv sync` is never run again after it (it would replace the stack).
FROM deps AS runtime

USER root
RUN uv pip install --index-url https://download.pytorch.org/whl/rocm6.4 \
        "torch==2.9.1+rocm6.4" "torchvision==0.24.1+rocm6.4" \
    && uv pip install "transformers==5.16.1" accelerate peft
USER satquery

# Every shell that touches the GPU sources scripts/rocm_env.sh; the container
# gets the same values as environment.
ENV HIP_VISIBLE_DEVICES=0 \
    ROCR_VISIBLE_DEVICES=0 \
    PYTORCH_HIP_ALLOC_CONF="expandable_segments:True" \
    HSA_OVERRIDE_GFX_VERSION=11.0.0
