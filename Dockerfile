# One image for both services (api, ui). FinBERT (the `finbert` uv extra, CPU torch on Linux) and
# its pinned weights are baked in, and the demo database is built from data/demo (D-48).
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf
WORKDIR /app

# Dependencies first: this layer is rebuilt only when pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra finbert

# Pinned FinBERT weights: needs only the config file, so a code change does not re-download.
COPY config/nlp.yaml config/nlp.yaml
COPY docker/bake_finbert.py docker/bake_finbert.py
RUN python docker/bake_finbert.py

COPY . .
# Demo database + trained models + cached LLM responses from the committed slice (no keys, no net).
RUN python -m src.demo load --force

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    API_URL=http://api:8000/api/v1
EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
