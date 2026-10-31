# FrameLab: one container serves the API and the built web app (docs/plan/10 §1).
#   docker build -t framelab .
#   docker run -p 8000:8000 -e MODEL_API_KEY=... -v framelab-data:/data framelab

# --- web build -------------------------------------------------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- runtime ---------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/app/.venv/bin:$PATH \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    FRAMELAB_RUNS_DIR=/data/runs \
    FRAMELAB_DEMO_DIR=/app/demo_runs \
    FRAMELAB_FORWARDED_HOPS=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.32
WORKDIR /app

# Dependencies first (cached while only the code changes), exactly as locked in uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/ backend/
RUN uv sync --frozen --no-dev \
    && playwright install --with-deps --only-shell chromium \
    && rm -rf /var/lib/apt/lists/*

COPY --from=web /web/dist web/dist
COPY demo_runs/ demo_runs/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)"
# One process: rate limits and live event fan-out are kept in memory.
CMD ["sh", "-c", "exec uvicorn framelab.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
