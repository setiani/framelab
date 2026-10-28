# --- web build ---
FROM node:20-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci || npm install
COPY web/ ./
RUN npm run build

# --- runtime ---
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 FRAMELAB_RUNS_DIR=/data/runs
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY backend/ backend/
RUN uv pip install --system -e . && python -m playwright install --with-deps chromium
COPY --from=web /web/dist web/dist
COPY demo_runs/ demo_runs/
EXPOSE 8000
CMD ["uvicorn", "framelab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
