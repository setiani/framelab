# FrameLab developer commands. Windows users: see scripts/dev.ps1 (same targets).
.PHONY: install dev api web test live-test lint format eval build-web docker doctor demo-cache

install:
	uv sync
	uv run playwright install chromium
	cd web && npm install

doctor:
	uv run framelab doctor

api:
	uv run uvicorn framelab.api.main:app --reload --port 8000

web:
	cd web && npm run dev

dev:
	@echo "Run 'make api' and 'make web' in two terminals."

test:
	uv run pytest -m "not live and not slow"

live-test:
	FRAMELAB_LIVE_TESTS=1 uv run pytest -m live

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

eval:
	uv run python eval/benchmark.py

build-web:
	cd web && npm run build

docker:
	docker build -t framelab .

demo-cache:
	uv run python scripts/build_demo_cache.py
