<#
.SYNOPSIS
  FrameLab developer commands for Windows PowerShell (same targets as the Makefile).
.EXAMPLE
  .\scripts\dev.ps1 install
  .\scripts\dev.ps1 test
  .\scripts\dev.ps1 api      # backend on http://localhost:8000
  .\scripts\dev.ps1 web      # frontend on http://localhost:5173
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "doctor", "api", "web", "test", "live-test", "lint", "format", "eval", "build-web", "docker", "demo-cache")]
    [string]$Task = "test"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

switch ($Task) {
    "install" {
        uv sync
        uv run playwright install chromium
        Push-Location web; npm install; Pop-Location
    }
    "doctor"     { uv run framelab doctor }
    "api"        { uv run uvicorn framelab.api.main:app --reload --port 8000 }
    "web"        { Push-Location web; npm run dev; Pop-Location }
    "test"       { uv run pytest -m "not live and not slow" }
    "live-test"  { $env:FRAMELAB_LIVE_TESTS = "1"; uv run pytest -m live; Remove-Item Env:FRAMELAB_LIVE_TESTS }
    "lint"       { uv run ruff check .; uv run ruff format --check . }
    "format"     { uv run ruff format .; uv run ruff check --fix . }
    "eval"       { uv run python eval/benchmark.py }
    "build-web"  { Push-Location web; npm run build; Pop-Location }
    "docker"     { docker build -t framelab . }
    "demo-cache" { uv run python scripts/build_demo_cache.py }
}
