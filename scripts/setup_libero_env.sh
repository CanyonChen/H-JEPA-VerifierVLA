#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

env -u PYTHONPATH \
    UV_PROJECT_ENVIRONMENT="$project_root/.venv-libero" \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=4 \
    uv sync --project envs/libero --frozen
env -u PYTHONPATH uv pip check --python .venv-libero/bin/python

bash scripts/run_libero_check.sh --render-only
