#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
env -u PYTHONPATH UV_PROJECT_ENVIRONMENT="$project_root/.venv-lilo" \
    UV_PYTHON_DOWNLOADS=automatic UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 \
    uv sync --project envs/lilo --frozen --no-active
env -u PYTHONPATH uv pip check --python .venv-lilo/bin/python
env -u PYTHONPATH .venv-lilo/bin/python -u -m scripts.check_lilo_env "$@"
