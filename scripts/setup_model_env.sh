#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Install the checked-in lockfile without re-resolving for server-specific indexes.
env -u PYTHONPATH \
    GIT_LFS_SKIP_SMUDGE=1 \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=4 \
    uv sync --frozen
env -u PYTHONPATH uv pip check --python .venv/bin/python
env -u PYTHONPATH .venv/bin/python scripts/check_model_env.py --device cpu

printf '\nActivate with: source "%s/.venv/bin/activate"\n' "$project_root"
