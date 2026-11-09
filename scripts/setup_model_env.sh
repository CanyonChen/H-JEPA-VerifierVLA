#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

env -u PYTHONPATH GIT_LFS_SKIP_SMUDGE=1 uv sync --locked
env -u PYTHONPATH uv pip check --python .venv/bin/python
env -u PYTHONPATH .venv/bin/python scripts/check_model_env.py --device cpu

printf '\nActivate with: source "%s/.venv/bin/activate"\n' "$project_root"
