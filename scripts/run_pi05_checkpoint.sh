#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec env -u PYTHONPATH \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}" \
    .venv/bin/python -u scripts/check_pi05_checkpoint.py "$@"
