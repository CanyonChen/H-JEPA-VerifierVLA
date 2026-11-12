#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec env -u PYTHONPATH \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS=cuda \
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}" \
    .venv/bin/python -u -m scripts.serve_pi05_libero \
    --port "${PI05_PORT:-8000}" \
    --checkpoint "${PI05_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_libero}"
