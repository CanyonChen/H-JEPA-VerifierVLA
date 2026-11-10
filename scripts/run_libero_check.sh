#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec env -u PYTHONPATH \
    LIBERO_CONFIG_PATH="$project_root/.venv-libero/libero-config" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}" \
    .venv-libero/bin/python -u scripts/check_libero_env.py "$@"
