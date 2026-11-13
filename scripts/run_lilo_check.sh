#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec env -u PYTHONPATH .venv-libero/bin/python -u -m scripts.check_lilo_env "$@"
