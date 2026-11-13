#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
echo "Reusing the existing B3 .venv-libero environment; running Ultra-Long preflight."
exec bash scripts/run_lilo_check.sh "$@"
