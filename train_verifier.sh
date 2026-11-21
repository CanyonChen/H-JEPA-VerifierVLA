#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -n "${VERIVLA_PYTHON:-}" ]]; then
    python_bin="$VERIVLA_PYTHON"
elif [[ -x .venv-robometer/bin/python ]]; then
    python_bin=.venv-robometer/bin/python
elif command -v python >/dev/null 2>&1; then
    python_bin=python
else
    python_bin=python3
fi
exec "$python_bin" -B -u -m scripts.train_verifier "$@"
