#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_path="$project_root/.venv-robometer/bin/python"

if [[ ! -x "$python_path" ]]; then
    UV_PYTHON_DOWNLOADS=automatic uv venv --python 3.10 "$project_root/.venv-robometer"
fi
env -u PYTHONPATH "$python_path" -I -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
env -u PYTHONPATH UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 \
    uv pip install --python "$python_path" --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision==0.23.0
env -u PYTHONPATH UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 \
    uv pip install --python "$python_path" --index-url https://pypi.org/simple \
    -r envs/robometer/requirements.txt
# The adapter imports official model classes. Upstream training and dataset-upload
# dependencies are outside this inference environment.
env -u PYTHONPATH UV_HTTP_TIMEOUT=600 uv pip install --python "$python_path" --no-deps --index-url https://pypi.org/simple \
    'robometer @ https://github.com/robometer/robometer/archive/352d160389daa964788de1ec933d1925f3a6de4f.zip'
env -u PYTHONPATH "$python_path" -I -c \
    'import torch, transformers; from robometer.models import RBM; print("Robometer imports OK", torch.__version__, transformers.__version__)'
