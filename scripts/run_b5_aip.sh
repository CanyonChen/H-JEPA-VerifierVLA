#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
unset PYTHONPATH
export PYTHONUNBUFFERED=1 B5_SCHEDULER_ALLOCATED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export LIBERO_CONFIG_PATH="$PWD/outputs/verifier_aip_seed7/libero-config"
export B5_LIBERO_ASSET_ROOT="${1:-$PWD}/openpi/third_party/libero/libero/libero"
mkdir -p "$PWD/outputs/verifier_aip_seed7"
log="$PWD/outputs/verifier_aip_seed7/job-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$log") 2>&1
echo "Job log: $log"
echo "Run revision: $(git rev-parse HEAD)"
.venv-robometer/bin/python - <<'PY'
import torch
dev = torch.cuda.get_device_properties(0)
print("Allocated GPU:", dev.name, "GiB:", round(dev.total_memory/2**30, 1),
      "visible devices:", torch.cuda.device_count(), flush=True)
if torch.cuda.device_count() != 1 or "H100" not in dev.name.upper():
    raise RuntimeError("This job requested one H100")
PY
B5_SCHEDULER_GPU_UUID="$(.venv-robometer/bin/python -c 'import torch; print(getattr(torch.cuda.get_device_properties(0), "uuid", "") or "")')"
if [[ -n "$B5_SCHEDULER_GPU_UUID" && "$B5_SCHEDULER_GPU_UUID" != GPU-* ]]; then
    B5_SCHEDULER_GPU_UUID="GPU-$B5_SCHEDULER_GPU_UUID"
fi
export B5_SCHEDULER_GPU_UUID
metrics="${log%.log}-gpu.csv"
nvidia-smi -i "${B5_SCHEDULER_GPU_UUID:-0}" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total \
    --format=csv -l 5 >"$metrics" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
.venv-robometer/bin/python -u -m scripts.b5_pipeline
