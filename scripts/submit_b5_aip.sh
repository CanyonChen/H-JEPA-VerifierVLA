#!/usr/bin/env bash
# The notebook stages code on shared storage, then submits a single AIP job.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
rev="$(git -C "$root" rev-parse --verify "$1^{commit}")"
checkout="$(dirname -- "$root")/.aip-verifier/$rev"
mkdir -p "$(dirname -- "$checkout")"
if [[ ! -d "$checkout" ]]; then
    git -C "$root" worktree add --quiet --detach "$checkout" "$rev"
fi
[[ "$(git -C "$checkout" rev-parse HEAD)" == "$rev" ]]
for item in .venv .venv-libero .venv-robometer outputs; do
    if [[ ! -e "$checkout/$item" ]]; then ln -s "$root/$item" "$checkout/$item"; fi
    [[ "$checkout/$item" -ef "$root/$item" ]]
done
if [[ -d "$root/data" && ! -e "$checkout/data" ]]; then ln -s "$root/data" "$checkout/data"; fi
notebook="${AIP_NOTEBOOK_NAME:-${NOTEBOOK_NAME:-h100}}"
name="b5-long100-joint40-$(date -u +%m%d%H%M%S)"
args=(train submit --project embodiedai --mode standalone --gpu 1 --gpu-type h100
      --cpu 10 --memory 100000 --type pytorch --notebook "$notebook" --name "$name")
if [[ -n "${AIP_POOL:-}" ]]; then args+=(--pool "$AIP_POOL"); fi
printf -v command 'bash %q %q' "$checkout/scripts/run_b5_aip.sh" "$root"
printf 'Submitting %s, notebook=%s, revision=%s\n' "$name" "$notebook" "$rev"
exec aip "${args[@]}" --command "$command"
