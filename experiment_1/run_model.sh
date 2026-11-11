#!/usr/bin/env bash
# Start a detached tmux session for one model.
set -euo pipefail

model="${1:?Pass a model key}"
case "$model" in
    llama2|qwen|vicuna) ;;
    *) printf 'Unknown model: %s\n' "$model" >&2; exit 2 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

socket="quantization-$model"
log_dir="$PWD/data/output/$model/logs/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p -- "$log_dir"

# A separate tmux server inherits this shell's Python environment and HF_TOKEN.
tmux -L "$socket" new-session -d -s "$model" \
    bash "$script_dir/run_jobs.sh" "$model" "$log_dir"

printf 'Attach: tmux -L %s attach -t %s\n' "$socket" "$model"
printf 'Logs: %s\n' "$log_dir"
