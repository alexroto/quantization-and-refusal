#!/usr/bin/env bash
# Launch the vicuna inference sequence in a detached tmux session.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_model.sh" vicuna
