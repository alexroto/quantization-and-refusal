#!/usr/bin/env bash
# Run inside the model's tmux session.
set -euo pipefail

model="${1:?Pass a model key}"
log_dir="${2:?Pass a log directory}"
python_bin="${PYTHON_BIN:-python}"
failed=0

for quantization in bf16 int8 int4; do
    for dataset in harmful harmless adversarial harmless_with_sensitive_words; do
        job="$quantization-$dataset"
        printf '\nStarting %s\n' "$job"

        if "$python_bin" -u -m experiment_1.run_inference \
            --model "$model" \
            --dataset "$dataset" \
            --quantization "$quantization" \
            --no-do-sample 2>&1 | tee "$log_dir/$job.log"; then
            printf 'Completed %s\n' "$job"
        else
            status=$?
            failed=$((failed + 1))
            printf 'Failed %s (exit %s)\n' "$job" "$status"
        fi
    done
done

printf '\nFinished %s: %s failed jobs out of 12.\n' "$model" "$failed" | tee "$log_dir/summary.log"
[[ "$failed" -eq 0 ]]
