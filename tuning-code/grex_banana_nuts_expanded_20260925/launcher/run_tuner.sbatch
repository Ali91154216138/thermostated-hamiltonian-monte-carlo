#!/bin/bash
#SBATCH --account=def-butlerlt
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=21-00:00:00
#SBATCH --signal=B:USR1@900

# Independent NUTS tuning only; shared periodic THC-v2 job is not changed or duplicated.
# Argument: experiment directory. Set memory/logs at submission.
set -eo pipefail
module load CCEnv
module load StdEnv/2023 python/3.12.4 scipy-stack/2025a
set -u
umask 077
experiment_dir="$1"
method_label=NUTS
filename=banana_nuts_tuner_EXPANDED_20260925.py
run_dir="$experiment_dir/$method_label"
source_file="$run_dir/code/$filename"
launcher="$experiment_dir/launcher/checkpoint_runner.py"
python_bin=/home/alireza7/venvs/pa312_data/bin/python
test -x "$python_bin"
test -f "$source_file"
test -f "$launcher"
cd "$run_dir"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
printf 'Job %s; method %s; started %s; host %s\n' "$SLURM_JOB_ID" "$method_label" "$(date -Is)" "$(hostname)"
printf '%s START %s\n' "$SLURM_JOB_ID" "$(date -Is)" >> "$run_dir/job_history.txt"
child_pid=''
time_notice=0
forward_time_notice() {
    time_notice=1
    if [[ -n "$child_pid" ]]; then kill -USR1 "$child_pid" 2>/dev/null || true; fi
}
trap forward_time_notice USR1
"$python_bin" -u "$launcher" --source "$source_file" --run-dir "$run_dir" &
child_pid=$!
set +e
while true; do
    wait "$child_pid"
    result=$?
    if [[ "$result" -eq 138 && "$time_notice" -eq 1 ]]; then time_notice=0; continue; fi
    if kill -0 "$child_pid" 2>/dev/null; then continue; fi
    break
done
set -e
if [[ "$result" -eq 75 ]]; then
    next_job=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
        --job-name="$SLURM_JOB_NAME" --mem=8G \
        --chdir="$run_dir" --output="$run_dir/logs/${method_label}_%j.out" \
        --error="$run_dir/logs/${method_label}_%j.err" \
        "$experiment_dir/launcher/run_tuner.sbatch" "$experiment_dir")
    printf '%s CONTINUATION %s %s\n' "$SLURM_JOB_ID" "$next_job" "$(date -Is)" >> "$run_dir/job_history.txt"
    printf 'Saved blocks; continuation job %s submitted.\n' "$next_job"
    exit 0
fi
printf '%s EXIT %s %s\n' "$SLURM_JOB_ID" "$result" "$(date -Is)" >> "$run_dir/job_history.txt"
exit "$result"
