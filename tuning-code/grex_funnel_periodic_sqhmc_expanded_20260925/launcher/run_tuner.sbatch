#!/bin/bash
#SBATCH --account=def-butlerlt
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=21-00:00:00
#SBATCH --signal=B:USR1@900

# Arguments: experiment directory, method, number of parts, optional part ID.
# Initial THC submission uses an array. A timed-out part resumes independently.
set -eo pipefail
module load CCEnv
module load StdEnv/2023 python/3.12.4 scipy-stack/2025a
set -u
umask 077
experiment_dir="$1"
method="$2"
parts="$3"
part="${4:-${SLURM_ARRAY_TASK_ID:-0}}"
case "$method" in
  periodic) label=Periodic_THC ;;
  sqhmc) label=S_QHMC ;;
  *) echo 'Unknown method'; exit 2 ;;
esac
run_dir="$experiment_dir/$label"
part_label=$(printf 'part_%03d' "$part")
part_dir="$run_dir/parts/$part_label"
mkdir -p "$part_dir/logs"
source_file="$experiment_dir/code/funnel_${method}_EXPANDED_20260925.py"
launcher="$experiment_dir/launcher/checkpoint_runner.py"
python_bin=/home/alireza7/venvs/pa312_data/bin/python
test -x "$python_bin"
test -f "$source_file"
cd "$run_dir"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
printf '%s START %s method=%s part=%s/%s host=%s\n' "$SLURM_JOB_ID" "$(date -Is)" "$method" "$part" "$parts" "$(hostname)" | tee -a "$part_dir/job_history.txt"
child_pid=''
time_notice=0
cancel_notice=0
forward_time_notice() {
  time_notice=1
  if [[ -n "$child_pid" ]]; then kill -USR1 "$child_pid" 2>/dev/null || true; fi
}
forward_cancel() {
  cancel_notice=1
  if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; fi
}
trap forward_time_notice USR1
trap forward_cancel TERM INT
"$python_bin" -u "$launcher" --method "$method" --source "$source_file" \
  --run-dir "$run_dir" --part "$part" --parts "$parts" &
child_pid=$!
if [[ "$cancel_notice" -eq 1 ]]; then kill -TERM "$child_pid"; fi
if [[ "$time_notice" -eq 1 ]]; then kill -USR1 "$child_pid"; fi
set +e
while true; do
  wait "$child_pid"
  result=$?
  if kill -0 "$child_pid" 2>/dev/null; then continue; fi
  # wait can be interrupted before it reaps a child; wait again for its code.
  if [[ "$result" -gt 128 && ( "$time_notice" -eq 1 || "$cancel_notice" -eq 1 ) ]]; then
    wait "$child_pid"; reaped=$?
    if [[ "$reaped" -ne 127 ]]; then result=$reaped; fi
  fi
  break
done
set -e
if [[ "$cancel_notice" -eq 1 ]]; then result=143; fi
if [[ "$result" -eq 75 ]]; then
  # No --array here: only THIS incomplete part is continued.
  next_job=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
    --job-name="FNL_${method}_p${part}_exp01" --chdir="$run_dir" \
    --output="$part_dir/logs/${label}_%j.out" --error="$part_dir/logs/${label}_%j.err" \
    "$experiment_dir/launcher/run_tuner.sbatch" "$experiment_dir" "$method" "$parts" "$part")
  printf '%s CONTINUATION %s %s\n' "$SLURM_JOB_ID" "$next_job" "$(date -Is)" | tee -a "$part_dir/job_history.txt"
  exit 0
fi
printf '%s EXIT %s %s\n' "$SLURM_JOB_ID" "$result" "$(date -Is)" | tee -a "$part_dir/job_history.txt"
exit "$result"
