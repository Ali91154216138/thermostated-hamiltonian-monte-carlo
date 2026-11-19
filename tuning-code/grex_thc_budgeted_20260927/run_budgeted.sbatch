#!/bin/bash
#SBATCH --account=def-butlerlt
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=21-00:00:00
#SBATCH --signal=B:USR1@900

# The normal 21-day allocation is not a search deadline. A finite selected
# search is completed fully, with checkpoint continuation if ever necessary.
set -eo pipefail
module load CCEnv
module load StdEnv/2023 python/3.12.4 scipy-stack/2025a
set -u
umask 077
root="$1"
key="$2"
action="${3:-run}"
part="${4:-${SLURM_ARRAY_TASK_ID:-0}}"
python_bin=/home/alireza7/venvs/pa312_data/bin/python
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
cd "$root"
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
printf 'JOB %s %s key=%s part=%s action=%s host=%s\n' "$SLURM_JOB_ID" "$(date -Is)" "$key" "$part" "$action" "$(hostname)"
"$python_bin" -u "$root/launcher/budgeted_runner.py" "$action" --key "$key" --root "$root" --part "$part" &
child_pid=$!
if [[ "$cancel_notice" -eq 1 ]]; then kill -TERM "$child_pid"; fi
if [[ "$time_notice" -eq 1 ]]; then kill -USR1 "$child_pid"; fi
set +e
while true; do
  wait "$child_pid"
  result=$?
  if kill -0 "$child_pid" 2>/dev/null; then continue; fi
  if [[ "$result" -gt 128 && ( "$time_notice" -eq 1 || "$cancel_notice" -eq 1 ) ]]; then
    wait "$child_pid"; reaped=$?
    if [[ "$reaped" -ne 127 ]]; then result=$reaped; fi
  fi
  break
done
set -e
if [[ "$cancel_notice" -eq 1 ]]; then result=143; fi
if [[ "$result" -eq 75 && "$action" == run ]]; then
  # Continue only this part after a normal allocation warning. Never on cancel.
  next_job=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
    --job-name="THC27_${key}_p${part}" --chdir="$root" \
    --output="$root/logs/${key}_p${part}_%j.out" --error="$root/logs/${key}_p${part}_%j.err" \
    "$root/launcher/run_budgeted.sbatch" "$root" "$key" run "$part")
  printf 'CONTINUATION %s part=%s %s\n' "$next_job" "$part" "$(date -Is)" >> "$root/job_history.txt"
  exit 0
fi
printf 'EXIT %s %s\n' "$SLURM_JOB_ID" "$result"
exit "$result"
