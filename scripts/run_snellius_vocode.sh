#!/bin/bash
# Run a configured vocoding batch on Snellius.

set -eu

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <job_name> <family> <key> <n_bands> <output_dir>" >&2
    exit 1
fi

job_name="$1"
family="$2"
key="$3"
n_bands="$4"
output_dir="$5"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
env_dir="$repo_root/.venv-snellius"
input_dir="/projects/0/prjs1489/data/spidr/wav"
nprocess="${SLURM_CPUS_PER_TASK:-64}"
archive_dir="$repo_root/archive"
stall_seconds=360
progress_interval_seconds=30

if [ ! -x "$env_dir/bin/python" ]; then
    "$script_dir/build_snellius_env.sh"
fi

source "$env_dir/bin/activate"

mkdir -p "$output_dir"
mkdir -p "$archive_dir"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

input_count=$(find "$input_dir" -type f -name '*.wav' | wc -l | tr -d ' ')
baseline_count=$(find "$output_dir" -type f -name '*.wav' | wc -l | tr -d ' ')
job_id="${SLURM_JOB_ID:-manual}"
progress_file="$archive_dir/progress_${job_id}.txt"

"$script_dir/output_progress_monitor.sh" \
    "$output_dir" \
    "$progress_file" \
    "$baseline_count" \
    "$input_count" \
    "$progress_interval_seconds" \
    "$stall_seconds" &
progress_pid=$!

cleanup() {
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
}

trap cleanup EXIT

get_progress_value() {
    if [ ! -f "$progress_file" ]; then
        return 1
    fi
    awk -F': ' -v key="$1" '$1 == key {print $2}' "$progress_file" \
        | tail -n 1
}

echo "=== Snellius vocoder job ==="
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "job_name: ${SLURM_JOB_NAME:-none}"
echo "cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "nprocess: $nprocess"
echo "frequency_family: $family"
echo "frequency_key: $key"
echo "n_bands: $n_bands"
echo "input_dir: $input_dir"
echo "input_wav_count: $input_count"
echo "output_dir: $output_dir"
echo "progress_file: $progress_file"
echo "venv: $env_dir"
echo "==========================="

srun python -u -m vocoder \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --nprocess "$nprocess" \
    --nbands "$n_bands" \
    --frequency_family "$family" \
    --frequency_key "$key" &
srun_pid=$!
repair_submitted=0
stalled_run=0

while kill -0 "$srun_pid" 2>/dev/null; do
    sleep "$progress_interval_seconds"
    if ! kill -0 "$srun_pid" 2>/dev/null; then
        break
    fi
    progress_status=$(get_progress_value 'status' || true)
    if [ "$progress_status" != 'stalled' ]; then
        continue
    fi
    stalled_run=1
    percent_complete=$(get_progress_value 'percent_complete' || true)
    if [ "$repair_submitted" -eq 0 ] && [ "$percent_complete" != '100%' ]; then
        "$script_dir/submit_repair_vocode_job.sh" "$job_name"
        repair_submitted=1
    fi
    echo "Detected stalled vocoder run for $job_name; stopping job."
    kill "$srun_pid" 2>/dev/null || true
    break
done

set +e
wait "$srun_pid"
srun_status=$?
set -e

if [ "$stalled_run" -eq 1 ]; then
    echo "Stopped stalled run for $job_name."
    exit 0
fi

exit "$srun_status"
