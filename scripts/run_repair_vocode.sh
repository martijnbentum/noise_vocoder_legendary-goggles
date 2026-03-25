#!/bin/bash
# Vocode only the files still missing from a configured Snellius job.

set -eu

cpus=64

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpus)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --cpus" >&2
                exit 1
            fi
            cpus="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 [--cpus n] <job_name> [missing_list]" >&2
    exit 1
fi

job_name="$1"
missing_list="${2:-}"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
env_dir="$repo_root/.venv-snellius"
input_dir='/projects/0/prjs1489/data/spidr/wav'
nprocess="${SLURM_CPUS_PER_TASK:-$cpus}"
archive_dir="$repo_root/archive"

. "$script_dir/snellius_jobs.sh"
load_vocode_job "$job_name"

if [ ! -x "$env_dir/bin/python" ]; then
    "$script_dir/build_snellius_env.sh"
fi

source "$env_dir/bin/activate"

mkdir -p "$archive_dir"
mkdir -p "$JOB_OUTPUT_DIR"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ -z "$missing_list" ]; then
    missing_list=$("$script_dir/find_missing_vocoded_wavs.sh" "$job_name")
fi

if [ ! -f "$missing_list" ]; then
    echo "Missing list does not exist: $missing_list" >&2
    exit 1
fi

job_id="${SLURM_JOB_ID:-manual}"
baseline_count=$(find "$JOB_OUTPUT_DIR" -type f -name '*.wav' | wc -l | tr -d ' ')
missing_count=$(wc -l < "$missing_list" | tr -d ' ')
progress_file="$archive_dir/progress_${job_id}.txt"

if [ "$missing_count" -eq 0 ]; then
    {
        echo 'status: done'
        echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "output_dir: $JOB_OUTPUT_DIR"
        echo "wav_files_present: $baseline_count"
        echo "wav_files_baseline: $baseline_count"
        echo 'wav_files_written: 0'
        echo 'chunk_dir_count: 0'
        echo 'last_wav_change_epoch: 0'
        echo 'seconds_since_last_wav_change: unknown'
        echo 'total_expected: 0'
    } > "$progress_file"
    echo 'No missing files found.'
    echo "missing_list: $missing_list"
    exit 0
fi

"$script_dir/output_progress_monitor.sh" \
    "$JOB_OUTPUT_DIR" \
    "$progress_file" \
    "$baseline_count" \
    "$missing_count" \
    30 \
    360 &
progress_pid=$!

cleanup() {
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
}

trap cleanup EXIT

echo '=== Snellius vocoder repair job ==='
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "job_name: ${SLURM_JOB_NAME:-none}"
echo "repair_target: $job_name"
echo "cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "nprocess: $nprocess"
echo "input_dir: $input_dir"
echo "output_dir: $JOB_OUTPUT_DIR"
echo "missing_list: $missing_list"
echo "missing_count: $missing_count"
echo "progress_file: $progress_file"
echo "venv: $env_dir"
echo '==============================='

export INPUT_DIR="$input_dir"
export OUTPUT_DIR="$JOB_OUTPUT_DIR"
export JOB_NAME="$job_name"
export JOB_FAMILY="$JOB_FAMILY"
export JOB_KEY="$JOB_KEY"
export JOB_NBANDS="$JOB_NBANDS"
export MISSING_LIST="$missing_list"
export NPROCESS="$nprocess"

python - <<'PY'
import os
from pathlib import Path

from vocoder import batch
from vocoder import file_io


def load_missing_files(missing_list):
    with Path(missing_list).open() as fin:
        return [
            line.strip()
            for line in fin
            if line.strip()
        ]


def main():
    input_dir = Path(os.environ['INPUT_DIR'])
    output_dir = Path(os.environ['OUTPUT_DIR'])
    family = os.environ['JOB_FAMILY']
    key = os.environ['JOB_KEY']
    n_bands = int(os.environ['JOB_NBANDS'])
    missing_list = Path(os.environ['MISSING_LIST'])
    nprocess = int(os.environ['NPROCESS'])

    input_files = sorted(input_dir.rglob('*.wav'))
    if not input_files:
        raise ValueError(f'No wav files found in input_dir: {input_dir}')

    shard_map = file_io.build_output_shard_map(input_files, input_dir)
    requested = set(load_missing_files(missing_list))
    missing_files = [
        str(input_file)
        for input_file in input_files
        if str(input_file) in requested
    ]
    if not missing_files:
        print('No matching missing files remained to process.', flush=True)
        return

    args = type('Args', (), {})()
    args.sample_rate = 16000
    args.butterworth_order = 4
    args.match_rms = False
    args.output_dir = str(output_dir)
    args.input_dir = str(input_dir)
    args.nbands = n_bands
    args.frequency_family = family
    args.frequency_key = key
    args.frequencies = None
    args.metadata_filename = ''
    args.failure_filename = (
        f'repair_failures_{os.environ["JOB_NAME"]}.jsonl'
    )

    print(
        'repair batch launch:',
        f'requested_cpus={nprocess}',
        'mode=sequential',
        flush=True,
    )
    batch.run_batch(args, missing_files, shard_map)


main()
PY
