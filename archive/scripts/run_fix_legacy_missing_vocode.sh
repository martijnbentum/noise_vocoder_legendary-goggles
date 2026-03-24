#!/bin/bash
# Convert legacy missing output entries back to source wav files and
# vocode only the still-missing current outputs.

set -eu

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <job_name> <missing_list>" >&2
    exit 1
fi

job_name="$1"
missing_list="$2"

if [ ! -f "$missing_list" ]; then
    echo "Missing list does not exist: $missing_list" >&2
    exit 1
fi

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
env_dir="$repo_root/.venv-snellius"
input_dir="/projects/0/prjs1489/data/spidr/wav"
nprocess="${SLURM_CPUS_PER_TASK:-128}"
archive_dir="$repo_root/archive"

. "$script_dir/snellius_jobs.sh"
load_vocode_job "$job_name"

if [ ! -x "$env_dir/bin/python" ]; then
    "$script_dir/build_snellius_env.sh"
fi

source "$env_dir/bin/activate"

mkdir -p "$archive_dir"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export REPO_ROOT="$repo_root"
export INPUT_DIR="$input_dir"
export OUTPUT_DIR="$JOB_OUTPUT_DIR"
export JOB_NAME="$job_name"
export JOB_FAMILY="$JOB_FAMILY"
export JOB_KEY="$JOB_KEY"
export JOB_NBANDS="$JOB_NBANDS"
export MISSING_LIST="$missing_list"
export NPROCESS="$nprocess"
export LEGACY_OUTPUT_DIR="$JOB_OUTPUT_DIR"

job_id="${SLURM_JOB_ID:-manual}"
baseline_count=$(find "$JOB_OUTPUT_DIR" -type f -name '*.wav' | wc -l | tr -d ' ')
missing_count=$(wc -l < "$missing_list" | tr -d ' ')
progress_file="$archive_dir/progress_${job_id}.txt"

"$script_dir/output_progress_monitor.sh" \
    "$JOB_OUTPUT_DIR" \
    "$progress_file" \
    "$baseline_count" \
    "$missing_count" &
progress_pid=$!

cleanup() {
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
}

trap cleanup EXIT

echo "=== Snellius legacy missing repair job ==="
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "job_name: ${SLURM_JOB_NAME:-none}"
echo "repair_target: $JOB_NAME"
echo "cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "nprocess: $nprocess"
echo "input_dir: $INPUT_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "legacy_output_dir: $LEGACY_OUTPUT_DIR"
echo "missing_list: $MISSING_LIST"
echo "progress_file: $progress_file"
echo "venv: $env_dir"
echo "========================================"

python - <<'PY'
import multiprocessing
import os
import time
from pathlib import Path

from vocoder import core


WORKER_MAX_TASKS = 100


def load_missing_entries(missing_list):
    with Path(missing_list).open() as fin:
        return [line.strip() for line in fin if line.strip()]


def build_worker_config(
    sample_rate,
    butterworth_order,
    match_rms,
    output_dir,
    input_dir,
    family,
    key,
    n_bands,
):
    frequencies = core.get_standard_bands(
        n_bands=n_bands,
        family=family,
        key=key,
    )
    return {
        'sample_rate': sample_rate,
        'butterworth_order': butterworth_order,
        'match_rms': match_rms,
        'output_dir': output_dir,
        'input_dir': input_dir,
        'frequencies': frequencies.tolist(),
    }


def run_missing_batch(tasks, total_files, nprocess):
    processed = 0
    failures = 0
    start_time = time.time()
    chunksize = core.compute_pool_chunksize(total_files, nprocess)
    failure_log = (
        Path(os.environ['OUTPUT_DIR'])
        / f'legacy_missing_failures_{os.environ["JOB_NAME"]}.jsonl'
    )
    print(
        'repair pool launch:',
        f'workers={nprocess}',
        f'chunksize={chunksize}',
        f'maxtasksperchild={WORKER_MAX_TASKS}',
        flush=True,
    )
    with multiprocessing.Pool(
        nprocess,
        maxtasksperchild=WORKER_MAX_TASKS,
    ) as pool:
        for result in pool.imap_unordered(
            core.handle_task,
            tasks,
            chunksize=chunksize,
        ):
            processed += 1
            if result['status'] != 'ok':
                failures += 1
                core.append_metadata(failure_log, result)
                print(
                    'repair file failed:',
                    result['input_filename'],
                    flush=True,
                )
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    print(
        'repair batch complete:',
        f'processed={processed}',
        f'elapsed={elapsed:.1f}s',
        f'rate={rate:.2f} files/s',
        flush=True,
    )
    if failures:
        print(f'repair_failed_files: {failures}', flush=True)


def main():
    input_dir = os.environ['INPUT_DIR']
    output_dir = os.environ['OUTPUT_DIR']
    legacy_output_dir = os.environ['LEGACY_OUTPUT_DIR']
    family = os.environ['JOB_FAMILY']
    key = os.environ['JOB_KEY']
    n_bands = int(os.environ['JOB_NBANDS'])
    nprocess = int(os.environ['NPROCESS'])
    missing_list = os.environ['MISSING_LIST']

    input_files = sorted(Path(input_dir).rglob('*.wav'))
    if not input_files:
        raise ValueError(f'No wav files found in input_dir: {input_dir}')
    shard_map = core.build_output_shard_map(
        input_files,
        input_dir,
        core.DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
    )
    input_set = {str(path) for path in input_files}
    missing_entries = load_missing_entries(missing_list)

    print(f'total_input_files: {len(input_files)}', flush=True)
    print(f'missing_list_entries: {len(missing_entries)}', flush=True)

    worker_config = build_worker_config(
        sample_rate=16000,
        butterworth_order=4,
        match_rms=False,
        output_dir=output_dir,
        input_dir=input_dir,
        family=family,
        key=key,
        n_bands=n_bands,
    )

    tasks = []
    already_present = 0
    invalid_entries = 0
    source_missing = 0
    seen_sources = set()

    for entry in missing_entries:
        try:
            source_filename = core.legacy_output_to_source_filename(
                entry,
                legacy_output_dir=legacy_output_dir,
                input_dir=input_dir,
                n_bands=n_bands,
            )
        except ValueError:
            invalid_entries += 1
            continue
        if source_filename in seen_sources:
            continue
        seen_sources.add(source_filename)
        if source_filename not in input_set:
            source_missing += 1
            continue
        shard_dir = shard_map[source_filename]
        output_filename = core.get_output_filename(
            source_filename,
            output_dir=output_dir,
            input_dir=input_dir,
            output_shard_dir=shard_dir,
            n_bands=n_bands,
        )
        if Path(output_filename).exists():
            already_present += 1
            continue
        tasks.append(
            {
                'filename': source_filename,
                'output_shard_dir': shard_dir,
                'worker_config': worker_config,
            }
        )

    print(f'legacy_entries_invalid: {invalid_entries}', flush=True)
    print(f'legacy_entries_source_missing: {source_missing}', flush=True)
    print(f'legacy_entries_already_present: {already_present}', flush=True)
    print(f'legacy_entries_remaining_to_vocode: {len(tasks)}', flush=True)

    if not tasks:
        print('No legacy missing files need vocoding.', flush=True)
        return

    if nprocess == 1:
        for task in tasks:
            result = core.handle_task(task)
            if result['status'] != 'ok':
                raise RuntimeError(
                    f'Repair failed for {result["input_filename"]}: '
                    f'{result["error_message"]}'
                )
        return

    run_missing_batch(tasks, len(tasks), nprocess)


if __name__ == '__main__':
    main()
PY
