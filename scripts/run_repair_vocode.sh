#!/bin/bash
# Migrate legacy vocoder outputs into the current shard layout and
# re-run only the still-missing files listed in an archive file.

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

. "$script_dir/snellius_jobs.sh"
load_vocode_job "$job_name"

if [ ! -x "$env_dir/bin/python" ]; then
    "$script_dir/build_snellius_env.sh"
fi

source "$env_dir/bin/activate"

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

echo "=== Snellius vocoder repair job ==="
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "job_name: ${SLURM_JOB_NAME:-none}"
echo "repair_target: $JOB_NAME"
echo "cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "nprocess: $nprocess"
echo "input_dir: $INPUT_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "missing_list: $MISSING_LIST"
echo "venv: $env_dir"
echo "==============================="

python - <<'PY'
import json
import multiprocessing
import os
import shutil
import time
from pathlib import Path

import numpy as np

from vocoder import core


MAX_OUTPUT_FILES_PER_DIR = core.DEFAULT_MAX_OUTPUT_FILES_PER_DIR
WORKER_MAX_TASKS = 100


def legacy_preserved_path(input_file, input_dir, output_dir, n_bands):
    input_path = Path(input_file)
    try:
        relative_parent = input_path.parent.relative_to(Path(input_dir))
    except ValueError:
        relative_parent = Path()
    directory = Path(output_dir) / relative_parent
    return directory / f'{input_path.stem}_vocoded_nbands-{n_bands}.wav'


def legacy_sharded_long_path(input_file, input_dir, output_dir, shard_dir, n_bands):
    directory = Path(output_dir) / shard_dir
    stem = core.build_output_stem(input_file, input_dir)
    return directory / f'{stem}_vocoded_nbands-{n_bands}.wav'


def load_missing_set(missing_list):
    with Path(missing_list).open() as fin:
        return {
            str(Path(line.strip()))
            for line in fin
            if line.strip()
        }


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


def migrate_existing_outputs(input_files, shard_map, input_dir, output_dir, n_bands):
    moved = 0
    skipped = 0
    for input_file in input_files:
        input_name = str(input_file)
        shard_dir = shard_map[input_name]
        target_path = Path(
            core.get_output_filename(
                input_name,
                output_dir=output_dir,
                input_dir=input_dir,
                output_shard_dir=shard_dir,
                n_bands=n_bands,
            )
        )
        if target_path.exists():
            continue
        legacy_candidates = [
            legacy_preserved_path(input_name, input_dir, output_dir, n_bands),
            legacy_sharded_long_path(
                input_name,
                input_dir,
                output_dir,
                shard_dir,
                n_bands,
            ),
        ]
        source_path = None
        for candidate in legacy_candidates:
            if candidate.exists():
                source_path = candidate
                break
        if source_path is None:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        moved += 1
    for candidate in Path(output_dir).rglob('*_vocoded_nbands-*.wav'):
        skipped += 1
    return moved, skipped


def run_missing_batch(tasks, total_files, nprocess):
    start_time = time.time()
    processed = 0
    failures = 0
    chunksize = max(1, min(50, total_files // (nprocess * 4) or 1))
    failure_log = (
        Path(os.environ['OUTPUT_DIR'])
        / f'repair_failures_{os.environ["JOB_NAME"]}.jsonl'
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
            if processed == 1 or processed % 100 == 0:
                core.log_progress(processed, total_files, start_time, result)
    core.log_progress(processed, total_files, start_time)
    if failures:
        print(f'repair_failed_files: {failures}', flush=True)


def main():
    input_dir = os.environ['INPUT_DIR']
    output_dir = os.environ['OUTPUT_DIR']
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
        MAX_OUTPUT_FILES_PER_DIR,
    )
    missing_set = load_missing_set(missing_list)

    print(f'total_input_files: {len(input_files)}', flush=True)
    print(f'missing_list_entries: {len(missing_set)}', flush=True)
    moved, leftover_legacy = migrate_existing_outputs(
        input_files,
        shard_map,
        input_dir,
        output_dir,
        n_bands,
    )
    print(f'migrated_legacy_outputs: {moved}', flush=True)
    print(f'legacy_long_suffix_remaining: {leftover_legacy}', flush=True)

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
    resolved_missing = 0
    for input_file in input_files:
        input_name = str(input_file)
        shard_dir = shard_map[input_name]
        output_filename = core.get_output_filename(
            input_name,
            output_dir=output_dir,
            input_dir=input_dir,
            output_shard_dir=shard_dir,
            n_bands=n_bands,
        )
        if output_filename not in missing_set:
            continue
        if Path(output_filename).exists():
            resolved_missing += 1
            continue
        tasks.append(
            {
                'filename': input_name,
                'output_shard_dir': shard_dir,
                'worker_config': worker_config,
            }
        )

    unmatched_missing = len(missing_set) - len(tasks) - resolved_missing
    print(f'missing_resolved_by_migration: {resolved_missing}', flush=True)
    print(f'missing_remaining_to_vocode: {len(tasks)}', flush=True)
    print(f'missing_unmatched_entries: {unmatched_missing}', flush=True)

    if not tasks:
        print('No missing files need vocoding after migration.', flush=True)
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
