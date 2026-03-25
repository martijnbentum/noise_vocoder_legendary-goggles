#!/bin/bash
# Vocode files from a text list of source or output filenames.

set -eu

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <job_name> <text_filename>" >&2
    exit 1
fi

job_name="$1"
text_filename="$2"

if [ ! -f "$text_filename" ]; then
    echo "Text filename does not exist: $text_filename" >&2
    exit 1
fi

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
env_dir="$repo_root/.venv-snellius"
input_dir='/projects/0/prjs1489/data/spidr/wav'
nprocess="${SLURM_CPUS_PER_TASK:-128}"
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

job_id="${SLURM_JOB_ID:-manual}"
baseline_count=$(find "$JOB_OUTPUT_DIR" -type f -name '*.wav' | wc -l | tr -d ' ')
progress_file="$archive_dir/progress_${job_id}.txt"
resolved_list="$archive_dir/selected_${job_name}_${job_id}.txt"

export INPUT_DIR="$input_dir"
export OUTPUT_DIR="$JOB_OUTPUT_DIR"
export JOB_NAME="$job_name"
export JOB_FAMILY="$JOB_FAMILY"
export JOB_KEY="$JOB_KEY"
export JOB_NBANDS="$JOB_NBANDS"
export TEXT_FILENAME="$text_filename"
export RESOLVED_LIST="$resolved_list"
export NPROCESS="$nprocess"

selected_count=$(python - <<'PY'
import os
import re
from pathlib import Path

from vocoder import core


input_dir = Path(os.environ['INPUT_DIR']).resolve()
output_dir = Path(os.environ['OUTPUT_DIR']).resolve()
text_filename = Path(os.environ['TEXT_FILENAME']).resolve()
resolved_list = Path(os.environ['RESOLVED_LIST'])
n_bands = int(os.environ['JOB_NBANDS'])

output_pattern = re.compile(
    rf'^[0-9a-f]{{8}}__.+_voc{n_bands}\.wav$'
)


def parse_lines(path):
    with path.open() as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            yield Path(line).expanduser().resolve()


def is_source_filename(path):
    try:
        path.relative_to(input_dir)
    except ValueError:
        return False
    return output_pattern.match(path.name) is None


def is_output_filename(path):
    try:
        path.relative_to(output_dir)
    except ValueError:
        return False
    return output_pattern.match(path.name) is not None


input_files = sorted(input_dir.rglob('*.wav'))
if not input_files:
    raise ValueError(f'No wav files found in input_dir: {input_dir}')

shard_map = core.build_output_shard_map(input_files, str(input_dir))
output_to_source = {}
for input_file in input_files:
    output_filename = Path(
        core.get_output_filename(
            str(input_file),
            output_dir=str(output_dir),
            input_dir=str(input_dir),
            output_shard_dir=shard_map[str(input_file)],
            n_bands=n_bands,
        )
    ).resolve()
    output_to_source[str(output_filename)] = str(input_file.resolve())

selected = []
seen = set()
for filename in parse_lines(text_filename):
    resolved = None
    if is_source_filename(filename):
        if not filename.is_file():
            raise ValueError(f'Source file does not exist: {filename}')
        resolved = str(filename)
    elif is_output_filename(filename):
        resolved = output_to_source.get(str(filename))
        if resolved is None:
            raise ValueError(
                f'Could not reconstruct source filename for {filename}'
            )
    else:
        raise ValueError(
            'Filename must be either a source wav in '
            f'{input_dir} or a vocoded output in {output_dir}: {filename}'
        )
    if resolved not in seen:
        selected.append(resolved)
        seen.add(resolved)

resolved_list.parent.mkdir(parents=True, exist_ok=True)
with resolved_list.open('w') as fout:
    for filename in selected:
        fout.write(f'{filename}\n')

print(len(selected))
PY
)

if [ "$selected_count" -eq 0 ]; then
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
    echo 'No files selected after normalization.'
    echo "resolved_list: $resolved_list"
    exit 0
fi

"$script_dir/output_progress_monitor.sh" \
    "$JOB_OUTPUT_DIR" \
    "$progress_file" \
    "$baseline_count" \
    "$selected_count" \
    30 \
    360 &
progress_pid=$!

cleanup() {
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
}

trap cleanup EXIT

echo '=== Snellius filename-list vocoder job ==='
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "job_name: ${SLURM_JOB_NAME:-none}"
echo "selection_target: $job_name"
echo "cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "nprocess: $nprocess"
echo "input_dir: $input_dir"
echo "output_dir: $JOB_OUTPUT_DIR"
echo "text_filename: $text_filename"
echo "resolved_list: $resolved_list"
echo "selected_count: $selected_count"
echo "progress_file: $progress_file"
echo "venv: $env_dir"
echo '========================================='

python - <<'PY'
import multiprocessing
import os
from pathlib import Path

from vocoder import core


WORKER_MAX_TASKS = 100


def load_selected_files(filename):
    with Path(filename).open() as fin:
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
    resolved_list = Path(os.environ['RESOLVED_LIST'])
    nprocess = int(os.environ['NPROCESS'])

    input_files = sorted(input_dir.rglob('*.wav'))
    if not input_files:
        raise ValueError(f'No wav files found in input_dir: {input_dir}')

    shard_map = core.build_output_shard_map(input_files, input_dir)
    requested = set(load_selected_files(resolved_list))
    selected_files = [
        str(input_file)
        for input_file in input_files
        if str(input_file) in requested
    ]
    if not selected_files:
        print('No matching source files remained to process.', flush=True)
        return

    worker_config = {
        'sample_rate': 16000,
        'butterworth_order': 4,
        'match_rms': False,
        'output_dir': str(output_dir),
        'input_dir': str(input_dir),
        'frequencies': core.get_standard_bands(
            n_bands=n_bands,
            family=family,
            key=key,
        ).tolist(),
    }
    tasks = list(core.iter_batch_tasks(selected_files, shard_map))
    chunksize = core.compute_pool_chunksize(len(tasks), nprocess)
    failure_path = (
        output_dir
        / f'filename_list_failures_{os.environ["JOB_NAME"]}.jsonl'
    )

    print(
        'selection pool launch:',
        f'workers={nprocess}',
        f'chunksize={chunksize}',
        f'maxtasksperchild={WORKER_MAX_TASKS}',
        flush=True,
    )
    with multiprocessing.Pool(
        nprocess,
        maxtasksperchild=WORKER_MAX_TASKS,
        initializer=core.init_pool_worker,
        initargs=(worker_config,),
    ) as pool:
        processed = 0
        failures = 0
        for result in pool.imap_unordered(
            core.handle_task,
            tasks,
            chunksize=chunksize,
        ):
            processed += 1
            if result['status'] != 'ok':
                failures += 1
                core.append_metadata(failure_path, result)
                print(
                    'selection file failed:',
                    result['input_filename'],
                    flush=True,
                )
        print(f'selection_processed: {processed}', flush=True)
        if failures:
            raise RuntimeError(
                f'Filename-list batch failed for {failures} files'
            )


main()
PY
