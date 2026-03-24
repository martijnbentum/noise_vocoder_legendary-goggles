#!/bin/bash
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <job_name>" >&2
    exit 1
fi

job_name="$1"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
. "$script_dir/snellius_jobs.sh"
load_vocode_job "$job_name"

input_dir='/projects/0/prjs1489/data/spidr/wav'
output_dir="$JOB_OUTPUT_DIR"
archive_dir="$repo_root/archive"
job_id="${SLURM_JOB_ID:-manual}"
missing_file="$archive_dir/missing_${job_name}_${job_id}.txt"

mkdir -p "$archive_dir"

python - "$input_dir" "$output_dir" "$JOB_NBANDS" "$missing_file" <<'PY'
import sys
from pathlib import Path

from vocoder import core


input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
n_bands = int(sys.argv[3])
missing_file = Path(sys.argv[4])

input_files = sorted(input_dir.rglob('*.wav'))
if not input_files:
    raise ValueError(f'No wav files found in input_dir: {input_dir}')

shard_map = core.build_output_shard_map(input_files, input_dir)
missing = []
for input_file in input_files:
    output_filename = Path(
        core.get_output_filename(
            str(input_file),
            output_dir=str(output_dir),
            input_dir=str(input_dir),
            output_shard_dir=shard_map[str(input_file)],
            n_bands=n_bands,
        )
    )
    if not output_filename.exists():
        missing.append(str(input_file))

missing_file.parent.mkdir(parents=True, exist_ok=True)
with missing_file.open('w') as fout:
    for filename in missing:
        fout.write(f'{filename}\n')
PY

echo "$missing_file"
