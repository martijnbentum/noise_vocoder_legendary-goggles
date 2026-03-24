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

input_dir="/projects/0/prjs1489/data/spidr/wav"
output_dir="$JOB_OUTPUT_DIR"
n_bands="$JOB_NBANDS"

if [ ! -d "$input_dir" ]; then
    echo "Input directory does not exist: $input_dir" >&2
    exit 1
fi

if [ ! -d "$output_dir" ]; then
    echo "Output directory does not exist: $output_dir" >&2
    exit 1
fi

archive_dir="$repo_root/archive"
mkdir -p "$archive_dir"

timestamp=$(date +"%Y%m%d_%H%M%S")
output_file="$archive_dir/missing_${job_name}_${timestamp}.txt"

while IFS= read -r input_file; do
    rel_path=${input_file#"$input_dir"/}
    rel_dir=$(dirname "$rel_path")
    base_name=$(basename "$rel_path" .wav)
    expected_name="${base_name}_vocoded_nbands-${n_bands}.wav"
    if [ "$rel_dir" = "." ]; then
        expected_file="$output_dir/$expected_name"
    else
        expected_file="$output_dir/$rel_dir/$expected_name"
    fi
    if [ ! -f "$expected_file" ]; then
        printf '%s\n' "$expected_file" >> "$output_file"
    fi
done < <(find "$input_dir" -type f -name '*.wav' | sort)

if [ ! -f "$output_file" ]; then
    : > "$output_file"
fi

echo "$output_file"
