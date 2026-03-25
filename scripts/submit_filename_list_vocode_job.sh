#!/bin/bash
# Submit a Snellius vocoder job for a text file with selected filenames.

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

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
. "$script_dir/snellius_jobs.sh"
load_vocode_job "$job_name"

text_filename=$(cd "$(dirname "$text_filename")" && pwd)/$(basename "$text_filename")

echo 'Submitting filename-list vocoder job'
echo "job_name: $job_name"
echo "text_filename: $text_filename"
echo "output_dir: $JOB_OUTPUT_DIR"

sbatch \
    --chdir="$repo_root" \
    --job-name="${job_name}_list" \
    --export=ALL,JOB_NAME="$job_name",TEXT_FILENAME="$text_filename" \
    "$repo_root/scripts/sbatch_filename_list_vocode_job.sh"
