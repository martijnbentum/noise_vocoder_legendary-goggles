#!/bin/bash
# Submit a Snellius vocoder job for a text file with selected filenames.

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

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [--cpus n] <job_name> <text_filename>" >&2
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
echo "cpus_per_task: $cpus"

sbatch \
    --chdir="$repo_root" \
    --cpus-per-task="$cpus" \
    --job-name="${job_name}_list" \
    --export=ALL,JOB_NAME="$job_name",TEXT_FILENAME="$text_filename" \
    "$repo_root/scripts/sbatch_filename_list_vocode_job.sh"
