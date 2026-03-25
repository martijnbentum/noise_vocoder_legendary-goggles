#!/bin/bash
set -eu

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$script_dir/snellius_jobs.sh"

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

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [--cpus n] <job_name>|--list" >&2
    exit 1
fi

job_name="$1"
if [ "$job_name" = "--list" ]; then
    list_vocode_jobs
    exit 0
fi

load_vocode_job "$job_name"

"$script_dir/submit_snellius_job.sh" \
    --cpus "$cpus" \
    "$script_dir/sbatch_vocode_job.sh" \
    "$JOB_OUTPUT_DIR" \
    "$job_name"
