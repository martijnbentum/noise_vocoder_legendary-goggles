#!/bin/bash
set -eu

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$script_dir/snellius_jobs.sh"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <job_name>|--list" >&2
    exit 1
fi

job_name="$1"
if [ "$job_name" = "--list" ]; then
    list_vocode_jobs
    exit 0
fi

load_vocode_job "$job_name"

"$script_dir/submit_snellius_job.sh" \
    "$script_dir/sbatch_vocode_job.sh" \
    "$JOB_OUTPUT_DIR" \
    "$job_name"
