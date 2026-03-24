#!/bin/bash
# Submit a repair job that migrates legacy outputs and vocodes only
# the still-missing files from an archive list.

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

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
. "$script_dir/snellius_jobs.sh"
load_vocode_job "$job_name"

echo "Submitting repair job"
echo "job_name: $job_name"
echo "missing_list: $missing_list"
echo "output_dir: $JOB_OUTPUT_DIR"

sbatch \
    --chdir="$repo_root" \
    --job-name="${job_name}_repair" \
    --export=ALL,JOB_NAME="$job_name",MISSING_LIST="$missing_list" \
    "$repo_root/scripts/sbatch_repair_vocode_job.sh"
