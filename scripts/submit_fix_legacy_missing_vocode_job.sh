#!/bin/bash
# Submit a repair job for legacy missing-list entries that still point to
# old output paths instead of source wav inputs.

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

echo "Submitting legacy missing repair job"
echo "job_name: $job_name"
echo "missing_list: $missing_list"
echo "legacy_output_dir: $JOB_OUTPUT_DIR"
echo "output_dir: $JOB_OUTPUT_DIR"

sbatch \
    --chdir="$repo_root" \
    --job-name="${job_name}_legacy_fix" \
    --export=ALL,JOB_NAME="$job_name",MISSING_LIST="$missing_list" \
    "$repo_root/scripts/sbatch_fix_legacy_missing_vocode_job.sh"
