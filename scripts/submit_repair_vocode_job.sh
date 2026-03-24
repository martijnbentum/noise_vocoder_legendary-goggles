#!/bin/bash
# Submit a repair job that finds and reruns only missing wav files.

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

echo 'Submitting repair job'
echo "job_name: $job_name"
echo "output_dir: $JOB_OUTPUT_DIR"

sbatch \
    --chdir="$repo_root" \
    --job-name="${job_name}_repair" \
    --export=ALL,JOB_NAME="$job_name" \
    "$repo_root/scripts/sbatch_repair_vocode_job.sh"
