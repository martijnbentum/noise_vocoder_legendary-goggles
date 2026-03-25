#!/bin/bash
# Submit a repair job that finds and reruns only missing wav files.

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

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [--cpus n] <job_name>" >&2
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
echo "cpus_per_task: $cpus"

sbatch \
    --chdir="$repo_root" \
    --cpus-per-task="$cpus" \
    --job-name="${job_name}_repair" \
    --export=ALL,JOB_NAME="$job_name" \
    "$repo_root/archive/scripts/sbatch_repair_vocode_job.sh"
