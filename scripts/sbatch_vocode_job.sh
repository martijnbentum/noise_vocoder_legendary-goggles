#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -eu

: "${JOB_NAME:?JOB_NAME is not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR is not set}"

repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
. "$repo_root/scripts/snellius_jobs.sh"
load_vocode_job "$JOB_NAME"

"$repo_root/scripts/run_snellius_vocode.sh" \
    "$JOB_NAME" \
    "$JOB_FAMILY" \
    "$JOB_KEY" \
    "$JOB_NBANDS" \
    "$OUTPUT_DIR"
