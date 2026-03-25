#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -eu

: "${JOB_NAME:?JOB_NAME is not set}"
: "${TEXT_FILENAME:?TEXT_FILENAME is not set}"

repo_root="${SLURM_SUBMIT_DIR:-$PWD}"

"$repo_root/scripts/run_filename_list_vocode.sh" \
    "$JOB_NAME" \
    "$TEXT_FILENAME"
