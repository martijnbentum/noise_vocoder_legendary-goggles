#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -eu

: "${JOB_NAME:?JOB_NAME is not set}"

repo_root="${SLURM_SUBMIT_DIR:-$PWD}"

"$repo_root/scripts/run_repair_vocode.sh" "$JOB_NAME"
