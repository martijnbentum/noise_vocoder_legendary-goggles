#!/bin/bash
#SBATCH --job-name=vocoder-default-16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -eu

repo_root="${SLURM_SUBMIT_DIR:-$PWD}"
: "${OUTPUT_DIR:?OUTPUT_DIR is not set}"

"$repo_root/scripts/run_snellius_vocode.sh" \
    default_family \
    16_band \
    16 \
    "$OUTPUT_DIR"
