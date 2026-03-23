#!/bin/bash
#SBATCH --job-name=vocoder-default-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -eu

repo_root="${SLURM_SUBMIT_DIR:-$PWD}"

"$repo_root/scripts/run_snellius_vocode.sh" \
    default_family \
    8_band \
    8 \
    /projects/0/prjs1489/data/vocoded_nband-8_spider_nl/wav/
