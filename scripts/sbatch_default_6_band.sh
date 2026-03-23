#!/bin/bash
#SBATCH --job-name=vocoder-default-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -eu

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_snellius_vocode.sh" \
    default_family \
    6_band \
    6 \
    /projects/0/prjs1489/data/spidr/vocoded_bands-6_spidr/wav
