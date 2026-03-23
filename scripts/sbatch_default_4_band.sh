#!/bin/bash
#SBATCH --job-name=vocoder-default-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -eu

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_snellius_vocode.sh" \
    default_family \
    4_band \
    4 \
    /projects/0/prjs1489/data/spidr/vocoded_bands-4_spidr/wav
