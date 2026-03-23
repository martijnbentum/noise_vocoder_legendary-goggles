#!/bin/bash
# Run a configured vocoding batch on Snellius.

set -eu

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <family> <key> <n_bands> <output_dir>" >&2
    exit 1
fi

family="$1"
key="$2"
n_bands="$3"
output_dir="$4"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
env_dir="$repo_root/.venv-snellius"
input_dir="/projects/0/prjs1489/data/spidr/wav"
nprocess="${SLURM_CPUS_PER_TASK:-64}"

if [ ! -x "$env_dir/bin/python" ]; then
    "$script_dir/build_snellius_env.sh"
fi

source "$env_dir/bin/activate"

mkdir -p "$output_dir"
export OMP_NUM_THREADS=1

python -m vocoder \
    --input_dir "$input_dir" \
    --output_dir "$output_dir" \
    --nprocess "$nprocess" \
    --nbands "$n_bands" \
    --frequency_family "$family" \
    --frequency_key "$key"
