#!/bin/bash
# Build the project environment on Snellius.

set -eu

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
env_dir="$repo_root/.venv-snellius"

python3 -m venv "$env_dir"
source "$env_dir/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e "$repo_root"
