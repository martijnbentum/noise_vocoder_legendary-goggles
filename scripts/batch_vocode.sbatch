#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/launcher/%x-%j.out
#SBATCH --error=slurm_out/launcher/%x-%j.err

set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config.json>" >&2
    exit 1
fi

config_path="$1"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
env_dir="$repo_root/.venv-snellius"
mode="${VOCODER_BATCH_MODE:-bootstrap}"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

if [ ! -x "$env_dir/bin/python" ]; then
    "$script_dir/build_snellius_env.sh"
fi

source "$env_dir/bin/activate"
mkdir -p "$repo_root/slurm_out"
mkdir -p "$repo_root/slurm_out/launcher"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ "$mode" = "chunk" ]; then
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
    echo "chunk task: $SLURM_ARRAY_TASK_ID"
    srun python -m vocoder.slurm_batch chunk \
        "$config_path" \
        "$SLURM_ARRAY_TASK_ID"
    exit 0
fi

if [ "$mode" = "finalize" ]; then
    echo "finalizing run"
    srun python -m vocoder.slurm_batch finalize "$config_path"
    exit 0
fi

read -r n_chunks max_parallel_chunks run_dir <<EOF
$(python -m vocoder.slurm_batch prepare "$config_path")
EOF

echo "prepared run dir: $run_dir"
echo "chunks: $n_chunks"
echo "max_parallel_chunks: $max_parallel_chunks"

array_job_id=$(sbatch \
    --parsable \
    --array="0-$(( n_chunks - 1 ))%$max_parallel_chunks" \
    --job-name="vocode_chunk" \
    --output="$repo_root/slurm_out/%x-%A_%a.out" \
    --error="$repo_root/slurm_out/%x-%A_%a.err" \
    --export=ALL,VOCODER_BATCH_MODE=chunk \
    "$0" \
    "$config_path")

finalize_job_id=$(sbatch \
    --parsable \
    --dependency="afterany:$array_job_id" \
    --job-name="vocode_finalize" \
    --output="$repo_root/slurm_out/%x-%j.out" \
    --error="$repo_root/slurm_out/%x-%j.err" \
    --export=ALL,VOCODER_BATCH_MODE=finalize \
    "$0" \
    "$config_path")

echo "submitted array job: $array_job_id"
echo "submitted finalize job: $finalize_job_id"
