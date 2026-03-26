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
submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
repo_root=$(cd "$submit_dir" && pwd)
script_dir="$repo_root/scripts"
if [ "${config_path#/}" = "$config_path" ]; then
    config_path="$repo_root/$config_path"
fi
env_dir="$repo_root/.venv-snellius"
mode="${VOCODER_BATCH_MODE:-bootstrap}"
cpus_per_task=16

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

if [ "$mode" = "group" ]; then
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
    echo "group task: $SLURM_ARRAY_TASK_ID"
    srun python -m vocoder.slurm_batch group \
        "$config_path" \
        "$SLURM_ARRAY_TASK_ID"
    exit 0
fi

if [ "$mode" = "finalize" ]; then
    echo "finalizing run"
    srun python -m vocoder.slurm_batch finalize "$config_path"
    exit 0
fi

read -r n_task_groups max_parallel_tasks run_dir <<EOF
$(python -m vocoder.slurm_batch prepare "$config_path")
EOF

echo "prepared run dir: $run_dir"
echo "task groups: $n_task_groups"
echo "max_parallel_tasks: $max_parallel_tasks"
echo "cpus_per_task: $cpus_per_task"

array_job_id=$(sbatch \
    --parsable \
    --array="0-$(( n_task_groups - 1 ))%$max_parallel_tasks" \
    --cpus-per-task="$cpus_per_task" \
    --job-name="vocode_group" \
    --output="$repo_root/slurm_out/%x-%A_%a.out" \
    --error="$repo_root/slurm_out/%x-%A_%a.err" \
    --export=ALL,VOCODER_BATCH_MODE=group \
    "$script_dir/batch_vocode.sbatch" \
    "$config_path")

finalize_job_id=$(sbatch \
    --parsable \
    --dependency="afterany:$array_job_id" \
    --job-name="vocode_finalize" \
    --output="$repo_root/slurm_out/%x-%j.out" \
    --error="$repo_root/slurm_out/%x-%j.err" \
    --export=ALL,VOCODER_BATCH_MODE=finalize \
    "$script_dir/batch_vocode.sbatch" \
    "$config_path")

echo "submitted array job: $array_job_id"
echo "submitted finalize job: $finalize_job_id"
