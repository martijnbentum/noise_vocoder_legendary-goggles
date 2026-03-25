#!/bin/bash
# Check output state locally, then submit the sbatch job.

set -eu

cpus=64

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpus)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --cpus" >&2
                exit 1
            fi
            cpus="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 [--cpus n] <sbatch_script> <output_dir> [job_name]" >&2
    exit 1
fi

sbatch_script="$1"
output_dir="$2"
job_name="${3:-}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

case "$sbatch_script" in
    /*) ;;
    *) sbatch_script="$repo_root/$sbatch_script" ;;
esac

mkdir -p "$output_dir"

existing_wavs=$(find "$output_dir" -type f -name '*.wav' -print -quit)
if [ -n "$existing_wavs" ]; then
    echo "Refusing to submit: output dir already contains wav files" >&2
    echo "output_dir: $output_dir" >&2
    echo "example_file: $existing_wavs" >&2
    exit 1
fi

echo "Submitting $sbatch_script"
echo "output_dir check passed: $output_dir"
echo "cpus_per_task: $cpus"
if [ -n "$job_name" ]; then
    sbatch \
        --chdir="$repo_root" \
        --cpus-per-task="$cpus" \
        --job-name="$job_name" \
        --export=ALL,OUTPUT_DIR="$output_dir",JOB_NAME="$job_name" \
        "$sbatch_script"
    exit 0
fi
sbatch \
    --chdir="$repo_root" \
    --cpus-per-task="$cpus" \
    --export=ALL,OUTPUT_DIR="$output_dir" \
    "$sbatch_script"
