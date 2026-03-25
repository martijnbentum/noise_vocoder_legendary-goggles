#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.err

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <wav_dir> [archive.tar] [strip_top_level]" >&2
    echo "strip_top_level: 0 keeps the folder name, 1 archives its contents" >&2
    exit 1
fi

wav_dir="$1"
archive_path="${2:-}"
strip_top_level="${3:-0}"

if [ ! -d "$wav_dir" ]; then
    echo "WAV directory does not exist: $wav_dir" >&2
    exit 1
fi

case "$strip_top_level" in
    0|1) ;;
    *)
        echo "strip_top_level must be 0 or 1: $strip_top_level" >&2
        exit 1
        ;;
esac

wav_dir=$(cd "$wav_dir" && pwd)
parent_dir=$(dirname "$wav_dir")
folder_name=$(basename "$wav_dir")

if [ -z "$archive_path" ]; then
    archive_path="$PWD/${folder_name}.tar"
fi

case "$archive_path" in
    *.tar) ;;
    *)
        echo "Archive path must end with .tar: $archive_path" >&2
        exit 1
        ;;
esac

archive_dir=$(dirname "$archive_path")
archive_name=$(basename "$archive_path")
mkdir -p "$archive_dir"
archive_dir=$(cd "$archive_dir" && pwd)
archive_path="$archive_dir/$archive_name"

echo "=== Snellius wav tar job ==="
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID:-none}"
echo "job_name: ${SLURM_JOB_NAME:-none}"
echo "wav_dir: $wav_dir"
echo "archive_path: $archive_path"
echo "strip_top_level: $strip_top_level"
echo "============================"

if [ "$strip_top_level" = "1" ]; then
    tar -cf "$archive_path" -C "$wav_dir" .
else
    tar -cf "$archive_path" -C "$parent_dir" "$folder_name"
fi

echo "Archive created:"
ls -lh "$archive_path"

echo "Archive contents preview:"
tar -tf "$archive_path" | head
