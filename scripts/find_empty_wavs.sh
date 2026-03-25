#!/bin/bash
# Find zero-byte wav files under a directory and record them in archive.

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
        *)
            break
            ;;
    esac
done

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [--cpus n] <directory>" >&2
    exit 1
fi

target_dir="$1"
nprocess="$cpus"

if [ ! -d "$target_dir" ]; then
    echo "Directory does not exist: $target_dir" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
archive_dir="$repo_root/archive"

target_dir=$(cd "$target_dir" && pwd)
dir_name=$(basename "$target_dir")
output_file="$archive_dir/empty_files_${dir_name}.txt"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/find_empty_wavs.XXXXXX")

mkdir -p "$archive_dir"

cleanup() {
    rm -rf "$tmp_dir"
}

trap cleanup EXIT

find "$target_dir" -type f -name '*.wav' -print0 \
    | xargs -0 -n 256 -P "$nprocess" bash -c '
        chunk_file="$1"
        shift
        for filename in "$@"; do
            if [ ! -s "$filename" ]; then
                printf "%s\n" "$filename" >> "$chunk_file"
            fi
        done
    ' _ "$tmp_dir/chunk.$RANDOM.$$.txt"

if find "$tmp_dir" -type f -print -quit | grep -q .; then
    find "$tmp_dir" -type f -print0 \
        | xargs -0 cat \
        | LC_ALL=C sort -u > "$output_file"
else
    : > "$output_file"
fi

echo "$output_file"
