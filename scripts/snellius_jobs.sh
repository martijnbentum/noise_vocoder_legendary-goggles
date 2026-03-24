#!/bin/bash

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "This script only defines Snellius job metadata." >&2
    echo "Use ./scripts/submit_vocode_job.sh --list to inspect jobs." >&2
    exit 1
fi

load_vocode_job() {
    case "$1" in
        default_4_band)
            JOB_FAMILY=default_family
            JOB_KEY=4_band
            JOB_NBANDS=4
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-4_spidr/wav/
            # JOB_FREQUENCIES=50 178 632 2249 7999
            JOB_FREQUENCIES='50 178 632 2249 7999'
            ;;
        default_6_band)
            JOB_FAMILY=default_family
            JOB_KEY=6_band
            JOB_NBANDS=6
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-6_spidr/wav/
            # JOB_FREQUENCIES=50 229 558 1161 2265 4290 7999
            JOB_FREQUENCIES='50 229 558 1161 2265 4290 7999'
            ;;
        default_8_band)
            JOB_FAMILY=default_family
            JOB_KEY=8_band
            JOB_NBANDS=8
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-8_spidr/wav/
            # JOB_FREQUENCIES=50 94 178 335 632 1193 2249 4242 7999
            JOB_FREQUENCIES='50 94 178 335 632 1193 2249 4242 7999'
            ;;
        default_16_band)
            JOB_FAMILY=default_family
            JOB_KEY=16_band
            JOB_NBANDS=16
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-16_spidr/wav/
            # JOB_FREQUENCIES=50 69 94 129 178 244 335 461 632 868 1193 1638 2249 3089 4242 5825 7999
            JOB_FREQUENCIES='50 69 94 129 178 244 335 461 632 868 1193 1638 2249 3089 4242 5825 7999'
            ;;
        speech_weighted_8_band)
            JOB_FAMILY=speech_weighted
            JOB_KEY=8_band
            JOB_NBANDS=8
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-speech_weighted-8_spidr/wav/
            # JOB_FREQUENCIES=50 180 350 600 950 1450 2200 3500 7999
            JOB_FREQUENCIES='50 180 350 600 950 1450 2200 3500 7999'
            ;;
        *)
            echo "Unknown Snellius job: $1" >&2
            return 1
            ;;
    esac
}


iter_vocode_jobs() {
    cat <<'EOF'
default_4_band
default_6_band
default_8_band
default_16_band
speech_weighted_8_band
EOF
}


list_vocode_jobs() {
    while IFS= read -r job_name; do
        load_vocode_job "$job_name"
        cat <<EOF
name: $job_name
  JOB_FAMILY=$JOB_FAMILY
  JOB_KEY=$JOB_KEY
  JOB_NBANDS=$JOB_NBANDS
  JOB_OUTPUT_DIR=$JOB_OUTPUT_DIR
  # JOB_FREQUENCIES=$JOB_FREQUENCIES
EOF
    done <<EOF
$(iter_vocode_jobs)
EOF
}
