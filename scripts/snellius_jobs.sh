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
            ;;
        default_6_band)
            JOB_FAMILY=default_family
            JOB_KEY=6_band
            JOB_NBANDS=6
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-6_spidr/wav/
            ;;
        default_8_band)
            JOB_FAMILY=default_family
            JOB_KEY=8_band
            JOB_NBANDS=8
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-8_spidr/wav/
            ;;
        default_16_band)
            JOB_FAMILY=default_family
            JOB_KEY=16_band
            JOB_NBANDS=16
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-16_spidr/wav/
            ;;
        speech_weighted_8_band)
            JOB_FAMILY=speech_weighted
            JOB_KEY=8_band
            JOB_NBANDS=8
            JOB_OUTPUT_DIR=/scratch-shared/mbentum1/vocoded_bands-speech_weighted-8_spidr/wav/
            ;;
        *)
            echo "Unknown Snellius job: $1" >&2
            return 1
            ;;
    esac
}


list_vocode_jobs() {
    cat <<'EOF'
default_4_band
default_6_band
default_8_band
default_16_band
speech_weighted_8_band
EOF
}
