#!/usr/bin/env bash
# Block until the 1-minute load average drops below $1 (default 4).
#
# A loaded box has INVERTED perf verdicts on this project more than once, and
# two sessions share this machine — so a sweep waits for quiet rather than
# recording whatever it happened to get.
set -euo pipefail
limit="${1:-4}"
while true; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if [ "${load%%.*}" -lt "$limit" ]; then
        echo "box quiet, loadavg $load"
        exit 0
    fi
    sleep 30
done
