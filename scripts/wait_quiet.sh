#!/usr/bin/env bash
# Block until this box's 1-minute load average drops below $1 (default 3).
#
# Benchmarks on this project have had verdicts INVERTED by a contended box
# (`mlrs-cpu-bench-separate-processes`), and a release build leaves loadavg
# decaying for minutes after it finishes. Gate the sweep on the machine being
# quiet rather than on a guess about how long that takes.
set -euo pipefail
threshold="${1:-3}"
while [ "$(cut -d. -f1 /proc/loadavg)" -ge "$threshold" ]; do
    sleep 15
done
echo "loadavg quiet: $(cat /proc/loadavg)"
