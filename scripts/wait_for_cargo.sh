#!/usr/bin/env bash
# Block until no cargo process is running, then exit. For waiting out a build
# without chaining sleeps.
set -euo pipefail
while pgrep -c cargo > /dev/null 2>&1; do
    sleep 30
done
echo "cargo finished"
