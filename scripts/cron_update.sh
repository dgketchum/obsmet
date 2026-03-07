#!/usr/bin/env bash
# obsmet daily update — run from cron or systemd timer
#
# Incremental normalize + daily build for all sources.
# The manifest tracks done keys, so re-running is cheap when nothing is new.
#
# Cron example (run at 03:00 daily):
#   0 3 * * * /home/dgketchum/code/obsmet/scripts/cron_update.sh >> /var/log/obsmet/cron_update.log 2>&1
#
# Or install as systemd timer — see scripts/obsmet-update.timer

set -euo pipefail

OBSMET_DIR="/home/dgketchum/code/obsmet"
LOG_DIR="/var/log/obsmet"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/update_${DATE}.log"

echo "=== obsmet update started $(date -Iseconds) ===" | tee -a "$LOG"

cd "$OBSMET_DIR"
uv run python -m obsmet.cli.main update --daily >> "$LOG" 2>&1

echo "=== obsmet update finished $(date -Iseconds) ===" | tee -a "$LOG"
