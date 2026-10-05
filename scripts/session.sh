#!/usr/bin/env bash
# One full benchmark session (float + INT8), aborted if any NVMe passes 75 C.
set -u
cd "$(dirname "$0")/.."
S=${1:?session}
( bash scripts/sweep.sh $S > runs/sweep_$S.log 2>&1; bash scripts/int8_sweep.sh $S > runs/int8_$S.log 2>&1; echo DONE > runs/session_$S.done ) &
PID=$!
while kill -0 $PID 2>/dev/null; do
  for h in /sys/class/hwmon/hwmon*; do
    if [ "$(cat $h/name)" = nvme ] && [ "$(cat $h/temp1_input)" -gt 75000 ]; then
      echo "NVME_HOT $(cat $h/temp1_input) — aborting"; pkill -P $PID; pkill -f "scripts/bench.py"; exit 1
    fi
  done
  sleep 15
done
echo "SESSION_DONE $S"
