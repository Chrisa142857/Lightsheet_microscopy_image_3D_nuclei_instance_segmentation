#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
MIN_AVAIL_GB=50
SLEEP_CHECK=60   # seconds between memory checks
SLEEP_BETWEEN=60 # seconds after launching a job (your original sleep)

MAX_SCREENS=45
SLEEP_SCREEN_CHECK=60   # seconds between checks

count_nis_screens() {
  local out
  out="$(screen -ls 2>/dev/null || true)"
  awk '/\t[0-9]+\.(nis_[^ ]+)/ {c++} END{print c+0}' <<<"$out"
}
wait_for_screen_slot() {
  local n
  while true; do
    n="$(count_nis_screens)"
    if (( n < MAX_SCREENS )); then
      return 0
    fi
    echo "[WAIT] $(date) nis_ screen sessions=${n} >= ${MAX_SCREENS}; sleeping ${SLEEP_SCREEN_CHECK}s..." >&2
    sleep "$SLEEP_SCREEN_CHECK"
  done
}

# Returns available RAM in GB (integer), using MemAvailable from /proc/meminfo
avail_gb() {
  awk '/^MemAvailable:/ { printf "%d\n", $2/1024/1024 }' /proc/meminfo
}

# Block until available RAM > MIN_AVAIL_GB
wait_for_ram() {
  local cur
  while true; do
    cur="$(avail_gb)"
    if (( cur > MIN_AVAIL_GB )); then
      return 0
    fi
    echo "[WAIT] $(date)  MemAvailable=${cur}GB <= ${MIN_AVAIL_GB}GB; sleeping ${SLEEP_CHECK}s..." >&2
    sleep "$SLEEP_CHECK"
  done
}

mkdir -p logs_pa_p4
starti=$2
i=$2
skipi=$3
while IFS= read -r cmd; do
  i=$((i+1))
  echo "[SUBMITTING] $(date) $((i-starti)) row" >&2
  if (( i <= skipi )); then
    echo "[SKIP] $(date) i=$i is smaller than $skipi" >&2
    continue
  fi

  # Wait here BEFORE starting next screen session
  wait_for_ram
  wait_for_screen_slot
  sess="nis_${i}"
  logfile="logs_pa_p4/${sess}.log"
  
  echo "[SUBMITTED] $(date) $sess" >&2
  screen -dmS "$sess" bash -lc "
    set -euo pipefail
    echo '[START] $(date)  host=$(hostname)  sess=$sess' | tee -a '$logfile'
    echo '[CMD]   $cmd' | tee -a '$logfile'
    $cmd 2>&1 | tee -a '$logfile'
    echo '[DONE]  $(date)  rc=$?' | tee -a '$logfile'
  "
  sleep "$SLEEP_BETWEEN"
done < $1
