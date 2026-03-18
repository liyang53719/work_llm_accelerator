#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <catapult-bin> <tcl-file> <log-file> <monitor-log>" >&2
  exit 2
fi

catapult_bin="$1"
tcl_file="$2"
log_file="$3"
monitor_log="$4"

threshold_kb="${QWEN_HLS_MEMORY_LIMIT_KB:-52428800}"
poll_interval="${QWEN_HLS_MEMORY_POLL_SEC:-5}"

log_dir="$(dirname "$log_file")"
mkdir -p "$log_dir"
mkdir -p "$(dirname "$monitor_log")"

pipe_path="$log_dir/.catapult_memory_guard_$$.pipe"
rm -f "$pipe_path"
mkfifo "$pipe_path"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_monitor() {
  local message="$1"
  printf '%s %s\n' "$(timestamp)" "$message" | tee -a "$monitor_log"
}

collect_child_pids() {
  local parent_pid="$1"
  ps -o pid= --ppid "$parent_pid" | awk '{print $1}'
}

sum_rss_tree_kb() {
  local root_pid="$1"
  local total_kb="0"
  local rss_kb

  if ! kill -0 "$root_pid" 2>/dev/null; then
    echo 0
    return
  fi

  rss_kb="$(ps -o rss= -p "$root_pid" | awk '{sum += $1} END {print sum + 0}')"
  total_kb=$((total_kb + rss_kb))

  local child_pid
  while IFS= read -r child_pid; do
    [[ -z "$child_pid" ]] && continue
    total_kb=$((total_kb + $(sum_rss_tree_kb "$child_pid")))
  done < <(collect_child_pids "$root_pid")

  echo "$total_kb"
}

extract_last_logged_memory_kb() {
  if [[ ! -f "$log_file" ]]; then
    echo 0
    return
  fi

  awk '
    match($0, /memory usage ([0-9]+)kB/, captures) { value = captures[1] }
    END { print value + 0 }
  ' "$log_file"
}

kill_tree() {
  local root_pid="$1"
  local child_pid
  while IFS= read -r child_pid; do
    [[ -z "$child_pid" ]] && continue
    kill_tree "$child_pid"
  done < <(collect_child_pids "$root_pid")

  kill -TERM "$root_pid" 2>/dev/null || true
}

cleanup() {
  rm -f "$pipe_path"
}

trap cleanup EXIT

: > "$log_file"
: > "$monitor_log"

tee -a "$log_file" < "$pipe_path" &
tee_pid=$!

"$catapult_bin" -shell -file "$tcl_file" > "$pipe_path" 2>&1 &
catapult_pid=$!

log_monitor "QWEN_MONITOR start threshold_kb=$threshold_kb poll_sec=$poll_interval catapult_pid=$catapult_pid tcl=$tcl_file"

guard_triggered=0
guard_reason=""

while kill -0 "$catapult_pid" 2>/dev/null; do
  rss_tree_kb="$(sum_rss_tree_kb "$catapult_pid")"
  logged_memory_kb="$(extract_last_logged_memory_kb)"
  log_monitor "QWEN_MONITOR sample rss_tree_kb=$rss_tree_kb logged_memory_kb=$logged_memory_kb threshold_kb=$threshold_kb"

  if (( rss_tree_kb > threshold_kb )); then
    guard_triggered=1
    guard_reason="rss_tree_kb=$rss_tree_kb"
    break
  fi

  if (( logged_memory_kb > threshold_kb )); then
    guard_triggered=1
    guard_reason="logged_memory_kb=$logged_memory_kb"
    break
  fi

  sleep "$poll_interval"
done

if (( guard_triggered == 1 )); then
  log_monitor "QWEN_MONITOR threshold_exceeded $guard_reason action=kill"
  kill_tree "$catapult_pid"
  sleep 2
  kill -KILL "$catapult_pid" 2>/dev/null || true
fi

set +e
wait "$catapult_pid"
catapult_status=$?
wait "$tee_pid"
tee_status=$?
set -e

if (( guard_triggered == 1 )); then
  log_monitor "QWEN_MONITOR exit status=148 reason=$guard_reason"
  exit 148
fi

log_monitor "QWEN_MONITOR exit status=$catapult_status tee_status=$tee_status"
exit "$catapult_status"