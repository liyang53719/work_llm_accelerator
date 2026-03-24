#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="$ROOT_DIR/work/tmp"
DEFAULT_VERDI_HOME="/home/yang/tools/synopsys/verdi/W-2024.09"

find_latest_solution_dir() {
  find "$ROOT_DIR" -maxdepth 3 -path "*/qwen_prefill_glue_top_v1_solution.v1/scverify/Verify_orig_cxx_osci.mk" -print \
    | sort -V \
    | tail -n1 \
    | xargs -r dirname \
    | xargs -r dirname
}

SOLUTION_DIR="${1:-$(find_latest_solution_dir)}"
if [[ -z "$SOLUTION_DIR" ]]; then
  echo "No qwen_prefill_glue_top_v1 solution with scverify/Verify_orig_cxx_osci.mk found" >&2
  exit 1
fi

SCVERIFY_DIR="$SOLUTION_DIR/scverify"
MAKEFILE="$SCVERIFY_DIR/Verify_orig_cxx_osci.mk"
if [[ ! -f "$MAKEFILE" ]]; then
  echo "SCVerify makefile not found: $MAKEFILE" >&2
  exit 1
fi

if [[ -z "${Novas_NOVAS_INST_DIR:-}" && -d "$DEFAULT_VERDI_HOME" ]]; then
  export Novas_NOVAS_INST_DIR="$DEFAULT_VERDI_HOME"
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "$(dirname "$SOLUTION_DIR")")_$(basename "$SOLUTION_DIR")_scverify_fsdb.log"

make -C "$SCVERIFY_DIR" \
  -f "$(basename "$MAKEFILE")" \
  SCVerify_WAVE_PROBES=true \
  LowPower_SWITCHING_ACTIVITY_TYPE=fsdb \
  2>&1 | tee "$LOG_FILE"

echo "SCVerify log written to $LOG_FILE"
