#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="$ROOT_DIR/work/tmp"
CLN22UL_ROOT="${QWEN_HLS_22NM_ROOT:-/home/yang/tools/arm/tsmc/cln22ul}"
STD_CELL_FLAVOR="${QWEN_HLS_22NM_STD_CELL_FLAVOR:-sc7mcpp140z_base_svt_c35}"
STD_CELL_REV="${QWEN_HLS_22NM_STD_CELL_REV:-r2p0}"
DB_BASENAME="${QWEN_HLS_22NM_DB_BASENAME:-sc7mcpp140z_cln22ul_base_svt_c35_tt_typical_max_0p80v_25c.db}"

find_latest_solution_dir() {
  find "$ROOT_DIR" -maxdepth 2 -path "*/qwen_prefill_glue_top_v1_solution.v1/concat_rtl.v.dc" -print \
    | sort -V \
    | tail -n1 \
    | xargs -r dirname
}

SOLUTION_DIR="${1:-$(find_latest_solution_dir)}"
if [[ -z "$SOLUTION_DIR" ]]; then
  echo "No qwen_prefill_glue_top_v1 solution with concat_rtl.v.dc found" >&2
  exit 1
fi

ORIG_DC_TCL="$SOLUTION_DIR/concat_rtl.v.dc"
ORIG_SDC="$SOLUTION_DIR/concat_rtl.v.dc.sdc"
if [[ ! -f "$ORIG_DC_TCL" || ! -f "$ORIG_SDC" ]]; then
  echo "Generated DC inputs not found under $SOLUTION_DIR" >&2
  exit 1
fi

DB_DIR="$CLN22UL_ROOT/$STD_CELL_FLAVOR/$STD_CELL_REV/db"
DB_PATH="$DB_DIR/$DB_BASENAME"
if [[ ! -f "$DB_PATH" ]]; then
  echo "22nm target db not found: $DB_PATH" >&2
  exit 1
fi

LIB_BASENAME="${DB_BASENAME%.db}"
mkdir -p "$LOG_DIR"
PATCHED_DC_TCL="$LOG_DIR/$(basename "$SOLUTION_DIR")_dc_22nm.tcl"
PATCHED_SDC="$LOG_DIR/$(basename "$SOLUTION_DIR")_dc_22nm.sdc"
LOG_FILE="$LOG_DIR/$(basename "$(dirname "$SOLUTION_DIR")")_$(basename "$SOLUTION_DIR")_dc_22nm.log"

sed \
  -e 's/^set_operating_conditions /# set_operating_conditions /' \
  -e 's/^set_driving_cell /# set_driving_cell /' \
  "$ORIG_SDC" > "$PATCHED_SDC"

sed \
  -e "s|^set target_library .*|set target_library $LIB_BASENAME|" \
  -e "s|^set link_library .*|set link_library {* $LIB_BASENAME dw_foundation.sldb standard.sldb}|" \
  -e "/^set synthetic_library /a\\
set search_path [concat \\\$search_path [list $DB_DIR]]" \
  -e "s|^read_sdc .*|read_sdc $PATCHED_SDC -version 1.7|" \
  "$ORIG_DC_TCL" > "$PATCHED_DC_TCL"

dc_shell -f "$PATCHED_DC_TCL" 2>&1 | tee "$LOG_FILE"

echo "Patched DC TCL: $PATCHED_DC_TCL"
echo "Patched SDC: $PATCHED_SDC"
echo "DC log written to $LOG_FILE"
