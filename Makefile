CATAPULT ?= catapult
TCL_PREFILL := hls/prefill_only/run_catapult_prefill_attention.tcl
LOG_DIR := work/tmp
LOG_FILE := $(LOG_DIR)/catapult_prefill_latest.log

.PHONY: catapult_prefill
catapult_prefill:
	mkdir -p $(LOG_DIR)
	$(CATAPULT) -shell -file $(TCL_PREFILL) | tee $(LOG_FILE)
