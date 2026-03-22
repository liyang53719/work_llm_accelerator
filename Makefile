CATAPULT ?= catapult
TCL_PREFILL := hls/prefill_only/run_catapult_prefill_attention.tcl
TCL_PREFILL_ATTENTION_STREAM := script/run_catapult_prefill_attention_stream.tcl
LOG_DIR := work/tmp
LOG_FILE := $(LOG_DIR)/catapult_prefill_latest.log
LOG_FILE_PREFILL_ATTENTION_STREAM := $(LOG_DIR)/catapult_prefill_attention_stream_latest.log

.PHONY: catapult_prefill catapult_prefill_attention_stream clean
catapult_prefill:
	mkdir -p $(LOG_DIR)
	$(CATAPULT) -shell -file $(TCL_PREFILL) | tee $(LOG_FILE)

catapult_prefill_attention_stream:
	mkdir -p $(LOG_DIR)
	$(CATAPULT) -shell -file $(TCL_PREFILL_ATTENTION_STREAM) | tee $(LOG_FILE_PREFILL_ATTENTION_STREAM)

clean:
	rm -rf Catapult_*
	rm -f Catapult_*.ccs
	rm -f catapult.log
	rm -f $(LOG_FILE) $(LOG_FILE_PREFILL_ATTENTION_STREAM)
