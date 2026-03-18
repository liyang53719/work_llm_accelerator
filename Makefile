CATAPULT ?= catapult
TCL_PREFILL := hls/prefill_only/run_catapult_prefill_attention.tcl
TCL_PREFILL_ATTENTION_CONTEXT := script/run_catapult_prefill_attention_context.tcl
TCL_PREFILL_ATTENTION_KV_CACHE := script/run_catapult_prefill_attention_kv_cache.tcl
TCL_PREFILL_ATTENTION_Q_CONTEXT_OUTPUT := script/run_catapult_prefill_attention_q_context_output.tcl
LOG_DIR := work/tmp
LOG_FILE := $(LOG_DIR)/catapult_prefill_latest.log
LOG_FILE_PREFILL_ATTENTION_CONTEXT := $(LOG_DIR)/catapult_prefill_attention_context_latest.log
MONITOR_FILE_PREFILL_ATTENTION_CONTEXT := $(LOG_DIR)/catapult_prefill_attention_context_monitor.log
LOG_FILE_PREFILL_ATTENTION_KV_CACHE := $(LOG_DIR)/catapult_prefill_attention_kv_cache_latest.log
LOG_FILE_PREFILL_ATTENTION_Q_CONTEXT_OUTPUT := $(LOG_DIR)/catapult_prefill_attention_q_context_output_latest.log

.PHONY: catapult_prefill catapult_prefill_attention_context catapult_prefill_attention_kv_cache catapult_prefill_attention_q_context_output clean
catapult_prefill:
	mkdir -p $(LOG_DIR)
	$(CATAPULT) -shell -file $(TCL_PREFILL) | tee $(LOG_FILE)

catapult_prefill_attention_context:
	mkdir -p $(LOG_DIR)
	bash script/run_catapult_with_memory_guard.sh \
		$(CATAPULT) \
		$(TCL_PREFILL_ATTENTION_CONTEXT) \
		$(LOG_FILE_PREFILL_ATTENTION_CONTEXT) \
		$(MONITOR_FILE_PREFILL_ATTENTION_CONTEXT)

catapult_prefill_attention_kv_cache:
	mkdir -p $(LOG_DIR)
	$(CATAPULT) -shell -file $(TCL_PREFILL_ATTENTION_KV_CACHE) | tee $(LOG_FILE_PREFILL_ATTENTION_KV_CACHE)

catapult_prefill_attention_q_context_output:
	mkdir -p $(LOG_DIR)
	$(CATAPULT) -shell -file $(TCL_PREFILL_ATTENTION_Q_CONTEXT_OUTPUT) | tee $(LOG_FILE_PREFILL_ATTENTION_Q_CONTEXT_OUTPUT)

clean:
	rm -rf Catapult_*
	rm -f Catapult_*.ccs
	rm -f catapult.log
	rm -f $(LOG_FILE) $(LOG_FILE_PREFILL_ATTENTION_CONTEXT) $(LOG_FILE_PREFILL_ATTENTION_KV_CACHE) $(LOG_FILE_PREFILL_ATTENTION_Q_CONTEXT_OUTPUT)
