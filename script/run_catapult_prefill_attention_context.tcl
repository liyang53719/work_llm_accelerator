set script_dir [file dirname [file normalize [info script]]]
set repo_root [file dirname $script_dir]
set hls_root [file join $repo_root hls]
set mgc_home ""

if {[info exists ::env(MGC_HOME)] && $::env(MGC_HOME) ne ""} {
	set mgc_home $::env(MGC_HOME)
} else {
	set mgc_home [file dirname [file dirname [file normalize [info nameofexecutable]]]]
}

set solution_name qwen_prefill_attention_context_solution
set top_function qwen_prefill_attention_context_query_tile_stream_catapult
set clock_period 2.0

if {[info exists ::env(QWEN_HLS_CLOCK_PERIOD)] && $::env(QWEN_HLS_CLOCK_PERIOD) ne ""} {
	set clock_period $::env(QWEN_HLS_CLOCK_PERIOD)
}

set design_files [list \
	[file join $hls_root prefill_only qwen_prefill_attention_context_stage_catapult.cpp] \
	[file join $hls_root prefill_only qwen_prefill_attention_kernel.h] \
	[file join $hls_root common llm_accel_types.h] \
	[file join $hls_root common qwen2_model_config.h]]

foreach design_file $design_files {
	if {![file exists $design_file]} {
		error "Required Catapult input file not found: $design_file"
	}
	if {![file readable $design_file]} {
		error "Required Catapult input file is not readable: $design_file"
	}
}

set search_path [join [list \
	[file join $hls_root prefill_only] \
	[file join $hls_root common] \
	[file join $hls_root include] \
	[file join $hls_root catapult_shims] \
	$hls_root \
	[file join $mgc_home shared include] \
	/usr/include/c++/10 \
	/usr/include/x86_64-linux-gnu/c++/10 \
	/usr/lib/gcc/x86_64-linux-gnu/10/include \
	/usr/include \
	/usr/include/x86_64-linux-gnu] " "]

set compiler_flags [list \
	"-I[file join $hls_root prefill_only]" \
	"-I[file join $hls_root common]" \
	"-I[file join $hls_root include]" \
	"-I[file join $hls_root catapult_shims]" \
	"-I$hls_root" \
	"-I[file join $mgc_home shared include]" \
	"-include limits.h" \
	"-include climits" \
	"-D_GCC_LIMITS_H_" \
	"-D_LIBC_LIMITS_H_" \
	"-D__EDG__" \
	"-I/usr/include/c++/10" \
	"-I/usr/include/x86_64-linux-gnu/c++/10" \
	"-I/usr/lib/gcc/x86_64-linux-gnu/10/include" \
	"-I/usr/include" \
	"-I/usr/include/x86_64-linux-gnu"]

proc emit_general_solution_metrics {} {
	set general {timing tm_latency_cycles \
				 timing tm_latency_time \
				 timing tm_thruput_cycles \
				 timing tm_thruput_time \
				 area total \
				 timing slack}

	foreach {type col} $general {
		set cpath /DATUM/FIELDS/${type}/COLUMNS
		if {[catch {set name [solution get ${cpath}/${col}/name]}] || [catch {set value [solution get ${cpath}/${col}/VALUE]}]} {
			continue
		}
		puts "QWEN_METRIC $type/$col $name $value"
	}
}

cd $script_dir

options defaults
solution new $solution_name
solution options set /Input/SearchPath $search_path
solution options set /Input/CompilerFlags [join $compiler_flags " "]
solution options set /Input/CppStandard c++14
solution options set /Flows/SCVerify/USE_CCS_BLOCK true

project new
foreach design_file $design_files {
	solution file add $design_file
}

go new
go analyze
solution design set $top_function -top
go compile

solution library add nangate-45nm_beh -- -rtlsyntool OasysRTL
solution library add ccs_sample_mem

go libraries
directive set -CLOCKS [list clk [list -CLOCK_PERIOD $clock_period]]
directive set /$top_function/core -MEM_MAP_THRESHOLD 129
directive set SCHED_USE_MULTICYCLE true

go assembly
go architect
emit_general_solution_metrics

if {![info exists ::env(QWEN_HLS_ENABLE_EXTRACT)] || $::env(QWEN_HLS_ENABLE_EXTRACT) ne "0"} {
	go extract
	emit_general_solution_metrics
}

solution report
exit