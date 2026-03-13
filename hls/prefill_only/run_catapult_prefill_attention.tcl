set script_dir [file dirname [file normalize [info script]]]
set project_root [file dirname $script_dir]
set mgc_home /home/yang/tools/Siemens_EDA/Catapult_Synthesis_2022.2-1008433/Mgc_home
set gcc_root $mgc_home/pkgs/dcs_gcc/gcc-10.3.0
set solution_name qwen_prefill_attention_kernel_solution
set top_function llm_accel::qwen_prefill_attention_kernel_catapult
set clock_period 1.0

if {[info exists ::env(QWEN_HLS_CLOCK_PERIOD)] && $::env(QWEN_HLS_CLOCK_PERIOD) ne ""} {
	set clock_period $::env(QWEN_HLS_CLOCK_PERIOD)
}

set design_files [list \
	[file join $script_dir qwen_prefill_attention_kernel.cpp] \
	[file join $script_dir qwen_prefill_attention_kernel.h] \
	[file join $project_root common llm_accel_types.h] \
	[file join $project_root common qwen2_model_config.h]]

foreach design_file $design_files {
	if {![file exists $design_file]} {
		error "Required Catapult input file not found: $design_file"
	}
	if {![file readable $design_file]} {
		error "Required Catapult input file is not readable: $design_file"
	}
}

set search_path [join [list \
	$script_dir \
	[file join $project_root common] \
	[file join $project_root catapult_shims] \
	$project_root \
	$mgc_home/shared/include \
	[file join $gcc_root include c++ 10.3.0] \
	[file join $gcc_root include c++ 10.3.0 x86_64-linux-gnu] \
	[file join $gcc_root lib gcc x86_64-linux-gnu 10.3.0 include-fixed] \
	[file join $gcc_root lib gcc x86_64-linux-gnu 10.3.0 include] \
	[file join $project_root catapult_shims_post] \
	/usr/include/c++/10 \
	/usr/include/x86_64-linux-gnu/c++/10 \
	/usr/lib/gcc/x86_64-linux-gnu/10/include-fixed \
	/usr/lib/gcc/x86_64-linux-gnu/10/include \
	/usr/include \
	/usr/include/x86_64-linux-gnu] " "]

set compiler_flags [list \
	"-I$script_dir" \
	"-I[file join $project_root common]" \
	"-I[file join $project_root catapult_shims]" \
	"-I$project_root" \
	"-I$mgc_home/shared/include" \
	"-I[file join $gcc_root include c++ 10.3.0]" \
	"-I[file join $gcc_root include c++ 10.3.0 x86_64-linux-gnu]" \
	"-I[file join $gcc_root lib gcc x86_64-linux-gnu 10.3.0 include-fixed]" \
	"-I[file join $gcc_root lib gcc x86_64-linux-gnu 10.3.0 include]" \
	"-I[file join $project_root catapult_shims_post]" \
	"-I/usr/include/c++/10" \
	"-I/usr/include/x86_64-linux-gnu/c++/10" \
	"-I/usr/lib/gcc/x86_64-linux-gnu/10/include-fixed" \
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
solution options set /Input/CppStandard c++17
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