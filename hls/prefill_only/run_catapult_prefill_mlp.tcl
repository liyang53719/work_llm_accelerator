set script_dir [file dirname [file normalize [info script]]]
set project_root [file dirname $script_dir]
set mgc_home ""
set gcc_root ""
set gcc_version ""
set solution_name qwen_prefill_mlp_kernel_solution
set top_function qwen_prefill_mlp_kernel
set clock_period 1.0
set enable_fp_operator_mapping 0
set use_legacy_dw_lib 0

if {[info exists ::env(MGC_HOME)] && $::env(MGC_HOME) ne ""} {
	set mgc_home $::env(MGC_HOME)
} else {
	set mgc_home [file dirname [file dirname [file normalize [info nameofexecutable]]]]
}

set gcc_root [lindex [glob -nocomplain -directory [file join $mgc_home pkgs dcs_gcc] gcc-*] 0]
if {$gcc_root ne ""} {
	set gcc_version [string range [file tail $gcc_root] 4 end]
}

if {[info exists ::env(QWEN_HLS_CLOCK_PERIOD)] && $::env(QWEN_HLS_CLOCK_PERIOD) ne ""} {
	set clock_period $::env(QWEN_HLS_CLOCK_PERIOD)
}

if {[info exists ::env(QWEN_HLS_USE_FP_OPERATOR_MAPPING)] && $::env(QWEN_HLS_USE_FP_OPERATOR_MAPPING) ne "0"} {
	set enable_fp_operator_mapping 1
}

if {[info exists ::env(QWEN_HLS_USE_LEGACY_DW_LIB)] && $::env(QWEN_HLS_USE_LEGACY_DW_LIB) ne "0"} {
	set use_legacy_dw_lib 1
	set enable_fp_operator_mapping 1
}

set design_files [list \
	[file join $script_dir qwen_prefill_mlp_kernel.cpp]]

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
	[file join $project_root include] \
	[file join $project_root catapult_shims] \
	$project_root \
	$mgc_home/shared/include \
	[file join $gcc_root include c++ $gcc_version] \
	[file join $gcc_root include c++ $gcc_version x86_64-linux-gnu] \
	[file join $gcc_root include c++ $gcc_version backward] \
	[file join $gcc_root lib gcc x86_64-linux-gnu $gcc_version include] \
	/usr/include \
	/usr/include/x86_64-linux-gnu] " "]

set compiler_flags [list \
	-DHLS_CATAPULT \
	"-I$script_dir" \
	"-I[file join $project_root common]" \
	"-I[file join $project_root include]" \
	"-I[file join $project_root catapult_shims]" \
	"-I$project_root" \
	"-I$mgc_home/shared/include" \
	"-D__EDG__" \
	"-I[file join $gcc_root include c++ $gcc_version]" \
	"-I[file join $gcc_root include c++ $gcc_version x86_64-linux-gnu]" \
	"-I[file join $gcc_root include c++ $gcc_version backward]" \
	"-I[file join $gcc_root lib gcc x86_64-linux-gnu $gcc_version include]" \
	"-I/usr/include" \
	"-I/usr/include/x86_64-linux-gnu"]

if {$enable_fp_operator_mapping} {
	lappend compiler_flags "-DQWEN_CATAPULT_USE_FP_OPERATOR_MAPPING=1"
}

if {$use_legacy_dw_lib} {
	lappend compiler_flags "-DUSING_CCS_LEGACY_DW_LIB"
}

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

proc source_saed_setup {mgc_home} {
	set saed_setup_tcl [file join $mgc_home pkgs siflibs saed setup_saedlib.tcl]
	if {![file exists $saed_setup_tcl]} {
		error "Required SAED setup script not found: $saed_setup_tcl"
	}
	source $saed_setup_tcl
}

cd $script_dir

options defaults
project new
flow package require /SCVerify
flow package require /DesignCompiler
solution new $solution_name
solution options set /Input/SearchPath $search_path
solution options set /Input/CompilerFlags [join $compiler_flags " "]
solution options set /Input/CppStandard c++11
solution options set /Flows/SCVerify/USE_CCS_BLOCK true
foreach design_file $design_files {
	solution file add $design_file
}

go new
go analyze
solution design set $top_function -top
go compile

solution library add saed32rvt_tt0p78v125c_dw_beh -- -rtlsyntool DesignCompiler -vendor SAED32 -technology {rvt tt0p78v125c}
solution library add ccs_sample_mem

go libraries
source_saed_setup $mgc_home
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