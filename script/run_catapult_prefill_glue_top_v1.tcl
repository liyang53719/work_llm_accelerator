set script_dir [file dirname [file normalize [info script]]]
set repo_root [file dirname $script_dir]
set hls_root [file join $repo_root hls]
set mgc_home ""
set gcc_root ""
set gcc_version ""

if {[info exists ::env(MGC_HOME)] && $::env(MGC_HOME) ne ""} {
	set mgc_home $::env(MGC_HOME)
} else {
	set mgc_home [file dirname [file dirname [file normalize [info nameofexecutable]]]]
}

set gcc_root [lindex [glob -nocomplain -directory [file join $mgc_home pkgs dcs_gcc] gcc-*] 0]
if {$gcc_root ne ""} {
	set gcc_version [string range [file tail $gcc_root] 4 end]
}

set solution_name qwen_prefill_glue_top_v1_solution
set top_function llm_accel::qwen_prefill_glue_top_v1_catapult
set clock_period 2.0

if {[info exists ::env(QWEN_HLS_CLOCK_PERIOD)] && $::env(QWEN_HLS_CLOCK_PERIOD) ne ""} {
	set clock_period $::env(QWEN_HLS_CLOCK_PERIOD)
}

set design_files [list \
	[file join $hls_root prefill_only qwen_prefill_glue_top_v1_catapult.cpp] \
	[file join $hls_root prefill_only qwen_prefill_attention_stream_top.cpp] \
	[file join $hls_root prefill_only qwen_prefill_attention_kernel.cpp] \
	[file join $hls_root prefill_only qwen_prefill_mlp_kernel.cpp]]

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
	[file join $gcc_root include c++ $gcc_version] \
	[file join $gcc_root include c++ $gcc_version x86_64-linux-gnu] \
	[file join $gcc_root include c++ $gcc_version backward] \
	[file join $gcc_root lib gcc x86_64-linux-gnu $gcc_version include]] " "]

set compiler_flags [list \
	-DHLS_CATAPULT \
	-DQWEN_HLS_GLUE_INLINE_CHILD_TOPS=1 \
	"-I[file join $hls_root prefill_only]" \
	"-I[file join $hls_root common]" \
	"-I[file join $hls_root include]" \
	"-I[file join $hls_root catapult_shims]" \
	"-I$hls_root" \
	"-D__EDG__"]

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

proc configure_saed_techlib_search_path {} {
	if {[info exists ::env(SAED32_EDK)] && [file isdirectory $::env(SAED32_EDK)]} {
		foreach path [list \
			[file join $::env(SAED32_EDK) tech] \
			[file join $::env(SAED32_EDK) lib SG db] \
			[file join $::env(SAED32_EDK) lib SG] \
			[file join $::env(SAED32_EDK) tech milkyway] \
			[file join $::env(SAED32_EDK) lib stdcell_rvt milkyway] \
			[file join $::env(SAED32_EDK) lib stdcell_hvt milkyway] \
			[file join $::env(SAED32_EDK) lib stdcell_lvt milkyway] \
			[file join $::env(SAED32_EDK) tech star_rcxt]] {
			if {[file exists $path]} {
				options set ComponentLibs/TechLibSearchPath $path -append
			}
		}
	}
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
solution options set /Flows/SCVerify/USE_CCS_BLOCK false
foreach design_file $design_files {
	solution file add $design_file
}

go new
go analyze
solution design set $top_function -top
go compile

configure_saed_techlib_search_path
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