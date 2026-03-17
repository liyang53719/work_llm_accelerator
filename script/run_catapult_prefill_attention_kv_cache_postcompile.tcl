set script_dir [file dirname [file normalize [info script]]]
set repo_root [file dirname $script_dir]

set top_function llm_accel::qwen_prefill_attention_kv_tile_catapult
set solution_dir_name llm_accel_qwen_prefill_attention_kv_tile_catapult.v1
set clock_period 2.0

if {[info exists ::env(QWEN_HLS_CLOCK_PERIOD)] && $::env(QWEN_HLS_CLOCK_PERIOD) ne ""} {
	set clock_period $::env(QWEN_HLS_CLOCK_PERIOD)
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

proc find_latest_solution_project {repo_root solution_dir_name} {
	set latest_project ""
	set latest_mtime -1

	foreach project_dir [glob -nocomplain -directory $repo_root Catapult_*] {
		set solution_dir [file join $project_dir $solution_dir_name]
		if {![file isdirectory $solution_dir]} {
			continue
		}
		set mtime [file mtime $solution_dir]
		if {$mtime > $latest_mtime} {
			set latest_mtime $mtime
			set latest_project $project_dir
		}
	}

	if {$latest_project eq ""} {
		error "No Catapult project found for solution '$solution_dir_name'"
	}

	return $latest_project
}

if {[info exists ::env(QWEN_HLS_PROJECT_DIR)] && $::env(QWEN_HLS_PROJECT_DIR) ne ""} {
	set project_dir $::env(QWEN_HLS_PROJECT_DIR)
} else {
	set project_dir [find_latest_solution_project $repo_root $solution_dir_name]
}

if {![file isdirectory $project_dir]} {
	error "Catapult project directory not found: $project_dir"
}

cd $script_dir

project load $project_dir 2026.1

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