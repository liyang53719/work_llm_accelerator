#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASE_DIR = ROOT / "tmp" / "reference_cases"
DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following text in one sentence: "
    "A small research team built a prototype LLM accelerator with a strict on-chip SRAM budget. "
    "They separated prefill and decode paths, validated each stage against PyTorch, and then introduced "
    "descriptor-driven scheduling so the same layer logic could be reused across the full network."
)


def run_step(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the host-side regression suite for decode/prefill validation paths.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--summary-prompt", type=str, default=DEFAULT_SUMMARY_PROMPT)
    args = parser.parse_args()

    if not args.skip_build:
        run_step(["bash", str(ROOT / "hls" / "build_host_wrappers.sh")], ROOT)

    steps = [
        [sys.executable, str(ROOT / "verification" / "validate_layer_dispatch_layout.py")],
        [sys.executable, str(ROOT / "verification" / "validate_all_layer_descriptor_dispatch.py"), "--decode-steps", "1"],
        [sys.executable, str(ROOT / "verification" / "validate_prefill_attention_smoke.py")],
        [sys.executable, str(ROOT / "verification" / "validate_prefill_mlp_smoke.py")],
        [sys.executable, str(ROOT / "verification" / "validate_prefill_top_wrapper_regression.py")],
        [sys.executable, str(ROOT / "verification" / "validate_decode_attention_smoke.py")],
        [sys.executable, str(ROOT / "verification" / "validate_decode_attention_history_regression.py")],
        [sys.executable, str(ROOT / "verification" / "validate_decode_mlp_smoke.py")],
        [sys.executable, str(ROOT / "verification" / "validate_decode_top_wrapper_regression.py")],
        [
            sys.executable,
            str(ROOT / "verification" / "generate_layer0_prefill_case.py"),
            "--prompt",
            args.summary_prompt,
            "--output-dir",
            str(REFERENCE_CASE_DIR),
            "--case-name",
            "summary_prefill_case",
        ],
        [
            sys.executable,
            str(ROOT / "verification" / "generate_layer0_decode_case.py"),
            "--prompt",
            args.summary_prompt,
            "--output-dir",
            str(REFERENCE_CASE_DIR),
            "--case-name",
            "summary_decode_case",
        ],
        [
            sys.executable,
            str(ROOT / "verification" / "validate_prefill_layer0_reference_wrapper_math.py"),
            "--case-path",
            str(REFERENCE_CASE_DIR / "summary_prefill_case.npz"),
        ],
        [
            sys.executable,
            str(ROOT / "verification" / "validate_decode_layer0_reference_wrapper_math.py"),
            "--case-path",
            str(REFERENCE_CASE_DIR / "summary_decode_case.npz"),
        ],
    ]

    for command in steps:
        run_step(command, ROOT)

    print("Host regression suite PASS")


if __name__ == "__main__":
    main()