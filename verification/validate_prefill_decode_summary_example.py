#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "tmp" / "reference_cases"
DEFAULT_PROMPT = (
    "Summarize the following text in one sentence: "
    "A small research team built a prototype LLM accelerator with a strict on-chip SRAM budget. "
    "They separated prefill and decode paths, validated each stage against PyTorch, and then introduced "
    "descriptor-driven scheduling so the same layer logic could be reused across the full network."
)


def run_step(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real summarization prompt through prefill+decode layer0 reference regression against PyTorch.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefill-case-name", type=str, default="summary_prefill_case")
    parser.add_argument("--decode-case-name", type=str, default="summary_decode_case")
    parser.add_argument("--atol", type=float, default=8e-2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefill_case_path = args.output_dir / f"{args.prefill_case_name}.npz"
    decode_case_path = args.output_dir / f"{args.decode_case_name}.npz"

    run_step(
        [
            sys.executable,
            str(ROOT / "verification" / "generate_layer0_prefill_case.py"),
            "--prompt",
            args.prompt,
            "--output-dir",
            str(args.output_dir),
            "--case-name",
            args.prefill_case_name,
        ]
    )
    run_step(
        [
            sys.executable,
            str(ROOT / "verification" / "generate_layer0_decode_case.py"),
            "--prompt",
            args.prompt,
            "--output-dir",
            str(args.output_dir),
            "--case-name",
            args.decode_case_name,
        ]
    )
    run_step(
        [
            sys.executable,
            str(ROOT / "verification" / "validate_prefill_layer0_reference_wrapper_math.py"),
            "--case-path",
            str(prefill_case_path),
            "--atol",
            str(args.atol),
        ]
    )
    run_step(
        [
            sys.executable,
            str(ROOT / "verification" / "validate_decode_layer0_reference_wrapper_math.py"),
            "--case-path",
            str(decode_case_path),
            "--atol",
            str(args.atol),
        ]
    )

    print("Prefill + decode summary example PASS")
    print(f"Prefill case: {prefill_case_path}")
    print(f"Decode case: {decode_case_path}")


if __name__ == "__main__":
    main()