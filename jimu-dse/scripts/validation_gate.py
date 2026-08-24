#!/usr/bin/env python3
"""Convert an unfiltered pytest result into a closed-loop gate record."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def analyze_pytest_output(
    output: str,
    pytest_returncode: int,
    expected_passed: int,
) -> dict[str, object]:
    passed_matches = re.findall(r"(?<!\d)(\d+)\s+passed\b", output)
    skipped_matches = re.findall(r"(?<!\d)(\d+)\s+skipped\b", output)
    passed = int(passed_matches[-1]) if passed_matches else 0
    skipped = int(skipped_matches[-1]) if skipped_matches else 0

    returncode = pytest_returncode
    reason = ""
    if pytest_returncode != 0:
        reason = f"pytest exited with {pytest_returncode}"
    elif skipped:
        returncode = 125
        reason = f"pytest reported {skipped} skipped checks"
    elif expected_passed and passed != expected_passed:
        returncode = 126
        reason = f"expected {expected_passed} passed tests, observed {passed}"

    return {
        "correctness_pass": returncode == 0,
        "returncode": returncode,
        "pytest_returncode": pytest_returncode,
        "passed": passed,
        "expected_passed": expected_passed,
        "skipped": skipped,
        "require_zero_skipped": True,
        "failure_reason": reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pytest-returncode", required=True, type=int)
    parser.add_argument("--expected-passed", required=True, type=int)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--command", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.log.read_text(encoding="utf-8", errors="replace")
    result = analyze_pytest_output(
        output,
        pytest_returncode=args.pytest_returncode,
        expected_passed=args.expected_passed,
    )
    result["validation_scope"] = args.scope
    result["command"] = args.command
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
