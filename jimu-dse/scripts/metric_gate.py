#!/usr/bin/env python3
"""Evaluate the G1 two-sequence DRAM and instruction-count acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_g1_metrics(
    before_seq2: dict[str, object],
    before_seq6: dict[str, object],
    after_seq2: dict[str, object],
    after_seq6: dict[str, object],
    instruction_regression_limit: float,
    instruction_gate_enabled: bool = True,
) -> dict[str, object]:
    b2 = int(before_seq2.get("total_bytes", 0))
    b6 = int(before_seq6.get("total_bytes", 0))
    a2 = int(after_seq2.get("total_bytes", 0))
    a6 = int(after_seq6.get("total_bytes", 0))
    i6_before = int(before_seq6.get("instr_count", 0))
    i6_after = int(after_seq6.get("instr_count", 0))
    max_i6 = (
        i6_before * (1.0 + instruction_regression_limit)
        if instruction_gate_enabled and i6_before > 0
        else None
    )

    failures: list[str] = []
    configs = {
        "before_seq2": before_seq2.get("firmware_config"),
        "before_seq6": before_seq6.get("firmware_config"),
        "after_seq2": after_seq2.get("firmware_config"),
        "after_seq6": after_seq6.get("firmware_config"),
    }
    if any(config is not None for config in configs.values()):
        if not all(isinstance(config, dict) for config in configs.values()):
            failures.append("incomplete firmware_config metadata")
        else:
            if configs["before_seq2"] != configs["after_seq2"]:
                failures.append("seq2 firmware_config changed during comparison")
            if configs["before_seq6"] != configs["after_seq6"]:
                failures.append("seq6 firmware_config changed during comparison")

    if b2 <= 0 or b6 <= 0:
        failures.append("invalid before metric: total_bytes must be positive")
    if a2 <= 0 or a6 <= 0:
        failures.append("invalid candidate metric: total_bytes must be positive")
    if a2 > b2:
        failures.append(f"seq2 total_bytes regressed: {a2} > {b2}")
    if a6 >= b6:
        failures.append(f"seq6 total_bytes did not strictly improve: {a6} >= {b6}")
    if instruction_gate_enabled:
        if i6_before <= 0 or i6_after <= 0:
            failures.append("invalid seq6 instruction count")
        elif max_i6 is not None and i6_after > max_i6:
            failures.append(
                "seq6 instruction count regressed beyond limit: "
                f"{i6_after} > {max_i6:.2f}"
            )

    return {
        "metric_pass": not failures,
        "failure_reasons": failures,
        "instruction_gate_enabled": instruction_gate_enabled,
        "instruction_regression_limit": (
            instruction_regression_limit if instruction_gate_enabled else None
        ),
        "firmware_config": {
            "seq2": configs["after_seq2"],
            "seq6": configs["after_seq6"],
        },
        "seq2": {
            "before_total_bytes": b2,
            "after_total_bytes": a2,
            "delta_bytes": a2 - b2,
        },
        "seq6": {
            "before_total_bytes": b6,
            "after_total_bytes": a6,
            "delta_bytes": a6 - b6,
            "before_instr_count": i6_before,
            "after_instr_count": i6_after,
            "max_instr_count": max_i6,
        },
        # Backward-compatible top-level fields used by the closed loop.
        "total_bytes": a6,
        "instr_count": i6_after,
        "dram_stats": after_seq6.get("dram_stats", {}),
    }


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-seq2", required=True, type=Path)
    parser.add_argument("--before-seq6", required=True, type=Path)
    parser.add_argument("--after-seq2", required=True, type=Path)
    parser.add_argument("--after-seq6", required=True, type=Path)
    parser.add_argument("--instruction-regression-limit", required=True, type=float)
    parser.add_argument(
        "--instruction-gate",
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate_g1_metrics(
        load_json(args.before_seq2),
        load_json(args.before_seq6),
        load_json(args.after_seq2),
        load_json(args.after_seq6),
        args.instruction_regression_limit,
        instruction_gate_enabled=args.instruction_gate == "on",
    )
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if result["metric_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
