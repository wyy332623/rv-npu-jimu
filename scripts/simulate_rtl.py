#!/usr/bin/env python3
"""Replay an existing Jimu event trace through the Verilator timing RTL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emulator.npu_rtl_sim import simulate_trace
from emulator.workload import WorkloadManifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cycle-simulate a firmware command trace with Jimu RTL"
    )
    parser.add_argument("--events", required=True,
                        help="trace-events.json or an object containing events")
    parser.add_argument("--profile", required=True, help="RTL timing YAML")
    parser.add_argument("--manifest", help="Optional workload YAML for metadata")
    parser.add_argument("--native-dim", type=int,
                        help="Override/inject metadata.native_dim")
    parser.add_argument("-o", "--output", default="_out/rtl-timing-schedule.json")
    parser.add_argument("--wave", help="Optional VCD output path")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.events).read_text(encoding="utf-8"))
    events = payload.get("events", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        parser.error("events input must be a list or an object containing events")
    metadata = {}
    if args.manifest:
        metadata.update(WorkloadManifest.load(args.manifest).metadata)
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        metadata.update(payload["metadata"])
    if args.native_dim:
        metadata["native_dim"] = args.native_dim
    if int(metadata.get("native_dim", 0)) < 1:
        parser.error("native dimension is required via --native-dim or --manifest")

    result = simulate_trace(
        events, metadata, args.profile, args.output, wave_path=args.wave
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "model": result["model"],
        "events": len(result["events"]),
        "rtl_predicted_npu_cycles": result["metrics"][
            "rtl_predicted_npu_cycles"
        ],
        "memory_compute_overlap_cycles": result["metrics"][
            "memory_compute_overlap_cycles"
        ],
        "max_concurrent_ops": result["metrics"]["max_concurrent_ops"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
