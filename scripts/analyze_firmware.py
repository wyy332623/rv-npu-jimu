#!/usr/bin/env python3
"""Run arbitrary supported firmware and emit unified optimisation evidence.

The workload manifest supplies tensor semantics and externally observable
regions.  Without it the tool still emits a low-level graph, but correctly
reports that functional equivalence has not been established.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emulator.firmware_runner import compare_observables, run_firmware
from emulator.npu_cross_layer_graph import build_cross_layer_graph
from emulator.npu_rtl_sim import simulate_trace
from emulator.workload import WorkloadManifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate cross-layer dataflow/timing evidence for firmware"
    )
    parser.add_argument("--manifest", required=True,
                        help="Workload YAML defining ELF, tensors, and observables")
    parser.add_argument("--elf", help="Override manifest firmware ELF")
    parser.add_argument("--profile", help="Override timed-device YAML profile")
    parser.add_argument(
        "--rtl-profile",
        help="Replay the executed trace through the Verilator RTL timing core",
    )
    parser.add_argument(
        "--rtl-wave", action="store_true",
        help="Write rtl-wave.vcd (can be large for full workloads)",
    )
    parser.add_argument("--native-dim", type=int,
                        help="Override metadata.native_dim")
    parser.add_argument("-o", "--output", default="_out/firmware-analysis")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--skip-functional-reference", action="store_true")
    args = parser.parse_args(argv)

    manifest = WorkloadManifest.load(args.manifest)
    elf = args.elf or manifest.firmware
    profile = args.profile or manifest.hardware_profile
    native_dim = args.native_dim or int(manifest.metadata.get("native_dim", 0))
    if not elf:
        parser.error("firmware ELF is required through --elf or manifest.firmware")
    if not profile:
        parser.error("timing profile is required through --profile or manifest.hardware_profile")
    if native_dim < 1:
        parser.error("native dimension is required through --native-dim or metadata.native_dim")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    functional = None
    if not args.skip_functional_reference:
        functional = run_firmware(
            elf, native_dim=native_dim, manifest=manifest,
            hidden_size=manifest.metadata.get("hidden_size"),
            seq_len=manifest.metadata.get("seq_len"),
        )
    timed = run_firmware(
        elf, native_dim=native_dim, manifest=manifest,
        timing_profile=profile,
        hidden_size=manifest.metadata.get("hidden_size"),
        seq_len=manifest.metadata.get("seq_len"),
    )

    rtl_schedule = None
    if args.rtl_profile:
        rtl_schedule = simulate_trace(
            timed.events,
            manifest.metadata,
            args.rtl_profile,
            out_dir / "rtl-timing-schedule.json",
            wave_path=(out_dir / "rtl-wave.vcd") if args.rtl_wave else None,
        )
        # Keep the generic schedule contract used by DFG mining tools while
        # retaining the explicit RTL filename for provenance.
        (out_dir / "timing-schedule.json").write_text(
            json.dumps(rtl_schedule, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if functional is None:
        equivalence = {
            "verified": False, "passed": None,
            "reason": "functional reference skipped",
        }
    elif not manifest.observables:
        equivalence = {
            "verified": False, "passed": None,
            "reason": "manifest declares no observable tensor regions",
        }
    else:
        equivalence = {
            "verified": True,
            **compare_observables(functional, timed, manifest),
        }

    selected_schedule = (
        rtl_schedule["events"] if rtl_schedule is not None else timed.timeline
    )
    selected_profile = (
        rtl_schedule["model"] if rtl_schedule is not None
        else timed.metrics.get("timed_profile")
    )
    graph = build_cross_layer_graph(
        timed.events, manifest=manifest, schedule=selected_schedule,
        profile_name=selected_profile,
    )
    summary = {
        "manifest": manifest.to_dict(),
        "functional": functional.summary() if functional else None,
        "timed": timed.summary(),
        "rtl": ({
            "backend": rtl_schedule["backend"],
            "model": rtl_schedule["model"],
            "metrics": rtl_schedule["metrics"],
            "resource_encoding": rtl_schedule["resource_encoding"],
            "functional_contract": (
                "RTL models timing/control; architectural values and "
                "equivalence are provided by the functional emulator"
            ),
        } if rtl_schedule is not None else None),
        "equivalence": equivalence,
        "graph": graph.metadata,
        "opportunity_count": len(graph.opportunities),
    }
    (out_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "trace-events.json").write_text(
        json.dumps(timed.events, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "timing-timeline.json").write_text(
        json.dumps(timed.timeline, indent=2, sort_keys=True), encoding="utf-8"
    )
    graph.write_json(out_dir / "cross-layer-graph.json")
    (out_dir / "cross-layer-graph.txt").write_text(
        graph.to_text(), encoding="utf-8"
    )
    dot_path = out_dir / "cross-layer-graph.dot"
    dot_path.write_text(graph.to_dot(), encoding="utf-8")
    if not args.no_render and shutil.which("dot"):
        subprocess.run(
            [shutil.which("dot"), "-Tsvg", str(dot_path),
             "-o", str(out_dir / "cross-layer-graph.svg")],
            check=False,
        )

    print(json.dumps({
        "output": str(out_dir.resolve()),
        "functional_equivalence": equivalence,
        "timed_wall_cycles": timed.metrics.get("timed_wall_cycles"),
        "rtl_predicted_npu_cycles": (
            rtl_schedule["metrics"].get("rtl_predicted_npu_cycles")
            if rtl_schedule is not None else None
        ),
        "commands": timed.metrics.get("timed_command_count"),
        "opportunities": len(graph.opportunities),
    }, sort_keys=True))
    return 0 if equivalence.get("passed") is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
