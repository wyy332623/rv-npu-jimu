#!/usr/bin/env python3
"""Measure one workload configuration through the shared runtime adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dag_workload_runtime import WorkloadConfig, get_workload_runtime
from emulator.npu_micro_op_dag import build_micro_op_dag, collapse_to_micro_ops


def _traffic(events: list[dict], dim: int) -> dict[str, int]:
    counts = {
        "vec_rd_ops": 0,
        "vec_wr_ops": 0,
        "mat_rd_ops": 0,
        "mat_wr_ops": 0,
    }
    mapping = {
        "V_RD_DRAM": "vec_rd_ops",
        "V_WR_DRAM": "vec_wr_ops",
        "M_RD_DRAM": "mat_rd_ops",
        "M_WR_DRAM": "mat_wr_ops",
    }
    for event in events:
        key = mapping.get(str(event.get("op", "")))
        if key:
            counts[key] += 1
    counts.update(
        {
            "vec_rd_elements": counts["vec_rd_ops"] * dim,
            "vec_wr_elements": counts["vec_wr_ops"] * dim,
            "mat_rd_elements": counts["mat_rd_ops"] * dim * dim,
            "mat_wr_elements": counts["mat_wr_ops"] * dim * dim,
        }
    )
    return counts


def _validate_build_metadata(
    path: Path,
    elf_path: Path,
    config: WorkloadConfig,
    workload: str,
) -> str:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected = config.as_dict()
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"firmware config mismatch: expected {expected}, got {actual}"
        )
    recorded_workload = metadata.get("workload")
    accepted_workloads = {workload}
    if workload == "adder_140p":
        accepted_workloads.add("adder")
    if recorded_workload and recorded_workload not in accepted_workloads:
        raise RuntimeError(
            f"firmware workload mismatch: expected {workload}, got {recorded_workload}"
        )
    digest = hashlib.sha256(elf_path.read_bytes()).hexdigest()
    if digest != metadata.get("elf_sha256"):
        raise RuntimeError("firmware ELF changed after build metadata was recorded")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="bert")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-head", type=int, required=True)
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--build-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = WorkloadConfig(
        args.dim, args.hidden, args.seq_len, args.num_head
    )
    runtime = get_workload_runtime(args.workload)
    elf_digest = _validate_build_metadata(
        args.build_metadata,
        args.elf,
        config,
        runtime.name,
    )
    run = runtime.run(config, args.elf)
    try:
        events = run.tracer.events
        traffic = _traffic(events, config.dim)
        total_elements = sum(
            traffic[key]
            for key in (
                "vec_rd_elements",
                "vec_wr_elements",
                "mat_rd_elements",
                "mat_wr_elements",
            )
        )
        micro_ops = collapse_to_micro_ops(events)
        nodes, edges = build_micro_op_dag(micro_ops)
        mv_mul_count = sum(
            1 for event in events if str(event.get("op", "")) == "MV_MUL"
        )
        phase_lines = [
            (
                f"  [{index:2d}] {phase['kind']}: "
                f"events={phase['event_end'] - phase['event_start']}"
            )
            for index, phase in enumerate(run.phase_ranges)
        ]
        tile_count = max(config.hidden_size // config.dim, 1)
        result = {
            "workload": runtime.name,
            "firmware_config": config.as_dict(),
            "elf_sha256": elf_digest,
            "total_bytes": total_elements * 4,
            "dram_stats": traffic,
            "instr_count": len(run.recorder.inst_trace),
            "expanded_event_count": len(events),
            "mv_mul_count": mv_mul_count,
            "mat_rd_ops": traffic["mat_rd_ops"],
            "clusters": phase_lines,
            "num_micro_ops": len(nodes),
            "num_edges": len(edges),
            "phase_ranges": run.phase_ranges,
            "runtime_metadata": run.metadata,
            "tile_structure": {
                "num_tiles": tile_count,
                "mv_mul_per_projection": (
                    tile_count * tile_count if runtime.name == "bert" else None
                ),
                "num_projections": 6 if runtime.name == "bert" else None,
                "total_projection_mv_mul": (
                    tile_count * tile_count * 6 * config.seq_len
                    if runtime.name == "bert"
                    else None
                ),
                "heads_per_tile": (
                    config.dim // (config.hidden_size // config.num_head)
                    if config.num_head > 0
                    else 1
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    finally:
        run.tracer.unpatch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
