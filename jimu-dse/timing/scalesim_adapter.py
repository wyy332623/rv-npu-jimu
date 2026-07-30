"""SCALE-Sim v2 adapter for NPU instruction traces.

SCALE-Sim models the systolic MVU portion of the workload. The adapter combines
its RTL-validated GEMM compute cycles with explicit trace-derived DRAM transfer
and auxiliary-instruction cycles. The result is a deterministic performance
estimate, not a replacement for cycle-accurate simulation of this NPU's RTL.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any


class ScaleSimUnavailable(RuntimeError):
    pass


def _load_scalesim():
    try:
        from scalesim.scale_sim import scalesim
    except ImportError as exc:
        raise ScaleSimUnavailable(
            "SCALE-Sim is not installed. Run: "
            "pip install -r requirements-timing.txt"
        ) from exc
    return scalesim


def _event_latency(event: dict[str, Any], profile: dict[str, Any]) -> int:
    op = str(event.get("op", ""))
    latencies = profile.get("instruction_latencies", {})
    if op in latencies:
        return int(latencies[op])
    if op.startswith("V_FUNC/") and "V_FUNC" in latencies:
        return int(latencies["V_FUNC"])
    return int(latencies.get("default", 1))


def _trace_memory_cycles(
    events: list[dict[str, Any]], dim: int, profile: dict[str, Any]
) -> tuple[int, int]:
    memory = profile["memory"]
    bytes_per_cycle = float(memory["bytes_per_cycle"])
    setup = int(memory["setup_cycles"])
    element_bytes = int(memory.get("element_bytes", 2))
    total = 0
    count = 0
    for event in events:
        op = str(event.get("op", ""))
        if op in {"V_RD_DRAM", "V_WR_DRAM"}:
            payload = dim * element_bytes
        elif op in {"M_RD_DRAM", "M_WR_DRAM"}:
            payload = dim * dim * element_bytes
        else:
            continue
        total += setup + math.ceil(payload / bytes_per_cycle)
        count += 1
    return total, count


def _write_gemm_topology(
    path: Path, events: list[dict[str, Any]], dim: int
) -> int:
    mv_events = [event for event in events if event.get("op") == "MV_MUL"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        # SCALE-Sim v2's parser intentionally discards the final CSV field,
        # so its native topology format requires a trailing comma.
        handle.write("Layer,M,N,K,\n")
        for index, _event in enumerate(mv_events):
            # One firmware MV_MUL consumes a resident DIMxDIM MRF tile and a
            # length-DIM input vector, producing one length-DIM output row.
            handle.write(f"mv_mul_{index},1,{dim},{dim},\n")
    return len(mv_events)


def simulate_trace(
    events: list[dict[str, Any]],
    hardware: dict[str, int],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Return hybrid SCALE-Sim/trace timing metrics for one firmware run."""
    scalesim_cls = _load_scalesim()
    dim = int(hardware["dim"])
    config_path = Path(profile["scalesim_config"]).resolve()
    if not config_path.is_file():
        raise ValueError(f"SCALE-Sim config does not exist: {config_path}")

    with tempfile.TemporaryDirectory(prefix="jimu-scalesim-") as temp:
        temp_path = Path(temp)
        topology = temp_path / "trace_gemm.csv"
        layer_count = _write_gemm_topology(topology, events, dim)
        if layer_count:
            simulator = scalesim_cls(
                save_disk_space=True,
                verbose=False,
                config=str(config_path),
                topology=str(topology),
                input_type_gemm=True,
            )
            simulator.run_scale(top_path=str(temp_path))
            reports = [
                layer.get_compute_report_items()
                for layer in simulator.runner.single_layer_sim_object_list
            ]
            scalesim_total = sum(int(items[0]) for items in reports)
            scalesim_stall = sum(int(items[1]) for items in reports)
        else:
            scalesim_total = 0
            scalesim_stall = 0

    scalesim_compute = max(0, scalesim_total - scalesim_stall)
    memory_cycles, memory_ops = _trace_memory_cycles(events, dim, profile)
    auxiliary_cycles = sum(
        _event_latency(event, profile)
        for event in events
        if event.get("op") not in {
            "MV_MUL", "V_RD_DRAM", "V_WR_DRAM", "M_RD_DRAM", "M_WR_DRAM"
        }
    )
    include_stalls = bool(profile.get("include_scalesim_stalls", False))
    predicted = scalesim_compute + memory_cycles + auxiliary_cycles
    if include_stalls:
        predicted += scalesim_stall

    return {
        "timing_backend": "scalesim-v2.0.2",
        "scalesim_layer_count": layer_count,
        "scalesim_compute_cycles": scalesim_compute,
        "scalesim_stall_cycles": scalesim_stall,
        "trace_memory_cycles": memory_cycles,
        "trace_memory_ops": memory_ops,
        "auxiliary_cycles": auxiliary_cycles,
        "predicted_npu_cycles": predicted,
        "include_scalesim_stalls": include_stalls,
    }
