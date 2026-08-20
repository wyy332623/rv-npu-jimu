from __future__ import annotations

from pathlib import Path

from emulator.npu_rtl_sim import RtlTimingProfile, encode_trace, simulate_trace
from emulator.npu_cross_layer_graph import build_cross_layer_graph


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "jimu-dse" / "timing" / "jimu-rtl-dim4.yaml"


def event(index, op, *, target="vector", uses=(), defs=(), memory=None):
    return {
        "idx": index,
        "command_id": index,
        "op": op,
        "target_unit": target,
        "uses": list(uses),
        "defs": list(defs),
        "memory": memory,
        "source": {"file": "synthetic.c", "line": index + 1},
    }


def memory(direction, address, elements):
    return {
        "direction": direction,
        "address": address,
        "elements": elements,
        "end_address": address + elements,
    }


def by_idx(result):
    return {item["idx"]: item for item in result["events"]}


def test_encoder_ssa_renames_pipeline_but_keeps_physical_storage():
    profile = RtlTimingProfile.load(PROFILE)
    commands, metadata = encode_trace([
        event(0, "V_RD", uses=[("VRF", 5, 0)], defs=[("pipe",)]),
        event(1, "V_GELU", uses=[("pipe",)], defs=[("pipe",)]),
        event(2, "V_WR", uses=[("pipe",)], defs=[("VRF", 5, 0)]),
    ], {"native_dim": 4}, profile)

    assert commands[0].write_mask & commands[1].read_mask
    assert commands[1].write_mask & commands[2].read_mask
    assert commands[0].read_mask & commands[2].write_mask
    assert metadata["pipeline_model"] == "SSA elastic tokens"
    assert metadata["conservative_hash_collisions"] == 0


def test_rtl_overlaps_independent_dram_and_compute(tmp_path):
    result = simulate_trace([
        event(0, "M_RD_DRAM", target="mmm", defs=[("MRF",)],
              memory=memory("read", 0, 16)),
        event(1, "V_GELU", uses=[("pipe",)], defs=[("pipe",)]),
    ], {"native_dim": 4}, PROFILE, tmp_path / "schedule.json")

    records = by_idx(result)
    assert records[1]["start_cycle"] < records[0]["end_cycle"]
    assert result["metrics"]["memory_compute_overlap_cycles"] > 0
    assert result["metrics"]["max_concurrent_ops"] == 2
    assert result["metrics"]["rtl_counter_memory_compute_overlap_cycles"] > 0
    metrics = result["metrics"]
    assert metrics["rtl_completion_makespan_cycles"] == metrics[
        "rtl_predicted_npu_cycles"
    ]
    assert metrics["rtl_idle_cycles"] == metrics["rtl_counter_cycles"]
    assert metrics["rtl_retirement_tail_cycles"] == (
        metrics["rtl_idle_cycles"] - metrics["rtl_completion_makespan_cycles"]
    )
    assert metrics["net_parallelism_savings_cycles"] == (
        metrics["serial_command_cycles"]
        - metrics["rtl_completion_makespan_cycles"]
    )
    assert metrics["gross_overlap_cycles"] - metrics[
        "scheduler_idle_hole_cycles"
    ] == metrics["net_parallelism_savings_cycles"]
    critical_ids = {
        item["idx"] for item in result["events"] if item["critical"]
    }
    assert {
        item["idx"]
        for item in result["optimization_diagnostics"][
            "critical_path_top_blockers"
        ]
    } <= critical_ids


def test_rtl_dependency_prevents_early_compute(tmp_path):
    result = simulate_trace([
        event(0, "M_RD_DRAM", target="mmm", defs=[("MRF",)],
              memory=memory("read", 0, 16)),
        event(1, "MV_MUL", target="mvu",
              uses=[("MRF",), ("pipe",)], defs=[("pipe",)]),
    ], {"native_dim": 4}, PROFILE, tmp_path / "schedule.json")

    records = by_idx(result)
    assert records[1]["start_cycle"] >= records[0]["end_cycle"]
    assert records[1]["dependency_reasons"]["0"] == ["RAW"]
    assert result["metrics"]["rtl_counter_dependency_stall_cycles"] > 0


def test_rtl_local_sram_read_port_conflict_is_visible(tmp_path):
    result = simulate_trace([
        event(0, "V_RD", uses=[("VRF", 5, 0)], defs=[("pipe",)]),
        event(1, "MV_MUL", target="mvu", uses=[("VRF", 5, 0)],
              defs=[("pipe",)]),
    ], {"native_dim": 4}, PROFILE, tmp_path / "schedule.json")

    records = by_idx(result)
    assert records[1]["start_cycle"] >= records[0]["end_cycle"]
    assert result["metrics"]["rtl_counter_bank_stall_cycles"] > 0


def test_cross_layer_graph_prefers_explicit_rtl_dependencies():
    events = [
        event(0, "V_RD", defs=[("pipe",)]),
        event(1, "M_RD_DRAM", target="mmm", defs=[("pipe",)]),
    ]
    schedule = [
        {"command_id": 0, "start_cycle": 1, "finish_cycle": 3,
         "dependency_predecessors": [], "dependency_reasons": {}},
        {"command_id": 1, "start_cycle": 2, "finish_cycle": 4,
         "dependency_predecessors": [], "dependency_reasons": {}},
    ]
    graph = build_cross_layer_graph(events, schedule=schedule)

    assert graph.metadata["dependency_model"] == "schedule_explicit"
    assert not any(edge.kind == "WAW" for edge in graph.edges)
