from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "parallel_scheduler",
    ROOT / "jimu-dse" / "timing" / "parallel_scheduler.py",
)
scheduler = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(scheduler)


def profile(queue_depth=2):
    return {
        "name": "test-parallel",
        "scheduler": {
            "issue_width": 1,
            "queue_depth": queue_depth,
            "implicit_chain_policy": "ordered_stream",
            "cross_chain_overlap": False,
            "chain_commit_cycles": 1,
            "resources": {
                "dram_bus": 1,
                "vmm": 1,
                "mmm": 1,
                "mvu": 1,
                "spu": 1,
            },
        },
    }


def event(idx, op, *, uses=(), defs=(), memory=None, chain_id=0):
    return {
        "idx": idx,
        "op": op,
        "uses": list(uses),
        "defs": list(defs),
        "memory": memory,
        "chain_id": chain_id,
        "raw_instruction_idx": idx,
        "expanded_idx": 0,
    }


def memory(direction, address, elements):
    return {
        "direction": direction,
        "address": address,
        "elements": elements,
        "end_address": address + elements,
    }


def by_idx(schedule):
    return {item["idx"]: item for item in schedule["events"]}


def test_independent_memory_and_compute_overlap():
    events = [
        event(0, "M_RD_DRAM", defs=[("MRF",)],
              memory=memory("read", 0, 16)),
        event(1, "V_GELU", uses=[("pipe",)], defs=[("pipe",)]),
    ]
    result = scheduler.schedule_trace(events, {0: 6, 1: 8}, profile(), 14)
    records = by_idx(result)

    assert records[0]["start_cycle"] == 0
    assert records[1]["start_cycle"] == 1
    assert result["metrics"]["parallel_predicted_npu_cycles"] == 9
    assert result["metrics"]["overlap_saved_cycles"] == 5
    assert result["metrics"]["memory_compute_overlap_cycles"] == 5


def test_dependency_prevents_memory_compute_overlap():
    events = [
        event(0, "M_RD_DRAM", defs=[("MRF",)],
              memory=memory("read", 0, 16)),
        event(1, "MV_MUL", uses=[("MRF",)], defs=[("pipe",)]),
    ]
    result = scheduler.schedule_trace(events, {0: 6, 1: 8}, profile(), 14)
    records = by_idx(result)

    assert records[1]["start_cycle"] == 6
    assert result["metrics"]["parallel_predicted_npu_cycles"] == 14
    assert result["metrics"]["memory_compute_overlap_cycles"] == 0


def test_shared_dram_bus_serializes_vector_and_matrix_transfers():
    events = [
        event(0, "V_RD_DRAM", defs=[("pipe",)],
              memory=memory("read", 0, 4)),
        event(1, "M_RD_DRAM", defs=[("MRF",)],
              memory=memory("read", 100, 16)),
    ]
    result = scheduler.schedule_trace(events, {0: 6, 1: 3}, profile(), 9)
    records = by_idx(result)

    assert records[0]["start_cycle"] == 0
    assert records[1]["start_cycle"] == 6
    assert result["metrics"]["parallel_predicted_npu_cycles"] == 9


def test_raw_war_waw_and_overlapping_dram_dependencies():
    events = [
        event(0, "V_WR", defs=[("VRF", 1, 0)]),
        event(1, "V_RD", uses=[("VRF", 1, 0)], defs=[("pipe",)]),
        event(2, "V_WR", defs=[("VRF", 1, 0)]),
        event(3, "V_RD_DRAM", memory=memory("read", 100, 8)),
        event(4, "V_WR_DRAM", memory=memory("write", 107, 8)),
    ]
    dependencies = scheduler.build_dependencies(events)

    assert 0 in dependencies[1]  # RAW
    assert {0, 1}.issubset(dependencies[2])  # WAW + WAR
    assert 3 in dependencies[4]  # overlapping DRAM read/write

    details = scheduler.build_dependency_details(events)
    assert details[1][0] == {"RAW"}
    assert details[2][0] == {"WAW"}
    assert details[2][1] == {"WAR"}
    assert details[4][3] == {"DRAM_WAR"}


def test_read_only_dram_ranges_do_not_create_data_edge():
    events = [
        event(0, "V_RD_DRAM", memory=memory("read", 100, 8)),
        event(1, "M_RD_DRAM", memory=memory("read", 104, 16)),
    ]
    assert scheduler.build_dependencies(events)[1] == set()


def test_explicit_chain_marker_is_a_completion_barrier():
    events = [
        event(0, "M_RD_DRAM", memory=memory("read", 0, 16), chain_id=0),
        event(1, "INST_ISSUE", chain_id=0),
        event(2, "V_GELU", chain_id=1),
    ]
    result = scheduler.schedule_trace(events, {0: 6, 1: 1, 2: 8}, profile(), 15)
    records = by_idx(result)

    assert records[1]["start_cycle"] == 6
    assert records[1]["end_cycle"] == 7
    assert records[2]["start_cycle"] == 7
    assert result["metrics"]["schedule_chain_count"] == 2


def test_queue_depth_bounds_limited_out_of_order_issue():
    events = [
        event(0, "M_RD_DRAM", memory=memory("read", 0, 16)),
        event(1, "V_RD_DRAM", memory=memory("read", 100, 4)),
        event(2, "MV_MUL"),
    ]
    wide = scheduler.schedule_trace(events, {0: 10, 1: 10, 2: 4}, profile(2), 24)
    narrow = scheduler.schedule_trace(events, {0: 10, 1: 10, 2: 4}, profile(1), 24)

    assert by_idx(wide)[2]["start_cycle"] == 1
    assert by_idx(narrow)[2]["start_cycle"] == 11


def test_no_parallel_opportunity_matches_serial_sum_and_is_stable():
    events = [event(0, "V_GELU"), event(1, "V_FUNC/SOFTMAX")]
    first = scheduler.schedule_trace(events, {0: 2, 1: 3}, profile(), 5)
    second = scheduler.schedule_trace(events, {0: 2, 1: 3}, profile(), 5)

    assert first == second
    assert first["implicit_chain"] is True
    assert first["metrics"]["parallel_predicted_npu_cycles"] == 5


def test_independent_vmm_mmm_and_mvu_units_overlap():
    events = [event(0, "V_GELU"), event(1, "M_RD"), event(2, "MV_MUL")]
    result = scheduler.schedule_trace(events, {0: 6, 1: 6, 2: 6}, profile(), 18)

    assert result["metrics"]["parallel_predicted_npu_cycles"] == 8
    assert result["metrics"]["max_concurrent_ops"] == 3


def test_schedule_reports_actionable_critical_path_diagnostics():
    events = [
        event(0, "M_RD_DRAM", defs=[("MRF",)],
              memory=memory("read", 0, 16)),
        event(1, "MV_MUL", uses=[("MRF",)], defs=[("pipe",)]),
    ]
    result = scheduler.schedule_trace(events, {0: 6, 1: 8}, profile(), 14)
    records = by_idx(result)
    diagnostics = result["optimization_diagnostics"]

    assert records[1]["dependency_reasons"] == {"0": ["RAW"]}
    assert "dependency:RAW" in records[1]["blocking_reasons"]
    assert records[1]["queue_wait_cycles"] == 6
    assert diagnostics["critical_path_event_count"] == 2
    assert diagnostics["critical_event_wait_cycles_by_reason"][
        "dependency:RAW"
    ] == 6
    assert diagnostics["critical_path_top_blockers"][0]["idx"] == 1
    assert diagnostics["resource_bottlenecks"][0]["resource"] == "mvu"
    assert "direct C source-line mapping is not available" in diagnostics[
        "source_mapping_note"
    ]
