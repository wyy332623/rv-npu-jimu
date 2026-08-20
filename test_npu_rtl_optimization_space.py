from __future__ import annotations

from pathlib import Path

from emulator.npu_rtl_sim import simulate_trace


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "jimu-dse" / "timing" / "jimu-rtl-dim4.yaml"
METADATA = {"native_dim": 4}


def event(index, op, *, target="vector", uses=(), defs=(), memory=None):
    return {
        "idx": index, "command_id": index, "op": op,
        "target_unit": target, "uses": list(uses), "defs": list(defs),
        "memory": memory, "source": {"file": "optimization-space.c", "line": index + 1},
    }


def dram_read(address, elements=16):
    return {
        "direction": "read", "address": address, "elements": elements,
        "end_address": address + elements,
    }


def run(events, path):
    return simulate_trace(events, METADATA, PROFILE, path)["metrics"]


def test_ping_pong_mrf_exposes_weight_prefetch(tmp_path):
    baseline = [
        event(0, "M_RD_DRAM", target="mmm", defs=[("MRF",)],
              memory=dram_read(0)),
        event(1, "MV_MUL", target="mvu", uses=[("MRF",), ("pipe",)],
              defs=[("pipe",)]),
        event(2, "M_RD_DRAM", target="mmm", defs=[("MRF",)],
              memory=dram_read(32)),
        event(3, "MV_MUL", target="mvu", uses=[("MRF",), ("pipe",)],
              defs=[("pipe",)]),
    ]
    ping_pong = [
        event(0, "M_RD_DRAM", target="mmm", defs=[("MRF", 0)],
              memory=dram_read(0)),
        event(1, "MV_MUL", target="mvu", uses=[("MRF", 0), ("pipe",)],
              defs=[("pipe",)]),
        event(2, "M_RD_DRAM", target="mmm", defs=[("MRF", 1)],
              memory=dram_read(32)),
        event(3, "MV_MUL", target="mvu", uses=[("MRF", 1), ("pipe",)],
              defs=[("pipe",)]),
    ]

    serial = run(baseline, tmp_path / "baseline.json")
    overlapped = run(ping_pong, tmp_path / "ping-pong.json")
    assert overlapped["rtl_predicted_npu_cycles"] < serial[
        "rtl_predicted_npu_cycles"
    ]
    assert overlapped["memory_compute_overlap_cycles"] > serial[
        "memory_compute_overlap_cycles"
    ]


def test_eliminating_dram_materialization_reduces_cycles(tmp_path):
    materialized = [
        event(0, "V_GELU", uses=[("pipe",)], defs=[("pipe",)]),
        event(1, "V_WR_DRAM", uses=[("pipe",)],
              memory={"direction": "write", "address": 64, "elements": 4,
                      "end_address": 68}),
        event(2, "V_RD_DRAM", defs=[("pipe",)], memory=dram_read(64, 4)),
        event(3, "V_RELU", uses=[("pipe",)], defs=[("pipe",)]),
    ]
    fused = [
        event(0, "V_GELU", uses=[("pipe",)], defs=[("pipe",)]),
        event(1, "V_RELU", uses=[("pipe",)], defs=[("pipe",)]),
    ]

    before = run(materialized, tmp_path / "materialized.json")
    after = run(fused, tmp_path / "fused.json")
    assert after["rtl_predicted_npu_cycles"] < before["rtl_predicted_npu_cycles"]
    assert after["serial_command_cycles"] < before["serial_command_cycles"]


def test_vrf_bank_rotation_removes_port_stall(tmp_path):
    conflict = [
        event(0, "V_RD", uses=[("VRF", 5, 0)], defs=[("pipe",)]),
        event(1, "V_RD", uses=[("VRF", 5, 32)], defs=[("pipe",)]),
    ]
    rotated = [
        event(0, "V_RD", uses=[("VRF", 5, 0)], defs=[("pipe",)]),
        event(1, "V_RD", uses=[("VRF", 5, 4)], defs=[("pipe",)]),
    ]

    before = run(conflict, tmp_path / "conflict.json")
    after = run(rotated, tmp_path / "rotated.json")
    assert before["rtl_counter_bank_stall_cycles"] > 0
    assert after["rtl_counter_bank_stall_cycles"] == 0
    assert after["rtl_predicted_npu_cycles"] < before["rtl_predicted_npu_cycles"]
