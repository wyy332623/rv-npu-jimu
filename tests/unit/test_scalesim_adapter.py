from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scalesim_adapter",
    ROOT / "jimu-dse" / "timing" / "scalesim_adapter.py",
)
adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(adapter)


def profile():
    value = yaml.safe_load(
        (ROOT / "jimu-dse" / "timing" / "scalesim-dim4.yaml").read_text()
    )
    value["scalesim_config"] = str(
        ROOT / "jimu-dse" / "timing" / "scalesim-dim4.cfg"
    )
    return value


def parallel_profile():
    value = yaml.safe_load(
        (ROOT / "jimu-dse" / "timing" / "scalesim-parallel-dim4.yaml").read_text()
    )
    value["scalesim_config"] = str(
        ROOT / "jimu-dse" / "timing" / "scalesim-dim4.cfg"
    )
    return value


def test_topology_uses_native_scalesim_trailing_comma(tmp_path):
    topology = tmp_path / "trace.csv"
    count = adapter._write_gemm_topology(
        topology,
        [{"op": "MV_MUL"}, {"op": "V_GELU"}, {"op": "MV_MUL"}],
        dim=4,
    )
    assert count == 2
    assert topology.read_text().splitlines() == [
        "Layer,M,N,K,",
        "mv_mul_0,1,4,4,",
        "mv_mul_1,1,4,4,",
    ]


def test_memory_cycles_use_actual_trace_operations():
    cycles, count = adapter._trace_memory_cycles(
        [
            {"op": "M_RD_DRAM"},
            {"op": "V_WR_DRAM"},
            {"op": "VRF"},
        ],
        dim=4,
        profile=profile(),
    )
    assert count == 2
    assert cycles == 9  # (2 + ceil(32/8)) + (2 + ceil(8/8))


def test_hybrid_cycle_breakdown(monkeypatch):
    class FakeLayer:
        def get_compute_report_items(self):
            return [10, 2, 0, 0, 0]

    class FakeScaleSim:
        def __init__(self, **kwargs):
            layers = len(Path(kwargs["topology"]).read_text().splitlines()) - 1
            self.runner = type("Runner", (), {})()
            self.runner.single_layer_sim_object_list = [
                FakeLayer() for _ in range(layers)
            ]

        def run_scale(self, top_path):
            return None

    monkeypatch.setattr(adapter, "_load_scalesim", lambda: FakeScaleSim)
    result = adapter.simulate_trace(
        [
            {"op": "M_RD_DRAM"},
            {"op": "MV_MUL"},
            {"op": "MV_MUL"},
            {"op": "V_WR_DRAM"},
            {"op": "V_GELU"},
        ],
        {"dim": 4, "hidden": 4},
        profile(),
    )
    assert result["scalesim_layer_count"] == 2
    assert result["scalesim_compute_cycles"] == 16
    assert result["scalesim_stall_cycles"] == 4
    assert result["trace_memory_cycles"] == 9
    assert result["auxiliary_cycles"] == 4
    assert result["predicted_npu_cycles"] == 29


def test_missing_backend_has_install_instruction(monkeypatch):
    original_import = __import__

    def fail_scalesim(name, *args, **kwargs):
        if name.startswith("scalesim"):
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_scalesim)
    with pytest.raises(adapter.ScaleSimUnavailable, match="requirements-timing"):
        adapter._load_scalesim()


def test_parallel_adapter_maps_each_scalesim_layer_to_dynamic_event(
    monkeypatch, tmp_path
):
    reports = [[10, 2, 0, 0, 0], [7, 1, 0, 0, 0]]

    class FakeLayer:
        def __init__(self, report):
            self.report = report

        def get_compute_report_items(self):
            return self.report

    class FakeScaleSim:
        def __init__(self, **_kwargs):
            self.runner = type("Runner", (), {})()
            self.runner.single_layer_sim_object_list = [
                FakeLayer(report) for report in reports
            ]

        def run_scale(self, top_path):
            return None

    monkeypatch.setattr(adapter, "_load_scalesim", lambda: FakeScaleSim)
    artifact = tmp_path / "timing-schedule.json"
    events = [
        {
            "idx": 10, "op": "MV_MUL", "uses": [], "defs": [],
            "chain_id": 0,
        },
        {
            "idx": 11, "op": "M_RD", "uses": [], "defs": [],
            "chain_id": 0,
        },
        {
            "idx": 12, "op": "MV_MUL", "uses": [], "defs": [],
            "chain_id": 0,
        },
    ]

    result = adapter.simulate_trace(
        events, {"dim": 4, "hidden": 4}, parallel_profile(), artifact
    )
    schedule = json.loads(artifact.read_text())
    records = {item["idx"]: item for item in schedule["events"]}

    assert records[10]["duration_cycles"] == 8
    assert records[12]["duration_cycles"] == 6
    assert result["scalesim_compute_cycles"] == 14
    assert result["parallel_predicted_npu_cycles"] < result["predicted_npu_cycles"]
