import struct

import numpy as np
import pytest

from emulator.npu_command import (
    OP_INST_ISSUE, OP_S_WR, OP_V_RD_DRAM, OP_V_RD_DRAM_INC,
)
from emulator.npu_device_mini import NPU_CHAIN_STATUS, NPU_INST_FIFO, NPU_STATUS
from emulator.npu_device_timed import TimedDeviceProfile, TimedNpuDevice
from emulator.npu_cross_layer_graph import build_cross_layer_graph
from emulator.firmware_runner import FirmwareRunResult, compare_observables
from emulator.workload import TensorRegion, WorkloadManifest
from iss.mini_rv64 import MiniRV64


class FakeFunctionalDevice:
    native_dim = 4

    def __init__(self):
        self.executed = []
        self.contexts = []

    def _push_instruction(self, raw):
        self.executed.append(raw)
        self.contexts.append(getattr(self, "_cpu_context", {}).copy())

    def load(self, addr, size):
        return bytes(size)

    def store(self, addr, data):
        pass

    def set_cpu_context(self, **context):
        self._cpu_context = context

    def set_source_map(self, source_map):
        self._source_map = source_map


def _raw(opcode, operand=0):
    return (opcode << 24) | operand


def test_workload_manifest_classifies_regions_and_rejects_overlap(tmp_path):
    path = tmp_path / "workload.yaml"
    path.write_text(
        """schema_version: 1
name: generic
tensors:
  - {name: input, address: 0x20, length: 8, shape: [8], frozen: true}
  - {name: output, address: 0x40, length: 4, observable: true}
""",
        encoding="utf-8",
    )
    manifest = WorkloadManifest.load(path)
    assert [region.name for region in manifest.classify(0x22, 2)] == ["input"]
    assert [region.name for region in manifest.observables] == ["output"]

    with pytest.raises(ValueError, match="overlapping tensor regions"):
        WorkloadManifest.from_dict({
            "tensors": [
                {"name": "a", "address": 0, "length": 4},
                {"name": "b", "address": 3, "length": 4},
            ]
        })
    with pytest.raises(ValueError, match="shape contains"):
        TensorRegion.from_dict({
            "name": "bad-shape", "address": 0, "length": 3,
            "shape": [2, 2],
        })


def test_observable_comparison_rejects_nonfinite_results():
    manifest = WorkloadManifest(tensors=[
        TensorRegion("output", 0, 1, observable=True),
    ])
    common = dict(
        elf="fw.elf", cpu_cycles=1, cpu_instructions=1, halted=True,
        raw_trace=[], events=[], metrics={}, timeline=[], device=None,
    )
    reference = FirmwareRunResult(
        **common, observables={"output": np.array([0.0])}
    )
    candidate = FirmwareRunResult(
        **common, observables={"output": np.array([float("nan")])}
    )

    result = compare_observables(reference, candidate, manifest)
    assert result["passed"] is False
    assert "NaN or infinity" in result["failures"][0]


def test_timed_device_exposes_busy_and_retires_in_firmware_order():
    inner = FakeFunctionalDevice()
    profile = TimedDeviceProfile.from_dict({
        "name": "test",
        "decoder_latency": 1,
        "front_end_fifo_depth": 4,
        "units": {
            "vmm": {"latency": 3, "initiation_interval": 1},
            "control": {"latency": 1},
        },
        "memory": {"bytes_per_cycle": 8, "setup_cycles": 0,
                   "element_bytes": 2},
    })
    timed = TimedNpuDevice(inner, profile)
    timed.set_cpu_context(pc=0x100, cycle=7, inst_count=3)
    first = _raw(OP_V_RD_DRAM, 0x20)
    marker = _raw(OP_INST_ISSUE)
    timed.store(NPU_INST_FIFO, first.to_bytes(4, "little"))
    timed.store(NPU_INST_FIFO, marker.to_bytes(4, "little"))

    assert int.from_bytes(timed.load(NPU_STATUS, 4), "little") & 1
    assert int.from_bytes(timed.load(NPU_CHAIN_STATUS, 4), "little") != 0
    assert inner.executed == []

    timed.run_until_idle()
    assert inner.executed == [first, marker]
    assert inner.contexts[0]["pc"] == 0x100
    assert int.from_bytes(timed.load(NPU_STATUS, 4), "little") == 2
    assert timed.metrics()["timed_command_count"] == 2
    assert timed.timeline[0]["finish_cycle"] >= 3


def test_timed_device_enforces_per_unit_fifo_and_attributes_stalls():
    inner = FakeFunctionalDevice()
    profile = TimedDeviceProfile.from_dict({
        "name": "fifo-test",
        "decoder_latency": 1,
        "issue_width": 4,
        "units": {
            "control": {
                "latency": 1, "initiation_interval": 5,
                "count": 1, "fifo_depth": 1,
            },
        },
    })
    timed = TimedNpuDevice(inner, profile)
    for register in range(3):
        raw = _raw(OP_S_WR, (register << 16) | 1)
        timed.store(NPU_INST_FIFO, raw.to_bytes(4, "little"))

    timeline = timed.timeline
    assert [item["start_cycle"] for item in timeline] == [1, 6, 11]
    assert timeline[2]["fifo_ready_cycle"] == 6
    assert timed.metrics()["timed_fifo_full_stall_cycles"] > 0
    assert (
        timed.metrics()["timed_queue_wait_cycles"]
        >= timed.metrics()["timed_fifo_full_stall_cycles"]
    )


def test_timed_device_models_increment_memory_span_and_transfer_count():
    inner = FakeFunctionalDevice()
    timed = TimedNpuDevice(inner, TimedDeviceProfile.from_dict({
        "decoder_latency": 0,
        "memory": {"bytes_per_cycle": 8, "setup_cycles": 2,
                   "element_bytes": 2},
    }))
    for register, value in ((2, 2), (3, 2)):
        raw = _raw(OP_S_WR, (register << 16) | value)
        timed.store(NPU_INST_FIFO, raw.to_bytes(4, "little"))
    timed.store(
        NPU_INST_FIFO, _raw(OP_V_RD_DRAM_INC, 4).to_bytes(4, "little")
    )

    inc = timed.timeline[-1]
    assert inc["op"] == "V_RD_DRAM_INC"
    assert inc["target_unit"] == "vmm"
    assert inc["memory"] == {
        "direction": "read", "address": 4, "elements": 4,
        "count": 4, "stride": 4, "total_elements": 16,
        "end_address": 20,
    }
    assert inc["duration_cycles"] >= 6


def test_minirv64_srli_uses_all_64_bits_and_ticks_device():
    class TickDevice:
        def __init__(self):
            self.ticks = 0

        def tick(self):
            self.ticks += 1

    cpu = MiniRV64()
    device = TickDevice()
    cpu.set_mmio_device(device)
    cpu.pc = 0
    cpu.regs[2] = 0xF000000000000000
    shamt = 36
    instruction = (
        (shamt << 20) | (2 << 15) | (5 << 12) | (3 << 7) | 0x13
    )
    cpu.mem[0:4] = struct.pack("<I", instruction)
    cpu.run(cycles=1)

    assert cpu.regs[3] == 0x0F000000
    assert device.ticks == 1


def test_cross_layer_graph_links_tensors_source_timing_and_opportunities(tmp_path):
    manifest = WorkloadManifest(name="graph-test", tensors=[
        TensorRegion("weights", 0x10, 4, frozen=True, role="parameter"),
        TensorRegion("scratch", 0x20, 4, role="intermediate"),
        TensorRegion("output", 0x30, 4, observable=True, role="output"),
    ])
    events = [
        {"idx": 0, "command_id": 0, "op": "V_RD_DRAM", "opcode": 20,
         "raw": 0, "chain_id": 0, "target_unit": "vmm",
         "uses": [("DRAM", 0x10)], "defs": [("pipe",)],
         "tensor_reads": ["weights"], "tensor_writes": [],
         "memory": {"direction": "read", "address": 0x10,
                    "elements": 4, "end_address": 0x14},
         "source": {"file": "fw.c", "line": 10}},
        {"idx": 1, "command_id": 1, "op": "V_WR_DRAM", "opcode": 21,
         "raw": 0, "chain_id": 0, "target_unit": "vmm",
         "uses": [("pipe",)], "defs": [("DRAM", 0x20)],
         "tensor_reads": [], "tensor_writes": ["scratch"],
         "memory": {"direction": "write", "address": 0x20,
                    "elements": 4, "end_address": 0x24},
         "source": {"file": "fw.c", "line": 11}},
        {"idx": 2, "command_id": 2, "op": "V_RD_DRAM", "opcode": 20,
         "raw": 0, "chain_id": 0, "target_unit": "vmm",
         "uses": [("DRAM", 0x20)], "defs": [("pipe",)],
         "tensor_reads": ["scratch"], "tensor_writes": [],
         "memory": {"direction": "read", "address": 0x20,
                    "elements": 4, "end_address": 0x24},
         "source": {"file": "fw.c", "line": 12}},
        {"idx": 3, "command_id": 3, "op": "V_RD_DRAM", "opcode": 20,
         "raw": 0, "chain_id": 0, "target_unit": "vmm",
         "uses": [("DRAM", 0x10)], "defs": [("pipe",)],
         "tensor_reads": ["weights"], "tensor_writes": [],
         "memory": {"direction": "read", "address": 0x10,
                    "elements": 4, "end_address": 0x14},
         "source": {"file": "fw.c", "line": 13}},
    ]
    timeline = [
        {"command_id": index, "start_cycle": index * 3,
         "finish_cycle": index * 3 + 2,
         "queue_wait_cycles": 2 if index == 2 else 0,
         "target_unit": "vmm", "critical_path": index == 2}
        for index in range(4)
    ]
    graph = build_cross_layer_graph(events, manifest, timeline,
                                    profile_name="test")
    kinds = {item["kind"] for item in graph.opportunities}
    assert "repeated_frozen_load" in kinds
    assert "intermediate_materialization" in kinds
    assert "scheduled_wait" in kinds
    assert graph.metadata["has_source_provenance"] is True
    assert any(edge.kind == "tensor_read" for edge in graph.edges)
    path = tmp_path / "graph.json"
    graph.write_json(path)
    assert '"schema_version": 1' in path.read_text(encoding="utf-8")
    assert "fw.c:12" in graph.to_text()


def test_cross_layer_graph_maps_raw_inc_timing_to_expanded_events():
    events = [
        {"idx": 0, "command_id": 0, "raw_instruction_idx": 0,
         "expanded_idx": 0, "op": "V_RD_DRAM", "uses": [], "defs": []},
        {"idx": 1, "command_id": 1, "raw_instruction_idx": 0,
         "expanded_idx": 1, "op": "V_RD_DRAM", "uses": [], "defs": []},
        {"idx": 2, "command_id": 2, "raw_instruction_idx": 1,
         "expanded_idx": 0, "op": "INST_ISSUE", "uses": [], "defs": []},
    ]
    schedule = [
        {"sequence": 0, "command_id": 0, "enqueue_cycle": 1,
         "start_cycle": 3, "finish_cycle": 8, "queue_wait_cycles": 2},
        {"sequence": 1, "command_id": 1, "enqueue_cycle": 2,
         "start_cycle": 8, "finish_cycle": 9, "queue_wait_cycles": 6},
    ]

    graph = build_cross_layer_graph(events, schedule=schedule)
    commands = {node.id: node for node in graph.nodes if node.layer == "command"}
    assert commands["command:0"].attributes["timing"]["finish_cycle"] == 8
    assert commands["command:1"].attributes["timing"]["finish_cycle"] == 8
    assert commands["command:2"].attributes["timing"]["finish_cycle"] == 9
    assert len([
        item for item in graph.opportunities if item["kind"] == "scheduled_wait"
    ]) == 2
