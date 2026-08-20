"""Lock-step timing wrapper for arbitrary firmware supported by NpuDeviceMini.

The wrapper deliberately reuses the functional device for architectural
effects.  Commands are queued and scheduled with configurable decoder, FIFO,
scoreboard, unit-pipeline, and DRAM constraints; the functional command is
retired only when its scheduled completion cycle is reached.  Consequently
firmware polling observes real BUSY/DONE/FULL transitions instead of the
functional emulator's immediate completion.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

from emulator.npu_command import (
    INC_BASE_OPCODES, OP_INST_ISSUE, OP_S_WR,
    OP_V_RD_DRAM_INC, OP_V_WR_DRAM_INC, MemoryAccess, NpuCommand,
    decode_executed, decode_raw, opcode_name,
)
from emulator.npu_device_mini import (
    NPU_CHAIN_STATUS, NPU_INST_FIFO, NPU_RESET, NPU_STATUS,
    REG_TILE_ROWS, STATUS_BUSY, STATUS_DONE,
)
from emulator.workload import SourceLocation, WorkloadManifest


STATUS_FULL = 0x04


@dataclass
class UnitTiming:
    latency: int = 1
    initiation_interval: int = 1
    count: int = 1
    fifo_depth: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_latency: int = 1):
        return cls(
            latency=max(0, int(data.get("latency", default_latency))),
            initiation_interval=max(1, int(data.get("initiation_interval", 1))),
            count=max(1, int(data.get("count", 1))),
            fifo_depth=max(1, int(data.get("fifo_depth", 16))),
        )


@dataclass
class TimedDeviceProfile:
    name: str = "generic-npu-timed-v1"
    decoder_latency: int = 1
    issue_width: int = 1
    front_end_fifo_depth: int = 64
    npu_ticks_per_cpu_cycle: int = 1
    default_latency: int = 1
    instruction_latencies: dict[str, int] = field(default_factory=dict)
    units: dict[str, UnitTiming] = field(default_factory=dict)
    memory_bytes_per_cycle: float = 8.0
    memory_setup_cycles: int = 2
    memory_element_bytes: int = 2

    def __post_init__(self):
        defaults = {
            "control": UnitTiming(1, 1, 1, 64),
            "vmm": UnitTiming(3, 1, 1, 16),
            "mmm": UnitTiming(5, 5, 1, 16),
            "mvu": UnitTiming(4, 1, 1, 16),
            "spu": UnitTiming(2, 1, 1, 16),
        }
        for name, timing in defaults.items():
            self.units.setdefault(name, timing)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimedDeviceProfile":
        timing = data.get("timed_device", data)
        memory = timing.get("memory", data.get("memory", {}))
        default_latency = int(timing.get("default_latency", 1))
        result = cls(
            name=str(timing.get("name", data.get("name", "generic-npu-timed-v1"))),
            decoder_latency=max(0, int(timing.get("decoder_latency", 1))),
            issue_width=max(1, int(timing.get("issue_width", 1))),
            front_end_fifo_depth=max(1, int(timing.get("front_end_fifo_depth", 64))),
            npu_ticks_per_cpu_cycle=max(1, int(
                timing.get("npu_ticks_per_cpu_cycle", 1)
            )),
            default_latency=max(0, default_latency),
            instruction_latencies={
                str(name): max(0, int(value))
                for name, value in timing.get(
                    "instruction_latencies", data.get("instruction_latencies", {})
                ).items()
            },
            units={
                str(name): UnitTiming.from_dict(value, default_latency)
                for name, value in timing.get("units", {}).items()
            },
            memory_bytes_per_cycle=float(memory.get("bytes_per_cycle", 8)),
            memory_setup_cycles=max(0, int(memory.get("setup_cycles", 2))),
            memory_element_bytes=max(1, int(memory.get("element_bytes", 2))),
        )
        if result.memory_bytes_per_cycle <= 0:
            raise ValueError("memory.bytes_per_cycle must be positive")
        return result

    @classmethod
    def load(cls, path: str | Path) -> "TimedDeviceProfile":
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("timed device profile root must be a mapping")
        return cls.from_dict(data)


@dataclass
class ScheduledCommand:
    sequence: int
    command: NpuCommand
    enqueue_cycle: int
    start_cycle: int
    finish_cycle: int
    unit_slot: int
    decoder_ready_cycle: int
    fifo_ready_cycle: int
    dependency_ready_cycle: int
    issue_ready_cycle: int
    unit_ready_cycle: int
    memory_ready_cycle: int
    resource_ready_cycle: int
    cpu_context: dict[str, Any]
    retired_cycle: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "sequence": self.sequence,
            "command_id": self.command.command_id,
            "raw": self.command.raw,
            "op": self.command.op,
            "opcode": self.command.opcode,
            "target_unit": self.command.target_unit,
            "chain_id": self.command.chain_id,
            "enqueue_cycle": self.enqueue_cycle,
            "start_cycle": self.start_cycle,
            "finish_cycle": self.finish_cycle,
            "duration_cycles": self.finish_cycle - self.start_cycle,
            "queue_wait_cycles": self.start_cycle - self.enqueue_cycle,
            "decoder_ready_cycle": self.decoder_ready_cycle,
            "fifo_ready_cycle": self.fifo_ready_cycle,
            "dependency_ready_cycle": self.dependency_ready_cycle,
            "issue_ready_cycle": self.issue_ready_cycle,
            "unit_ready_cycle": self.unit_ready_cycle,
            "memory_ready_cycle": self.memory_ready_cycle,
            "resource_ready_cycle": self.resource_ready_cycle,
            "unit_slot": self.unit_slot,
            "retired_cycle": self.retired_cycle,
            "source": self.command.source.to_dict(),
            "tensor_reads": list(self.command.memory.tensors)
                if self.command.memory and self.command.memory.direction == "read"
                else [],
            "tensor_writes": list(self.command.memory.tensors)
                if self.command.memory and self.command.memory.direction == "write"
                else [],
        }
        if self.command.memory:
            result["memory"] = self.command.memory.to_dict()
        return result


class TimedNpuDevice:
    """MMIO-compatible timing wrapper around ``NpuDeviceMini``."""

    _UNIT_STATUS_BITS = {
        "vmm": 0x1, "mmm": 0x2, "mvu": 0x4, "spu": 0x8,
        "control": 0x10,
    }

    def __init__(self, inner_device, profile: TimedDeviceProfile | dict | None = None,
                 manifest: WorkloadManifest | None = None):
        self._inner = inner_device
        if profile is None:
            self.profile = TimedDeviceProfile()
        elif isinstance(profile, TimedDeviceProfile):
            self.profile = profile
        else:
            self.profile = TimedDeviceProfile.from_dict(profile)
        self.manifest = manifest
        self.cycle_count = 0
        self._sequence = 0
        self._chain_id = 0
        self._pending: deque[ScheduledCommand] = deque()
        self._timeline: list[ScheduledCommand] = []
        self._cpu_context: dict[str, Any] = {}
        self._source_map = None
        self._issue_available = [0] * self.profile.issue_width
        self._unit_available = {
            name: [0] * timing.count for name, timing in self.profile.units.items()
        }
        # Each entry is the cycle at which an already accepted command leaves
        # that unit's input FIFO and starts execution.  Keeping future release
        # cycles makes finite FIFO capacity affect scheduling even when a CPU
        # emits a whole chain in one cycle window.
        self._unit_fifo_releases: dict[str, list[int]] = {
            name: [] for name in self.profile.units
        }
        self._memory_available = 0
        self._last_write: dict[tuple, int] = {}
        self._last_reads: dict[tuple, list[int]] = defaultdict(list)
        self._shadow_regs: dict[int, int] = {}
        self._shadow_dram_addr = int(getattr(inner_device, "_dram_addr", 0))
        self._overflow_count = 0
        self._max_fifo_occupancy = 0
        self._poll_reads = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def set_cpu_context(self, *, pc=None, cycle=None, inst_count=None):
        self._cpu_context = {"pc": pc, "cycle": cycle, "inst_count": inst_count}
        if hasattr(self._inner, "set_cpu_context"):
            self._inner.set_cpu_context(pc=pc, cycle=cycle, inst_count=inst_count)

    def set_source_map(self, source_map):
        self._source_map = source_map
        if hasattr(self._inner, "set_source_map"):
            self._inner.set_source_map(source_map)

    def load(self, addr: int, size: int) -> bytes:
        if addr == NPU_STATUS:
            self._poll_reads += 1
            value = STATUS_BUSY if self._pending else STATUS_DONE
            if len(self._pending) >= self.profile.front_end_fifo_depth:
                value |= STATUS_FULL
            return int(value).to_bytes(size, "little")
        if addr == NPU_CHAIN_STATUS:
            self._poll_reads += 1
            value = 0
            for item in self._pending:
                value |= self._UNIT_STATUS_BITS.get(item.command.target_unit, 0)
            return int(value).to_bytes(size, "little")
        return self._inner.load(addr, size)

    def store(self, addr: int, data: bytes):
        if addr == NPU_INST_FIFO:
            self._enqueue(int.from_bytes(data, "little"))
            return
        if addr == NPU_RESET and int.from_bytes(data, "little"):
            self._reset_timing()
        self._inner.store(addr, data)

    def tick(self, count: int | None = None):
        ticks = (self.profile.npu_ticks_per_cpu_cycle
                 if count is None else max(0, int(count)))
        for _ in range(ticks):
            self.cycle_count += 1
            self._retire_ready()

    def run_until_idle(self, max_ticks: int = 1_000_000) -> int:
        ticks = 0
        while self._pending and ticks < max_ticks:
            self.tick(1)
            ticks += 1
        if self._pending:
            raise TimeoutError(
                f"timed NPU did not drain after {max_ticks} ticks; "
                f"{len(self._pending)} commands remain"
            )
        return ticks

    @property
    def timeline(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._timeline]

    def metrics(self) -> dict[str, Any]:
        unit_busy: dict[str, int] = defaultdict(int)
        stalls: dict[str, int] = defaultdict(int)
        for item in self._timeline:
            unit_busy[item.command.target_unit] += (
                item.finish_cycle - item.start_cycle
            )
            # Attribute only the cycles that extend the readiness frontier.
            # The sum is therefore the actual queue wait, without counting
            # overlapping dependency/resource constraints more than once.
            frontier = item.enqueue_cycle
            for name, ready in (
                ("decoder", item.decoder_ready_cycle),
                ("fifo_full", item.fifo_ready_cycle),
                ("scoreboard", item.dependency_ready_cycle),
                ("issue", item.issue_ready_cycle),
                ("unit", item.unit_ready_cycle),
                ("dram", item.memory_ready_cycle),
            ):
                if ready > frontier:
                    stalls[name] += ready - frontier
                    frontier = ready
        first_enqueue = min(
            (item.enqueue_cycle for item in self._timeline), default=0
        )
        last_retire = max(
            (
                item.retired_cycle
                if item.retired_cycle is not None else item.finish_cycle
                for item in self._timeline
            ),
            default=first_enqueue,
        )
        busy_intervals = sorted(
            (item.enqueue_cycle, item.retired_cycle or item.finish_cycle)
            for item in self._timeline
        )
        active_cycles = 0
        interval_end = None
        for start, end in busy_intervals:
            if interval_end is None or start > interval_end:
                active_cycles += max(0, end - start)
                interval_end = end
            elif end > interval_end:
                active_cycles += end - interval_end
                interval_end = end
        return {
            "timed_profile": self.profile.name,
            # Exclude the conventional post-completion firmware spin loop from
            # the optimisation objective, while retaining total lock-step time
            # for debugging cycle-limit behaviour.
            "timed_wall_cycles": max(0, last_retire - first_enqueue),
            "timed_active_cycles": active_cycles,
            "timed_simulator_cycles": self.cycle_count,
            "timed_first_enqueue_cycle": first_enqueue,
            "timed_last_retire_cycle": last_retire,
            "timed_command_count": len(self._timeline),
            "timed_pending_commands": len(self._pending),
            "timed_poll_reads": self._poll_reads,
            "timed_frontend_overflow_count": self._overflow_count,
            "timed_max_fifo_occupancy": self._max_fifo_occupancy,
            "timed_decoder_stall_cycles": stalls["decoder"],
            "timed_fifo_full_stall_cycles": stalls["fifo_full"],
            "timed_scoreboard_stall_cycles": stalls["scoreboard"],
            "timed_issue_stall_cycles": stalls["issue"],
            "timed_unit_stall_cycles": stalls["unit"],
            "timed_dram_stall_cycles": stalls["dram"],
            "timed_queue_wait_cycles": sum(
                max(0, item.start_cycle - item.enqueue_cycle)
                for item in self._timeline
            ),
            # Compatibility aliases for callers of the first timed prototype.
            "timed_dependency_stall_cycles": stalls["scoreboard"],
            "timed_resource_stall_cycles": sum(
                stalls[name] for name in ("fifo_full", "issue", "unit", "dram")
            ),
            "timed_unit_busy_cycles": dict(sorted(unit_busy.items())),
        }

    def _enqueue(self, raw: int):
        if len(self._pending) >= self.profile.front_end_fifo_depth:
            # Firmware in this project historically assumes a chain-sized FIFO
            # and does not poll FULL.  Preserve execution while making an
            # undersized profile visible and scoreable.
            self._overflow_count += 1
        self._max_fifo_occupancy = max(
            self._max_fifo_occupancy, len(self._pending) + 1
        )
        source = (
            self._source_map.lookup(self._cpu_context.get("pc"))
            if self._source_map is not None
            else SourceLocation(pc=self._cpu_context.get("pc"))
        )
        command = decode_raw(
            raw, command_id=self._sequence, chain_id=self._chain_id,
            native_dim=int(self._inner.native_dim),
            tile_rows=self._shadow_regs.get(REG_TILE_ROWS, 1), source=source,
            cpu_cycle=self._cpu_context.get("cycle"), manifest=self.manifest,
        )
        if command.inc_parent_opcode is not None:
            count = self._increment_count()
            raw_opcode = command.inc_parent_opcode
            base_opcode = INC_BASE_OPCODES[raw_opcode]
            increment = command.opd1
            full_operand = command.full_operand
            if raw_opcode in (OP_V_RD_DRAM_INC, OP_V_WR_DRAM_INC):
                full_operand = self._shadow_dram_addr + increment
                self._shadow_dram_addr += increment * count
            command = decode_executed(
                command_id=self._sequence, raw=raw, opcode=base_opcode,
                opd0=command.opd0, opd1=0,
                full_operand=full_operand,
                native_dim=int(self._inner.native_dim),
                tile_rows=self._shadow_regs.get(REG_TILE_ROWS, 1),
                chain_id=self._chain_id,
                raw_instruction_idx=self._sequence, source=source,
                cpu_cycle=self._cpu_context.get("cycle"), manifest=self.manifest,
            )
            command.op = opcode_name(raw_opcode, command.opd0)
            command.inc_parent_opcode = raw_opcode
            if command.memory:
                tensors = ()
                if self.manifest:
                    names = {
                        region.name
                        for index in range(count)
                        for region in self.manifest.classify(
                            full_operand + index * increment,
                            command.memory.elements,
                        )
                    }
                    tensors = tuple(sorted(names))
                command.memory = MemoryAccess(
                    command.memory.direction, full_operand,
                    command.memory.elements, tensors,
                    count=count, stride=increment,
                )
        if command.opcode == OP_S_WR:
            self._shadow_regs[command.opd0] = command.opd1
        scheduled = self._schedule(command)
        self._pending.append(scheduled)
        self._timeline.append(scheduled)
        self._sequence += 1
        if command.opcode == OP_INST_ISSUE:
            self._chain_id += 1

    def _schedule(self, command: NpuCommand) -> ScheduledCommand:
        enqueue = self.cycle_count
        decoder_ready = enqueue + self.profile.decoder_latency
        dependency_ready = enqueue
        for resource in command.uses:
            dependency_ready = max(dependency_ready, self._last_write.get(resource, 0))
        for resource in command.defs:
            dependency_ready = max(dependency_ready, self._last_write.get(resource, 0))
            dependency_ready = max(
                [dependency_ready, *self._last_reads.get(resource, [])]
            )

        issue_slot = min(range(len(self._issue_available)),
                         key=self._issue_available.__getitem__)
        issue_ready = self._issue_available[issue_slot]
        unit = self.profile.units[command.target_unit]
        fifo_releases = self._unit_fifo_releases[command.target_unit]
        fifo_releases[:] = [cycle for cycle in fifo_releases if cycle > enqueue]
        fifo_ready = enqueue
        if len(fifo_releases) >= unit.fifo_depth:
            fifo_ready = sorted(fifo_releases)[-unit.fifo_depth]
        slots = self._unit_available[command.target_unit]
        unit_slot = min(range(len(slots)), key=slots.__getitem__)
        unit_ready = slots[unit_slot]
        memory_ready = self._memory_available if command.memory else enqueue
        resource_ready = max(issue_ready, unit_ready, memory_ready, fifo_ready)
        start = max(
            decoder_ready,
            dependency_ready,
            resource_ready,
        )
        duration = self._duration(command, unit)
        finish = start + duration
        self._issue_available[issue_slot] = start + 1
        slots[unit_slot] = start + unit.initiation_interval
        fifo_releases.append(start)
        if command.memory:
            self._memory_available = finish
        for resource in command.uses:
            self._last_reads[resource].append(finish)
        for resource in command.defs:
            self._last_write[resource] = finish
            self._last_reads[resource].clear()
        return ScheduledCommand(
            sequence=self._sequence, command=command,
            enqueue_cycle=enqueue, start_cycle=start, finish_cycle=finish,
            unit_slot=unit_slot, decoder_ready_cycle=decoder_ready,
            fifo_ready_cycle=fifo_ready,
            dependency_ready_cycle=dependency_ready,
            issue_ready_cycle=issue_ready, unit_ready_cycle=unit_ready,
            memory_ready_cycle=memory_ready,
            resource_ready_cycle=resource_ready,
            cpu_context=dict(self._cpu_context),
        )

    def _duration(self, command: NpuCommand, unit: UnitTiming) -> int:
        duration = int(self.profile.instruction_latencies.get(
            command.op, unit.latency if unit.latency is not None
            else self.profile.default_latency,
        ))
        if command.memory:
            payload = (
                command.memory.total_elements * self.profile.memory_element_bytes
            )
            duration = max(duration, self.profile.memory_setup_cycles + math.ceil(
                payload / self.profile.memory_bytes_per_cycle
            ))
        if command.inc_parent_opcode is not None and command.memory is None:
            duration *= self._increment_count()
        return max(0, duration)

    def _increment_count(self) -> int:
        return max(1, self._shadow_regs.get(2, 1) * self._shadow_regs.get(3, 1))

    def _retire_ready(self):
        # Architectural state retires in firmware order.  Independent commands
        # may overlap in the timing model, but their visible side effects never
        # violate the functional emulator's chain semantics.
        while self._pending and self._pending[0].finish_cycle <= self.cycle_count:
            item = self._pending.popleft()
            context = item.cpu_context
            if hasattr(self._inner, "set_cpu_context"):
                self._inner.set_cpu_context(
                    pc=context.get("pc"), cycle=context.get("cycle"),
                    inst_count=context.get("inst_count"),
                )
            self._inner._push_instruction(item.command.raw)
            item.retired_cycle = self.cycle_count

    def _reset_timing(self):
        self._pending.clear()
        self._timeline.clear()
        self.cycle_count = 0
        self._sequence = 0
        self._chain_id = 0
        self._issue_available = [0] * self.profile.issue_width
        self._unit_available = {
            name: [0] * timing.count for name, timing in self.profile.units.items()
        }
        self._unit_fifo_releases = {name: [] for name in self.profile.units}
        self._memory_available = 0
        self._last_write.clear()
        self._last_reads.clear()
        self._shadow_regs.clear()
        self._shadow_dram_addr = int(getattr(self._inner, "_dram_addr", 0))
        self._overflow_count = 0
        self._max_fifo_occupancy = 0
        self._poll_reads = 0
