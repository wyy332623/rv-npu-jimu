"""Verilator-backed RTL timing simulation for Jimu firmware traces.

The existing functional emulator remains the architectural/numerical oracle.
This module converts its dynamic command trace into dependency tokens and
replays those commands through ``rtl/jimu_npu_timing_core.sv``.  The RTL owns
all cycle decisions and counters; Python only encodes commands and enriches
the resulting schedule with source/tensor provenance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RTL_SOURCE = REPO_ROOT / "rtl" / "jimu_npu_timing_core.sv"
HARNESS_SOURCE = REPO_ROOT / "sim" / "jimu_rtl_harness.cpp"
UNIT_NAMES = ("load", "store", "mvu", "vector", "control")
UNIT_INDEX = {name: index for index, name in enumerate(UNIT_NAMES)}
STALL_NAMES = ("barrier", "dependency", "unit", "dram", "bank")
VIRTUAL_RESOURCES = {"pipe", "vpipe_a"}


class RtlSimulatorUnavailable(RuntimeError):
    """Raised when Verilator (or the WSL bridge) is unavailable."""


class RtlSimulationError(RuntimeError):
    """Raised when RTL compilation or execution fails."""


@dataclass
class RtlTimingProfile:
    name: str = "jimu-rtl-v1"
    rob_depth: int = 16
    resource_bits: int = 128
    scratchpad_banks: int = 8
    chain_fence: bool = True
    default_latency: int = 1
    instruction_latencies: dict[str, int] = field(default_factory=dict)
    instruction_initiation_intervals: dict[str, int] = field(default_factory=dict)
    unit_initiation_intervals: dict[str, int] = field(default_factory=dict)
    memory_bytes_per_cycle: float = 8.0
    memory_setup_cycles: int = 2
    memory_read_setup_cycles: int = 2
    memory_write_setup_cycles: int = 2
    memory_minimum_transfer_bytes: int = 0
    memory_element_bytes: int = 2
    dram_alias_granule_elements: int = 4
    on_chip_bytes_per_cycle: float = 8.0
    on_chip_read_setup_cycles: int = 1
    on_chip_write_setup_cycles: int = 1
    on_chip_element_bytes: int = 2
    mvu_lanes: int = 4
    mvu_pipeline_cycles: int = 3
    mvu_hdl_controller_timing: bool = False
    vector_lanes: int = 4
    vector_pipeline_cycles: int = 1
    wsl_distro: str = "Ubuntu-22.04"

    def __post_init__(self):
        if self.rob_depth < 2 or self.rob_depth > 32:
            raise ValueError("rtl.rob_depth must be between 2 and 32")
        if self.resource_bits != 128:
            raise ValueError("the current RTL harness requires resource_bits=128")
        if self.scratchpad_banks < 1 or self.scratchpad_banks > 32:
            raise ValueError("rtl.scratchpad_banks must be between 1 and 32")
        if self.memory_bytes_per_cycle <= 0:
            raise ValueError("memory.bytes_per_cycle must be positive")
        if self.on_chip_bytes_per_cycle <= 0:
            raise ValueError("on_chip_memory.bytes_per_cycle must be positive")
        if self.memory_minimum_transfer_bytes < 0:
            raise ValueError("memory.minimum_transfer_bytes cannot be negative")
        for name in UNIT_NAMES:
            self.unit_initiation_intervals.setdefault(name, 1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RtlTimingProfile":
        rtl = data.get("rtl", data)
        memory = rtl.get("memory", data.get("memory", {}))
        on_chip = rtl.get(
            "on_chip_memory", data.get("on_chip_memory", {})
        )
        mvu = rtl.get("mvu", {})
        vector = rtl.get("vector", {})
        unit_data = rtl.get("units", {})
        unit_iis = {
            name: max(1, int(
                value.get("initiation_interval", 1)
                if isinstance(value, dict) else value
            ))
            for name, value in unit_data.items()
        }
        return cls(
            name=str(rtl.get("name", data.get("name", "jimu-rtl-v1"))),
            rob_depth=int(rtl.get("rob_depth", 16)),
            resource_bits=int(rtl.get("resource_bits", 128)),
            scratchpad_banks=int(rtl.get("scratchpad_banks", 8)),
            chain_fence=bool(rtl.get("chain_fence", True)),
            default_latency=max(1, int(rtl.get("default_latency", 1))),
            instruction_latencies={
                str(name): max(1, int(value))
                for name, value in rtl.get(
                    "instruction_latencies",
                    data.get("instruction_latencies", {}),
                ).items()
            },
            instruction_initiation_intervals={
                str(name): max(1, int(value))
                for name, value in rtl.get(
                    "instruction_initiation_intervals",
                    data.get("instruction_initiation_intervals", {}),
                ).items()
            },
            unit_initiation_intervals=unit_iis,
            memory_bytes_per_cycle=float(memory.get("bytes_per_cycle", 8)),
            memory_setup_cycles=max(0, int(memory.get("setup_cycles", 2))),
            memory_read_setup_cycles=max(0, int(memory.get(
                "read_setup_cycles", memory.get("setup_cycles", 2)
            ))),
            memory_write_setup_cycles=max(0, int(memory.get(
                "write_setup_cycles", memory.get("setup_cycles", 2)
            ))),
            memory_minimum_transfer_bytes=max(
                0, int(memory.get("minimum_transfer_bytes", 0))
            ),
            memory_element_bytes=max(1, int(memory.get("element_bytes", 2))),
            dram_alias_granule_elements=max(
                1, int(memory.get("alias_granule_elements", 4))
            ),
            on_chip_bytes_per_cycle=float(on_chip.get("bytes_per_cycle", 8)),
            on_chip_read_setup_cycles=max(
                0, int(on_chip.get("read_setup_cycles", 1))
            ),
            on_chip_write_setup_cycles=max(
                0, int(on_chip.get("write_setup_cycles", 1))
            ),
            on_chip_element_bytes=max(
                1, int(on_chip.get("element_bytes", 2))
            ),
            mvu_lanes=max(1, int(mvu.get("lanes", 4))),
            mvu_pipeline_cycles=max(0, int(mvu.get("pipeline_cycles", 3))),
            mvu_hdl_controller_timing=(
                mvu.get("controller_timing") == "hdl-derived"
            ),
            vector_lanes=max(1, int(vector.get("lanes", 4))),
            vector_pipeline_cycles=max(
                0, int(vector.get("pipeline_cycles", 1))
            ),
            wsl_distro=str(rtl.get("wsl_distro", "Ubuntu-22.04")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RtlTimingProfile":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("RTL profile root must be a mapping")
        return cls.from_dict(data)


@dataclass
class EncodedCommand:
    sim_id: int
    event: dict[str, Any]
    unit_name: str
    latency: int
    initiation_interval: int
    read_identities: tuple[tuple, ...]
    write_identities: tuple[tuple, ...]
    read_mask: int = 0
    write_mask: int = 0
    bank_read_mask: int = 0
    bank_write_mask: int = 0
    uses_dram: bool = False
    barrier: bool = False

    def harness_line(self) -> str:
        word_mask = (1 << 64) - 1
        return " ".join((
            str(self.sim_id), str(UNIT_INDEX[self.unit_name]), str(self.latency),
            str(self.initiation_interval),
            f"{self.read_mask & word_mask:016x}",
            f"{(self.read_mask >> 64) & word_mask:016x}",
            f"{self.write_mask & word_mask:016x}",
            f"{(self.write_mask >> 64) & word_mask:016x}",
            f"{self.bank_read_mask:08x}",
            f"{self.bank_write_mask:08x}", str(int(self.uses_dram)),
            str(int(self.barrier)),
        ))


def _tuple_resource(value: Any) -> tuple:
    if isinstance(value, (list, tuple)):
        return tuple(_tuple_resource(item) if isinstance(item, (list, tuple))
                     else item for item in value)
    return (value,)


def _physical_identity(resource: tuple) -> tuple:
    return ("physical", resource)


def _semantic_accesses(
    events: list[dict[str, Any]], profile: RtlTimingProfile,
) -> tuple[list[tuple[tuple[tuple, ...], tuple[tuple, ...]]], dict[tuple, tuple]]:
    current_virtual: dict[tuple, tuple] = {}
    virtual_version: dict[tuple, int] = defaultdict(int)
    accesses = []
    identity_resource: dict[tuple, tuple] = {}
    granule = profile.dram_alias_granule_elements

    for event in events:
        reads: list[tuple] = []
        writes: list[tuple] = []
        for raw in event.get("uses", []):
            resource = _tuple_resource(raw)
            if resource and resource[0] == "DRAM":
                continue
            if resource and resource[0] in VIRTUAL_RESOURCES:
                identity = current_virtual.get(resource)
                if identity is None:
                    identity = ("virtual-input", resource)
                    current_virtual[resource] = identity
                    identity_resource[identity] = resource
            else:
                identity = _physical_identity(resource)
                identity_resource[identity] = resource
            reads.append(identity)

        memory = event.get("memory")
        if memory:
            address = int(memory.get("address", 0))
            elements = int(memory.get(
                "total_elements",
                int(memory.get("elements", 0)) * int(memory.get("count", 1)),
            ))
            end = max(address + 1, int(memory.get("end_address", address + elements)))
            ranges = [
                _physical_identity(("DRAM", block))
                for block in range(address // granule, (end - 1) // granule + 1)
            ]
            for identity in ranges:
                identity_resource[identity] = identity[1]
            if memory.get("direction") == "read":
                reads.extend(ranges)
            else:
                writes.extend(ranges)

        # Uses precede defs so read/modify/write operations consume the prior
        # virtual pipeline token and define a new SSA token.
        for raw in event.get("defs", []):
            resource = _tuple_resource(raw)
            if resource and resource[0] == "DRAM":
                continue
            if resource and resource[0] in VIRTUAL_RESOURCES:
                version = virtual_version[resource]
                virtual_version[resource] += 1
                identity = ("virtual", resource, version)
                current_virtual[resource] = identity
                identity_resource[identity] = resource
            else:
                identity = _physical_identity(resource)
                identity_resource[identity] = resource
            writes.append(identity)
        accesses.append((tuple(dict.fromkeys(reads)), tuple(dict.fromkeys(writes))))
    return accesses, identity_resource


def _allocate_resource_bits(
    accesses: list[tuple[tuple[tuple, ...], tuple[tuple, ...]]], bits: int,
    reuse_distance: int,
) -> tuple[dict[tuple, int], dict[str, Any]]:
    intervals: dict[tuple, list[int]] = {}
    for index, (reads, writes) in enumerate(accesses):
        for identity in (*reads, *writes):
            interval = intervals.setdefault(identity, [index, index])
            interval[1] = index

    active: list[tuple[int, int]] = []
    free_bits: list[int] = []
    allocation: dict[tuple, int] = {}
    next_bit = 0
    collisions = 0
    peak_live = 0
    for identity, (start, end) in sorted(
        intervals.items(), key=lambda item: (item[1][0], repr(item[0]))
    ):
        # A semantic lifetime ending at instruction N may still have a live
        # producer in the finite ROB while N+1 is decoded.  Reuse a scoreboard
        # bit only after the two identities cannot coexist in one ROB window.
        while active and active[0][0] + reuse_distance < start:
            _, released = heapq.heappop(active)
            heapq.heappush(free_bits, released)
        if free_bits:
            bit = heapq.heappop(free_bits)
        elif next_bit < bits:
            bit = next_bit
            next_bit += 1
        else:
            digest = hashlib.sha256(repr(identity).encode("utf-8")).digest()
            bit = int.from_bytes(digest[:8], "little") % bits
            collisions += 1
        allocation[identity] = bit
        heapq.heappush(active, (end, bit))
        peak_live = max(peak_live, len(active))
    return allocation, {
        "unique_semantic_resources": len(intervals),
        "resource_bits": bits,
        "peak_program_order_live_resources": peak_live,
        "scoreboard_bit_reuse_distance": reuse_distance,
        "conservative_hash_collisions": collisions,
    }


def _identity_bank(identity: tuple, identity_resource: dict[tuple, tuple],
                   banks: int, native_dim: int) -> int | None:
    if not identity or identity[0] != "physical":
        return None
    resource = identity_resource.get(identity, ())
    if not resource:
        return None
    kind = resource[0]
    if kind == "MRF":
        return (int(resource[1]) if len(resource) > 1 else 0) % banks
    if kind == "VRF":
        bank = int(resource[1]) if len(resource) > 1 else 0
        address = int(resource[2]) if len(resource) > 2 else 0
        return (bank * 17 + address // max(1, native_dim)) % banks
    if kind == "SRF":
        address = int(resource[1]) if len(resource) > 1 else 0
        return (1 + address) % banks
    return None


def _unit_name(event: dict[str, Any]) -> str:
    memory = event.get("memory")
    if memory:
        return "load" if memory.get("direction") == "read" else "store"
    target = str(event.get("target_unit", "control"))
    if target == "mvu":
        return "mvu"
    if target == "control":
        return "control"
    return "vector"


def _instruction_timing_value(mapping: dict[str, int],
                              event: dict[str, Any]) -> int | None:
    """Look up an opcode contract, preferring an ``OP@opd0`` variant."""
    op = str(event.get("op", ""))
    if event.get("opd0") is not None:
        variant = f"{op}@{int(event['opd0'])}"
        if variant in mapping:
            return mapping[variant]
    return mapping.get(op)


def _transfer_cycles(*, elements: int, element_bytes: int,
                     bytes_per_cycle: float, setup_cycles: int,
                     minimum_transfer_bytes: int = 0) -> int:
    payload = max(elements * element_bytes, minimum_transfer_bytes)
    return max(1, setup_cycles + math.ceil(payload / bytes_per_cycle))


def _hdl_mvu_contract(native_dim: int, lanes: int) -> tuple[int, int]:
    """MVU command envelope inferred from the supplied HDL controller model."""
    active_lanes = max(1, min(lanes, native_dim))
    groups = math.ceil(native_dim / active_lanes)
    dot_latency = math.ceil(math.log2(active_lanes)) + 1
    total_fires = native_dim * groups
    if groups == 1:
        last_fire = total_fires
    else:
        last_fire = 1 + (total_fires - 1) * (dot_latency + 2)
    latency = last_fire + dot_latency + 1
    return latency, latency + 2


def _duration_details(event: dict[str, Any], metadata: dict[str, Any],
                      profile: RtlTimingProfile) -> tuple[int, str, str | None]:
    op = str(event.get("op", ""))
    configured = _instruction_timing_value(
        profile.instruction_latencies, event
    )
    if configured is not None:
        return configured, "instruction_contract", None
    memory = event.get("memory")
    if memory:
        elements = int(memory.get(
            "total_elements",
            int(memory.get("elements", 0)) * int(memory.get("count", 1)),
        ))
        direction = str(memory.get("direction", "read"))
        setup = (
            profile.memory_read_setup_cycles
            if direction == "read" else profile.memory_write_setup_cycles
        )
        return _transfer_cycles(
            elements=elements,
            element_bytes=profile.memory_element_bytes,
            bytes_per_cycle=profile.memory_bytes_per_cycle,
            setup_cycles=setup,
            minimum_transfer_bytes=profile.memory_minimum_transfer_bytes,
        ), "external_dram_transfer", "dram"
    native_dim = max(1, int(metadata.get("native_dim", 1)))
    if op == "M_RD" and int(event.get("opd0", -1)) == 18:
        latency = 1 + native_dim * native_dim
        return latency, "hdl_vec_to_mat_drain", "on_chip"
    if op in ("V_RD", "V_WR"):
        direction = "read" if op == "V_RD" else "write"
        setup = (
            profile.on_chip_read_setup_cycles
            if direction == "read" else profile.on_chip_write_setup_cycles
        )
        return _transfer_cycles(
            elements=native_dim,
            element_bytes=profile.on_chip_element_bytes,
            bytes_per_cycle=profile.on_chip_bytes_per_cycle,
            setup_cycles=setup,
        ), "on_chip_vector_transfer", "on_chip"
    if op.startswith("MV_MUL"):
        if profile.mvu_hdl_controller_timing:
            latency, _ = _hdl_mvu_contract(native_dim, profile.mvu_lanes)
            return latency, "hdl_mvu_controller", None
        return max(1, profile.mvu_pipeline_cycles + math.ceil(
            native_dim / profile.mvu_lanes
        )), "mvu_formula", None
    if _unit_name(event) == "vector":
        return max(1, profile.vector_pipeline_cycles + math.ceil(
            native_dim / profile.vector_lanes
        )), "vector_formula", None
    return profile.default_latency, "default", None


def _duration(event: dict[str, Any], metadata: dict[str, Any],
              profile: RtlTimingProfile) -> int:
    return _duration_details(event, metadata, profile)[0]


def _initiation_interval(event: dict[str, Any], unit: str,
                         metadata: dict[str, Any],
                         profile: RtlTimingProfile) -> int:
    configured = _instruction_timing_value(
        profile.instruction_initiation_intervals, event
    )
    if configured is not None:
        return configured
    native_dim = max(1, int(metadata.get("native_dim", 1)))
    if str(event.get("op", "")) == "M_RD" and int(
        event.get("opd0", -1)
    ) == 18:
        return 1 + native_dim * native_dim
    if (
        str(event.get("op", "")).startswith("MV_MUL")
        and profile.mvu_hdl_controller_timing
    ):
        return _hdl_mvu_contract(native_dim, profile.mvu_lanes)[1]
    return profile.unit_initiation_intervals[unit]


def encode_trace(events: Iterable[dict[str, Any]], metadata: dict[str, Any],
                 profile: RtlTimingProfile) -> tuple[list[EncodedCommand], dict]:
    event_list = [dict(event) for event in events]
    accesses, identity_resource = _semantic_accesses(event_list, profile)
    bit_allocation, allocation_metadata = _allocate_resource_bits(
        accesses, profile.resource_bits, profile.rob_depth
    )
    native_dim = max(1, int(metadata.get("native_dim", 1)))
    commands = []
    for sim_id, (event, (reads, writes)) in enumerate(zip(event_list, accesses)):
        unit = _unit_name(event)
        latency, latency_source, memory_tier = _duration_details(
            event, metadata, profile
        )
        read_mask = sum(1 << bit_allocation[item] for item in reads)
        write_mask = sum(1 << bit_allocation[item] for item in writes)
        bank_reads = 0
        bank_writes = 0
        for identity in reads:
            bank = _identity_bank(
                identity, identity_resource, profile.scratchpad_banks, native_dim
            )
            if bank is not None:
                bank_reads |= 1 << bank
        for identity in writes:
            bank = _identity_bank(
                identity, identity_resource, profile.scratchpad_banks, native_dim
            )
            if bank is not None:
                bank_writes |= 1 << bank
        commands.append(EncodedCommand(
            sim_id=sim_id,
            event=event,
            unit_name=unit,
            latency=latency,
            initiation_interval=_initiation_interval(
                event, unit, metadata, profile
            ),
            read_identities=reads,
            write_identities=writes,
            read_mask=read_mask,
            write_mask=write_mask,
            bank_read_mask=bank_reads,
            bank_write_mask=bank_writes,
            uses_dram=bool(event.get("memory")),
            barrier=profile.chain_fence and event.get("op") == "INST_ISSUE",
        ))
        commands[-1].event["timing_model"] = {
            "latency_source": latency_source,
            "memory_tier": memory_tier,
            "latency_cycles": latency,
            "initiation_interval": commands[-1].initiation_interval,
        }
    allocation_metadata["ssa_pipeline_tokens"] = sum(
        1 for identity in bit_allocation if identity[0] == "virtual"
    )
    allocation_metadata["pipeline_model"] = "SSA elastic tokens"
    return commands, allocation_metadata


class _ToolRunner:
    def __init__(self, profile: RtlTimingProfile):
        self.wsl = False
        self.distro = profile.wsl_distro
        if shutil.which("verilator"):
            return
        if os.name == "nt" and shutil.which("wsl.exe"):
            probe = subprocess.run(
                ["wsl.exe", "-d", self.distro, "--", "which", "verilator"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace",
            )
            if probe.returncode == 0:
                self.wsl = True
                return
        raise RtlSimulatorUnavailable(
            "Verilator is required; install it locally or in the configured WSL distro"
        )

    def path(self, value: Path) -> str:
        resolved = value.resolve()
        if not self.wsl:
            return str(resolved)
        drive, tail = os.path.splitdrive(str(resolved))
        if not drive:
            return str(resolved).replace("\\", "/")
        tail = tail.lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive[0].lower()}/{tail}"

    def run(self, arguments: list[str | Path]) -> subprocess.CompletedProcess:
        converted = [self.path(item) if isinstance(item, Path) else str(item)
                     for item in arguments]
        if self.wsl:
            converted = ["wsl.exe", "-d", self.distro, "--", *converted]
        result = subprocess.run(
            converted, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode:
            output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            raise RtlSimulationError(output.strip())
        return result


def _build_simulator(profile: RtlTimingProfile, build_root: Path,
                     runner: _ToolRunner) -> Path:
    digest = hashlib.sha256()
    digest.update(RTL_SOURCE.read_bytes())
    digest.update(HARNESS_SOURCE.read_bytes())
    digest.update(json.dumps(asdict(profile), sort_keys=True).encode("utf-8"))
    build_dir = build_root / digest.hexdigest()[:16]
    executable = build_dir / "Vjimu_npu_timing_core"
    if executable.exists():
        return executable
    build_dir.mkdir(parents=True, exist_ok=True)
    runner.run([
        "verilator", "--cc", "--exe", "--build", "--trace", "-Wno-fatal",
        "-Wno-WIDTHEXPAND", "--top-module", "jimu_npu_timing_core",
        "-Mdir", build_dir,
        f"-GROB_DEPTH={profile.rob_depth}",
        f"-GRESOURCE_BITS={profile.resource_bits}",
        f"-GBANKS={profile.scratchpad_banks}",
        "-CFLAGS", f"-std=c++17 -DJIMU_ROB_DEPTH={profile.rob_depth}",
        RTL_SOURCE, HARNESS_SOURCE,
    ])
    if not executable.exists():
        raise RtlSimulationError(f"Verilator did not create {executable}")
    return executable


def _parse_harness_output(path: Path) -> tuple[dict[int, dict[str, int]], dict]:
    rows: dict[int, dict[str, int]] = {}
    metrics: dict[str, int] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    header = None
    for line in lines:
        if not line or line == "#jimu-rtl-schedule-v1":
            continue
        if line.startswith("#metrics"):
            for field in next(csv.reader([line]))[1:]:
                name, value = field.split("=", 1)
                metrics[name] = int(value)
            continue
        if header is None:
            header = next(csv.reader([line]))
            continue
        values = next(csv.reader([line]))
        record = {name: int(value) for name, value in zip(header, values)}
        rows[record["id"]] = record
    return rows, metrics


def _identity_label(identity: tuple) -> str:
    if identity[0] == "physical":
        return ":".join(str(value) for value in identity[1])
    if identity[0] == "virtual":
        return f"{identity[1][0]}#{identity[2]}"
    return ":".join(str(value) for value in identity[1])


def _dependency_details(commands: list[EncodedCommand]) -> tuple[dict, dict, dict]:
    predecessors: dict[int, set[int]] = defaultdict(set)
    reasons: dict[int, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    resources: dict[int, dict[int, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )

    def add(index: int, prior: int, reason: str,
            identity: tuple | None = None):
        predecessors[index].add(prior)
        reasons[index][prior].add(reason)
        if identity is not None:
            resources[index][prior][reason].add(_identity_label(identity))

    last_writer: dict[tuple, int] = {}
    last_readers: dict[tuple, set[int]] = defaultdict(set)
    last_barrier: int | None = None
    since_barrier: list[int] = []
    for command in commands:
        index = command.sim_id
        if last_barrier is not None and not command.barrier:
            add(index, last_barrier, "CHAIN_FENCE")
        for identity in command.read_identities:
            if identity in last_writer:
                prior = last_writer[identity]
                add(index, prior, "RAW", identity)
            last_readers[identity].add(index)
        for identity in command.write_identities:
            if identity in last_writer:
                prior = last_writer[identity]
                add(index, prior, "WAW", identity)
            for prior in last_readers.get(identity, set()):
                if prior != index:
                    add(index, prior, "WAR", identity)
            last_writer[identity] = index
            last_readers[identity].clear()
        if command.barrier:
            for prior in since_barrier:
                add(index, prior, "CHAIN_FENCE")
            since_barrier = []
            last_barrier = index
        else:
            since_barrier.append(index)
    return predecessors, reasons, resources


def _interval_union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = None
    for start, finish in sorted(intervals):
        if end is None or start > end:
            total += max(0, finish - start)
            end = finish
        elif finish > end:
            total += finish - end
            end = finish
    return total


def _intersection_length(left: list[tuple[int, int]],
                         right: list[tuple[int, int]]) -> int:
    boundaries = sorted({value for interval in (*left, *right) for value in interval})
    return sum(
        finish - start
        for start, finish in zip(boundaries, boundaries[1:])
        if any(a < finish and b > start for a, b in left)
        and any(a < finish and b > start for a, b in right)
    )


def _max_concurrency(records: list[dict[str, Any]]) -> int:
    points = []
    for record in records:
        points.append((record["start_cycle"], 1))
        points.append((record["end_cycle"], -1))
    active = maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _resource_predecessors(records: list[dict[str, Any]]) -> None:
    last_unit: dict[str, int] = {}
    last_dram: int | None = None
    by_sim = {record["sim_id"]: record for record in records}
    for record in sorted(records, key=lambda item: (item["start_cycle"], item["sim_id"])):
        values = []
        unit = record["target_unit"]
        if unit in last_unit:
            values.append(last_unit[unit])
        last_unit[unit] = record["sim_id"]
        if record.get("memory"):
            if last_dram is not None:
                values.append(last_dram)
            last_dram = record["sim_id"]
        record["resource_predecessors"] = sorted(set(values))
        record["predecessors"] = sorted(set(
            record["dependency_predecessors"] + record["resource_predecessors"]
        ))
    # Mark one causal critical chain for graph consumers.
    if not records:
        return
    cursor = max(records, key=lambda item: (item["end_cycle"], item["sim_id"]))
    while cursor is not None and not cursor.get("critical"):
        cursor["critical"] = True
        candidates = [
            by_sim[index] for index in cursor["predecessors"]
            if index in by_sim and by_sim[index]["end_cycle"] <= cursor["start_cycle"]
        ]
        cursor = max(candidates, key=lambda item: item["end_cycle"]) \
            if candidates else None


def _build_schedule(commands: list[EncodedCommand], rows: dict[int, dict[str, int]],
                    counters: dict[str, int], profile: RtlTimingProfile,
                    allocation_metadata: dict) -> dict[str, Any]:
    dependencies, dependency_reasons, dependency_resources = _dependency_details(
        commands
    )
    records = []
    for command in commands:
        timing = rows[command.sim_id]
        stalls = {
            name: timing[f"{name}_stall"] for name in STALL_NAMES
        }
        event = command.event
        resources = [command.unit_name]
        if command.uses_dram:
            resources.append("dram_bus")
        resources.extend(
            f"sram_r_bank{index}"
            for index in range(profile.scratchpad_banks)
            if command.bank_read_mask & (1 << index)
        )
        resources.extend(
            f"sram_w_bank{index}"
            for index in range(profile.scratchpad_banks)
            if command.bank_write_mask & (1 << index)
        )
        records.append({
            "sim_id": command.sim_id,
            "idx": event.get("idx", command.sim_id),
            "command_id": event.get("command_id", event.get("idx", command.sim_id)),
            "raw_instruction_idx": event.get("raw_instruction_idx"),
            "expanded_idx": event.get("expanded_idx"),
            "chain_id": event.get("chain_id", 0),
            "op": event.get("op", ""),
            "target_unit": command.unit_name,
            "start_cycle": timing["start"],
            "end_cycle": timing["finish"],
            "finish_cycle": timing["finish"],
            "duration_cycles": timing["finish"] - timing["start"],
            "queue_entry_cycle": timing["enqueue"],
            "queue_wait_cycles": timing["start"] - timing["enqueue"],
            "rtl_stall_cycles_by_reason": stalls,
            "blocking_reasons": [name for name, value in stalls.items() if value],
            "resources": resources,
            "resource_slots": {command.unit_name: 0},
            "dependency_predecessors": sorted(dependencies[command.sim_id]),
            "dependency_reasons": {
                str(prior): sorted(values)
                for prior, values in dependency_reasons[command.sim_id].items()
            },
            "dependency_resources": {
                str(prior): {
                    reason: sorted(values)
                    for reason, values in reason_map.items()
                }
                for prior, reason_map in dependency_resources[
                    command.sim_id
                ].items()
            },
            "memory": event.get("memory"),
            "timing_model": event.get("timing_model", {}),
            "source": event.get("source", {}),
            "tensor_reads": event.get("tensor_reads", []),
            "tensor_writes": event.get("tensor_writes", []),
            "critical": False,
        })
    _resource_predecessors(records)

    makespan = max((record["end_cycle"] for record in records), default=0)
    serial_cycles = sum(record["duration_cycles"] for record in records)
    active_union_cycles = _interval_union_length([
        (record["start_cycle"], record["end_cycle"]) for record in records
    ])
    gross_overlap_cycles = serial_cycles - active_union_cycles
    scheduler_idle_hole_cycles = makespan - active_union_cycles
    net_parallelism_savings_cycles = serial_cycles - makespan
    memory_intervals = [
        (record["start_cycle"], record["end_cycle"])
        for record in records if record.get("memory")
    ]
    compute_intervals = [
        (record["start_cycle"], record["end_cycle"])
        for record in records if record["target_unit"] in ("mvu", "vector")
    ]
    unit_intervals = {
        name: [
            (record["start_cycle"], record["end_cycle"])
            for record in records if record["target_unit"] == name
        ]
        for name in UNIT_NAMES
    }
    overlap = _intersection_length(memory_intervals, compute_intervals)
    dram_busy_cycles = _interval_union_length(memory_intervals)
    logical_dram_payload_bytes = 0
    modeled_dram_transaction_bytes = 0
    for record in records:
        memory = record.get("memory")
        if not memory:
            continue
        elements = int(memory.get(
            "total_elements",
            int(memory.get("elements", 0)) * int(memory.get("count", 1)),
        ))
        payload_bytes = elements * profile.memory_element_bytes
        logical_dram_payload_bytes += payload_bytes
        modeled_dram_transaction_bytes += max(
            payload_bytes, profile.memory_minimum_transfer_bytes
        )
    busy_cycles = {
        name: _interval_union_length(intervals)
        for name, intervals in unit_intervals.items()
    }
    utilization = {
        name: (cycles / makespan if makespan else 0.0)
        for name, cycles in busy_cycles.items()
    }
    busy_cycles["dram_bus"] = dram_busy_cycles
    utilization["dram_bus"] = (
        dram_busy_cycles / makespan if makespan else 0.0
    )
    def blocker_summary(selected_records):
        return sorted(
            (
            {
                "idx": record["idx"], "op": record["op"],
                "wait_cycles": sum(record["rtl_stall_cycles_by_reason"].values()),
                "reasons": record["blocking_reasons"],
                "blocking_reasons": record["blocking_reasons"],
                "source": record["source"],
            }
            for record in selected_records
            if sum(record["rtl_stall_cycles_by_reason"].values())
            ),
            key=lambda item: (-item["wait_cycles"], item["idx"]),
        )[:20]

    top_blockers = blocker_summary(records)
    critical_records = [record for record in records if record["critical"]]
    critical_path_top_blockers = blocker_summary(critical_records)
    critical_waits: dict[str, int] = defaultdict(int)
    for record in critical_records:
        for reason, cycles in record["rtl_stall_cycles_by_reason"].items():
            critical_waits[reason] += cycles
    metrics = {
        "rtl_predicted_npu_cycles": makespan,
        "rtl_completion_makespan_cycles": makespan,
        "rtl_idle_cycles": counters.get("rtl_counter_cycles", makespan),
        "rtl_retirement_tail_cycles": max(
            0, counters.get("rtl_counter_cycles", makespan) - makespan
        ),
        "parallel_predicted_npu_cycles": makespan,
        "serial_command_cycles": serial_cycles,
        # Backward-compatible alias. This is a net comparison against serial
        # command duration, not a measurement of intersecting intervals.
        "overlap_saved_cycles": net_parallelism_savings_cycles,
        "net_parallelism_savings_cycles": net_parallelism_savings_cycles,
        "gross_overlap_cycles": gross_overlap_cycles,
        "scheduler_idle_hole_cycles": scheduler_idle_hole_cycles,
        "memory_compute_overlap_cycles": overlap,
        "memory_compute_overlap_ratio": (
            overlap / dram_busy_cycles
            if memory_intervals else 0.0
        ),
        "logical_dram_payload_bytes": logical_dram_payload_bytes,
        "modeled_dram_transaction_bytes": modeled_dram_transaction_bytes,
        "max_concurrent_ops": _max_concurrency(records),
        **{f"{name}_utilization": value for name, value in utilization.items()},
        **counters,
    }
    return {
        "schema_version": 1,
        "model": profile.name,
        "backend": "verilator-rtl",
        "architecture": "decoupled-load-store-execute-vector-control",
        "profile": asdict(profile),
        "resource_encoding": allocation_metadata,
        "serial_cycles": serial_cycles,
        "metrics": metrics,
        "resource_busy_cycles": busy_cycles,
        "optimization_diagnostics": {
            "top_blockers": top_blockers,
            "critical_path_top_blockers": critical_path_top_blockers,
            "critical_path_top_events": sorted(
                critical_records,
                key=lambda item: (-item["duration_cycles"], item["sim_id"]),
            )[:20],
            "critical_event_wait_cycles_by_reason": dict(critical_waits),
            "resource_bottlenecks": sorted(
                ({"resource": name, "utilization": value}
                 for name, value in utilization.items()),
                key=lambda item: -item["utilization"],
            ),
            "critical_path_event_count": sum(
                bool(record["critical"]) for record in records
            ),
            "critical_path_method": (
                "post-hoc causal predecessor chain ending at the latest "
                "completion; not a formal zero-slack path"
            ),
            "stall_counter_semantics": (
                "per-cycle pressure on the oldest blocked command; counters "
                "may coexist with a younger dispatch and are not additive "
                "makespan losses"
            ),
            "source_mapping_note": (
                "source.file/source.line are authoritative when present; "
                "dynamic idx values are not source line numbers"
            ),
        },
        "events": records,
    }


def simulate_trace(
    events: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
    profile: RtlTimingProfile | dict[str, Any] | str | Path,
    artifact_path: str | Path | None = None,
    *,
    wave_path: str | Path | None = None,
    build_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compile/reuse the RTL model, replay a trace, and return its schedule."""
    if isinstance(profile, (str, Path)):
        resolved_profile = RtlTimingProfile.load(profile)
    elif isinstance(profile, dict):
        resolved_profile = RtlTimingProfile.from_dict(profile)
    else:
        resolved_profile = profile
    commands, allocation_metadata = encode_trace(
        list(events), metadata, resolved_profile
    )
    runner = _ToolRunner(resolved_profile)
    build_dir = Path(build_root or (REPO_ROOT / "_out" / "rtl-build"))
    executable = _build_simulator(resolved_profile, build_dir, runner)

    artifact = Path(artifact_path) if artifact_path else None
    work_dir = artifact.parent if artifact else Path(tempfile.mkdtemp(
        prefix="jimu-rtl-"
    ))
    work_dir.mkdir(parents=True, exist_ok=True)
    command_path = work_dir / "rtl-commands.txt"
    raw_output_path = work_dir / "rtl-harness-schedule.csv"
    command_path.write_text(
        "# id unit latency ii reads_lo reads_hi writes_lo writes_hi "
        "bank_reads bank_writes dram barrier\n"
        + "\n".join(command.harness_line() for command in commands)
        + "\n",
        encoding="utf-8",
    )
    arguments: list[str | Path] = [executable, command_path, raw_output_path]
    if wave_path:
        wave = Path(wave_path)
        wave.parent.mkdir(parents=True, exist_ok=True)
        arguments.append(wave)
    runner.run(arguments)
    rows, counters = _parse_harness_output(raw_output_path)
    if len(rows) != len(commands):
        raise RtlSimulationError(
            f"RTL returned {len(rows)} events for {len(commands)} commands"
        )
    result = _build_schedule(
        commands, rows, counters, resolved_profile, allocation_metadata
    )
    if artifact:
        artifact.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    return result
