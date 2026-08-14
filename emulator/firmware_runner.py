"""Reusable runner for functional and lock-step timed firmware execution."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np

from emulator.npu_device_mini import MEM_DRAM, NpuDeviceMini
from emulator.npu_device_timed import TimedDeviceProfile, TimedNpuDevice
from emulator.npu_event_trace import EventTracer
from emulator.trace_recorder import TraceRecorder
from emulator.workload import WorkloadManifest
from iss.mini_rv64 import MiniRV64


Initializer = Callable[[NpuDeviceMini, WorkloadManifest | None], None]


@dataclass
class FirmwareRunResult:
    elf: str
    cpu_cycles: int
    cpu_instructions: int
    halted: bool
    raw_trace: list[int]
    events: list[dict[str, Any]]
    metrics: dict[str, Any]
    timeline: list[dict[str, Any]]
    observables: dict[str, np.ndarray]
    device: NpuDeviceMini

    def summary(self) -> dict[str, Any]:
        return {
            "elf": self.elf,
            "cpu_cycles": self.cpu_cycles,
            "cpu_instructions": self.cpu_instructions,
            "halted": self.halted,
            "raw_instruction_count": len(self.raw_trace),
            "expanded_event_count": len(self.events),
            "metrics": self.metrics,
            "observables": {
                name: {"length": int(value.size)}
                for name, value in self.observables.items()
            },
        }


def run_firmware(
    elf_path: str | Path,
    *,
    native_dim: int,
    manifest: WorkloadManifest | None = None,
    timing_profile: TimedDeviceProfile | dict[str, Any] | str | Path | None = None,
    initializer: Initializer | None = None,
    hidden_size: int | None = None,
    seq_len: int | None = None,
    cycle_limit: int | None = None,
    drain_limit: int = 1_000_000,
) -> FirmwareRunResult:
    """Execute one ELF and return generic trace, timing, and observable state."""
    elf = str(Path(elf_path).resolve())
    inner = NpuDeviceMini(native_dim=native_dim)
    metadata = manifest.metadata if manifest else {}
    inner.set_hidden_size(int(hidden_size or metadata.get("hidden_size", native_dim)))
    inner.set_seq_len(int(seq_len or metadata.get("seq_len", 1)))
    prepare = initializer or load_initializer(manifest.initializer if manifest else None)
    if prepare:
        prepare(inner, manifest)

    tracer = EventTracer(inner, manifest=manifest)
    timed = None
    if timing_profile is not None:
        profile = (
            TimedDeviceProfile.load(timing_profile)
            if isinstance(timing_profile, (str, Path))
            else timing_profile
        )
        timed = TimedNpuDevice(inner, profile=profile, manifest=manifest)
        device = timed
    else:
        device = inner
    recorder = TraceRecorder(device)
    cpu = MiniRV64()
    cpu.set_mmio_device(recorder)
    cpu.load_elf(elf)
    limit = int(cycle_limit or (manifest.cycle_limit if manifest else 300_000))
    try:
        cpu.run(cycles=limit)
        if timed and (manifest is None or manifest.drain_on_halt):
            timed.run_until_idle(max_ticks=drain_limit)
        metrics = {
            "cpu_cycles": cpu.cycle_count,
            "cpu_instructions": cpu.inst_count,
            "raw_instruction_count": len(recorder.inst_trace),
            "expanded_event_count": len(tracer.events),
            **(timed.metrics() if timed else {}),
        }
        observables = _capture_observables(inner, manifest)
        return FirmwareRunResult(
            elf=elf, cpu_cycles=cpu.cycle_count,
            cpu_instructions=cpu.inst_count, halted=cpu.halted,
            raw_trace=list(recorder.inst_trace), events=list(tracer.events),
            metrics=metrics, timeline=timed.timeline if timed else [],
            observables=observables, device=inner,
        )
    finally:
        tracer.unpatch()


def compare_observables(reference: FirmwareRunResult,
                        candidate: FirmwareRunResult,
                        manifest: WorkloadManifest) -> dict[str, Any]:
    failures = []
    regions = {region.name: region for region in manifest.observables}
    max_diff = 0.0
    for name, region in regions.items():
        if name not in reference.observables or name not in candidate.observables:
            failures.append(f"missing observable {name}")
            continue
        left = reference.observables[name]
        right = candidate.observables[name]
        if left.shape != right.shape:
            failures.append(f"observable {name} shape mismatch: {left.shape} != {right.shape}")
            continue
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            failures.append(f"observable {name} contains NaN or infinity")
            continue
        diff = float(np.max(np.abs(left - right))) if left.size else 0.0
        max_diff = max(max_diff, diff)
        tolerance = region.tolerance if region.tolerance is not None else 0.0
        if diff > tolerance:
            failures.append(
                f"observable {name} max_diff={diff:g} exceeds tolerance={tolerance:g}"
            )
    return {"passed": not failures, "max_diff": max_diff, "failures": failures}


def _capture_observables(inner: NpuDeviceMini,
                         manifest: WorkloadManifest | None) -> dict[str, np.ndarray]:
    if not manifest:
        return {}
    dram = inner._vrf[MEM_DRAM]
    return {
        region.name: dram[region.address:region.end_address].copy()
        for region in manifest.observables
        if region.location.lower() == "dram"
    }


def load_initializer(path: str | None) -> Initializer | None:
    if not path:
        return None
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(
        f"npu_workload_initializer_{module_path.stem}", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workload initializer {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initialize = getattr(module, "initialize", None)
    if not callable(initialize):
        raise AttributeError(
            f"workload initializer {module_path} must define initialize(npu, manifest)"
        )
    return initialize
