"""Generic workload metadata and ELF source provenance for firmware analysis.

The functional emulator can execute firmware without knowing tensor names or
which DRAM ranges are externally observable.  Optimisation, however, needs
that semantic contract.  This module keeps the contract optional: arbitrary
firmware still runs without a manifest, while annotated workloads gain stable
tensor identities, correctness boundaries, and source-level trace locations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SourceLocation:
    pc: int | None = None
    file: str | None = None
    line: int | None = None
    function: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items()
                if value is not None}


@dataclass(frozen=True)
class TensorRegion:
    """One element-addressed NPU DRAM tensor or generic buffer region."""

    name: str
    address: int
    length: int
    shape: tuple[int, ...] = ()
    dtype: str = "fp16"
    location: str = "dram"
    observable: bool = False
    frozen: bool = False
    role: str = "intermediate"
    tolerance: float | None = None

    @property
    def end_address(self) -> int:
        return self.address + self.length

    def overlaps(self, address: int, length: int = 1) -> bool:
        return address < self.end_address and self.address < address + length

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorRegion":
        address = _as_int(data.get("address", 0), "tensor.address")
        shape = tuple(int(value) for value in data.get("shape", ()))
        length_value = data.get("length")
        if length_value is None:
            length_value = _product(shape) if shape else 1
        length = _as_int(length_value, "tensor.length")
        if address < 0 or length < 1:
            raise ValueError("tensor address must be non-negative and length positive")
        if shape and _product(shape) != length:
            raise ValueError(
                f"tensor {data.get('name')!r} shape contains {_product(shape)} "
                f"elements but length is {length}"
            )
        tolerance = (
            float(data["tolerance"])
            if data.get("tolerance") is not None else None
        )
        if tolerance is not None and tolerance < 0:
            raise ValueError("tensor tolerance must be non-negative")
        return cls(
            name=str(data["name"]), address=address, length=length,
            shape=shape, dtype=str(data.get("dtype", "fp16")),
            location=str(data.get("location", "dram")),
            observable=bool(data.get("observable", False)),
            frozen=bool(data.get("frozen", False)),
            role=str(data.get("role", "intermediate")),
            tolerance=tolerance,
        )


@dataclass
class WorkloadManifest:
    """Portable correctness and tensor-semantics contract for one workload."""

    name: str = "firmware-workload"
    firmware: str | None = None
    tensors: list[TensorRegion] = field(default_factory=list)
    hardware_profile: str | None = None
    initializer: str | None = None
    cycle_limit: int = 300_000
    drain_on_halt: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def classify(self, address: int, length: int = 1) -> list[TensorRegion]:
        return [region for region in self.tensors
                if region.location.lower() == "dram"
                and region.overlaps(address, length)]

    @property
    def observables(self) -> list[TensorRegion]:
        return [region for region in self.tensors if region.observable]

    def validate(self) -> None:
        if self.cycle_limit < 1:
            raise ValueError("cycle_limit must be positive")
        names: set[str] = set()
        for region in self.tensors:
            if not region.name or region.name in names:
                raise ValueError(f"duplicate or empty tensor name: {region.name!r}")
            names.add(region.name)
        ordered = sorted(
            (region for region in self.tensors
             if region.location.lower() == "dram"),
            key=lambda region: (region.address, region.end_address),
        )
        for left, right in zip(ordered, ordered[1:]):
            if left.end_address > right.address:
                raise ValueError(
                    f"overlapping tensor regions {left.name!r} and {right.name!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "firmware": self.firmware,
            "hardware_profile": self.hardware_profile,
            "initializer": self.initializer,
            "cycle_limit": self.cycle_limit,
            "drain_on_halt": self.drain_on_halt,
            "tensors": [region.to_dict() for region in self.tensors],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkloadManifest":
        version = int(data.get("schema_version", 1))
        if version != 1:
            raise ValueError(f"unsupported workload schema_version {version}")
        result = cls(
            name=str(data.get("name", "firmware-workload")),
            firmware=(str(data["firmware"])
                      if data.get("firmware") is not None else None),
            tensors=[TensorRegion.from_dict(item)
                     for item in data.get("tensors", [])],
            hardware_profile=(str(data["hardware_profile"])
                              if data.get("hardware_profile") is not None
                              else None),
            initializer=(str(data["initializer"])
                         if data.get("initializer") is not None else None),
            cycle_limit=int(data.get("cycle_limit", 300_000)),
            drain_on_halt=bool(data.get("drain_on_halt", True)),
            metadata=dict(data.get("metadata", {})),
        )
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "WorkloadManifest":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is required
            raise RuntimeError("PyYAML is required to load workload manifests") from exc
        manifest_path = Path(path)
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("workload manifest root must be a mapping")
        result = cls.from_dict(data)
        if result.firmware and not Path(result.firmware).is_absolute():
            result.firmware = str((manifest_path.parent / result.firmware).resolve())
        if result.hardware_profile and not Path(result.hardware_profile).is_absolute():
            result.hardware_profile = str(
                (manifest_path.parent / result.hardware_profile).resolve()
            )
        if result.initializer and not Path(result.initializer).is_absolute():
            result.initializer = str(
                (manifest_path.parent / result.initializer).resolve()
            )
        return result


class ElfSourceMap:
    """Best-effort PC to function/source-line map from an ELF file."""

    def __init__(self, entries: Iterable[tuple[int, int, SourceLocation]] = ()):
        self._entries = sorted(entries, key=lambda item: (item[0], item[1]))

    @classmethod
    def from_elf(cls, path: str | Path) -> "ElfSourceMap":
        try:
            from elftools.elf.elffile import ELFFile
        except ImportError:
            return cls()
        functions: list[tuple[int, int, str]] = []
        lines: dict[int, tuple[str, int]] = {}
        try:
            with Path(path).open("rb") as handle:
                elf = ELFFile(handle)
                symtab = elf.get_section_by_name(".symtab")
                if symtab is not None:
                    for symbol in symtab.iter_symbols():
                        info = symbol["st_info"]
                        if info["type"] != "STT_FUNC" or int(symbol["st_size"]) <= 0:
                            continue
                        start = int(symbol["st_value"])
                        functions.append((start, start + int(symbol["st_size"]), symbol.name))
                if elf.has_dwarf_info():
                    dwarf = elf.get_dwarf_info()
                    for unit in dwarf.iter_CUs():
                        program = dwarf.line_program_for_CU(unit)
                        if program is None:
                            continue
                        previous = None
                        for entry in program.get_entries():
                            state = entry.state
                            if state is None:
                                continue
                            if previous is not None and not previous.end_sequence:
                                file_entry = program["file_entry"][previous.file - 1]
                                file_name = file_entry.name.decode("utf-8", errors="replace")
                                lines[int(previous.address)] = (file_name, int(previous.line or 0))
                            previous = state
        except (OSError, ValueError, IndexError, KeyError):
            return cls()

        addresses = sorted(lines)
        entries: list[tuple[int, int, SourceLocation]] = []
        for index, address in enumerate(addresses):
            end = addresses[index + 1] if index + 1 < len(addresses) else address + 4
            file_name, line = lines[address]
            function = next((name for start, stop, name in functions
                             if start <= address < stop), None)
            entries.append((address, end, SourceLocation(
                pc=address, file=file_name, line=line or None, function=function,
            )))
        for start, end, function in functions:
            if not any(left <= start < right for left, right, _ in entries):
                entries.append((start, end, SourceLocation(pc=start, function=function)))
        return cls(entries)

    def lookup(self, pc: int | None) -> SourceLocation:
        if pc is None:
            return SourceLocation()
        for start, end, location in reversed(self._entries):
            if start <= pc < end:
                return SourceLocation(
                    pc=pc, file=location.file, line=location.line,
                    function=location.function,
                )
        return SourceLocation(pc=pc)


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError as exc:
            raise ValueError(f"{field_name} is not an integer: {value!r}") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not an integer: {value!r}") from exc
