"""Canonical decoded NPU command model shared by tracing and timing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from emulator.workload import SourceLocation, WorkloadManifest


# Keep the canonical numeric ISA in one analysis-facing module.  The functional
# implementation remains authoritative; tests guard these values against it.
OP_S_WR = 0; OP_S_RD = 1; OP_V_RD = 2; OP_M_RD = 3
OP_V_WR = 5; OP_M_WR = 6; OP_MV_MUL = 7
OP_VV_ADD = 8; OP_VV_A_SUB_B = 9; OP_VV_B_SUB_A = 10; OP_VV_MUL = 11
OP_V_SIGM = 12; OP_V_TANH = 13; OP_V_RELU = 14; OP_VV_MAX = 15
OP_V_RD_INC = 16; OP_V_WR_INC = 17; OP_VV_ADD_INC = 18; OP_VV_MAX_INC = 19
OP_V_RD_DRAM = 20; OP_V_WR_DRAM = 21
OP_V_RD_DRAM_INC = 22; OP_V_WR_DRAM_INC = 23
OP_M_RD_DRAM = 24; OP_M_WR_DRAM = 25; OP_V_RD_3D = 26; OP_MV_MUL_INC = 27
OP_VV_MIN = 30; OP_VV_MUL_INC = 31
OP_VV_A_SUB_B_INC = 32; OP_VV_B_SUB_A_INC = 33
OP_S_RECIP = 35; OP_V_EXP = 37; OP_S_SQRT = 38; OP_SS_MUL = 40
OP_V_GELU = 42; OP_V_FUNC = 43; OP_SS_ADD = 44; OP_INST_ISSUE = 45
REG_TILE_ROWS = 1

MEM_FILL = 12
MEM_SPU_ADD_REDUCE = 14; MEM_SPU_MAX_REDUCE = 15
MEM_SPU_ABSMAX_REDUCE = 16; MEM_SPU_BROADCAST = 17
MEM_VEC_TO_MAT_ROW = 18
SUB_SOFTMAX = 0; SUB_LAYERNORM = 1

INC_OPCODES = {
    OP_V_RD_INC, OP_V_WR_INC, OP_V_RD_DRAM_INC, OP_V_WR_DRAM_INC,
    OP_MV_MUL_INC, OP_VV_ADD_INC, OP_VV_MAX_INC, OP_VV_MUL_INC,
    OP_VV_A_SUB_B_INC, OP_VV_B_SUB_A_INC,
}
INC_BASE_OPCODES = {
    OP_V_RD_INC: OP_V_RD, OP_V_WR_INC: OP_V_WR,
    OP_V_RD_DRAM_INC: OP_V_RD_DRAM, OP_V_WR_DRAM_INC: OP_V_WR_DRAM,
    OP_MV_MUL_INC: OP_MV_MUL,
    OP_VV_ADD_INC: OP_VV_ADD, OP_VV_MAX_INC: OP_VV_MAX,
    OP_VV_MUL_INC: OP_VV_MUL,
    OP_VV_A_SUB_B_INC: OP_VV_A_SUB_B,
    OP_VV_B_SUB_A_INC: OP_VV_B_SUB_A,
}

OPCODE_NAMES = {
    OP_S_WR: "S_WR", OP_S_RD: "S_RD", OP_V_RD: "V_RD", OP_M_RD: "M_RD",
    OP_V_WR: "V_WR", OP_M_WR: "M_WR", OP_MV_MUL: "MV_MUL",
    OP_VV_ADD: "VV_ADD", OP_VV_A_SUB_B: "VV_A_SUB_B",
    OP_VV_B_SUB_A: "VV_B_SUB_A", OP_VV_MUL: "VV_MUL",
    OP_V_SIGM: "V_SIGM", OP_V_TANH: "V_TANH", OP_V_RELU: "V_RELU",
    OP_VV_MAX: "VV_MAX", OP_V_RD_INC: "V_RD_INC", OP_V_WR_INC: "V_WR_INC",
    OP_VV_ADD_INC: "VV_ADD_INC", OP_VV_MAX_INC: "VV_MAX_INC",
    OP_V_RD_DRAM: "V_RD_DRAM", OP_V_WR_DRAM: "V_WR_DRAM",
    OP_V_RD_DRAM_INC: "V_RD_DRAM_INC", OP_V_WR_DRAM_INC: "V_WR_DRAM_INC",
    OP_M_RD_DRAM: "M_RD_DRAM", OP_M_WR_DRAM: "M_WR_DRAM",
    OP_V_RD_3D: "V_RD_3D", OP_MV_MUL_INC: "MV_MUL_INC",
    OP_VV_MIN: "VV_MIN", OP_VV_MUL_INC: "VV_MUL_INC",
    OP_VV_A_SUB_B_INC: "VV_A_SUB_B_INC",
    OP_VV_B_SUB_A_INC: "VV_B_SUB_A_INC", OP_V_EXP: "V_EXP",
    OP_S_RECIP: "S_RECIP", OP_S_SQRT: "S_SQRT", OP_SS_MUL: "SS_MUL",
    OP_V_GELU: "V_GELU", OP_V_FUNC: "V_FUNC", OP_SS_ADD: "SS_ADD",
    OP_INST_ISSUE: "INST_ISSUE",
}


@dataclass(frozen=True)
class MemoryAccess:
    direction: str
    address: int
    elements: int
    tensors: tuple[str, ...] = ()
    count: int = 1
    stride: int = 0

    @property
    def end_address(self) -> int:
        return self.address + (self.count - 1) * self.stride + self.elements

    @property
    def total_elements(self) -> int:
        return self.elements * self.count

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["end_address"] = self.end_address
        if self.tensors:
            result["tensors"] = list(self.tensors)
        else:
            result.pop("tensors", None)
        if self.count == 1:
            result.pop("count", None)
            result.pop("stride", None)
        else:
            result["total_elements"] = self.total_elements
        return result


@dataclass
class NpuCommand:
    command_id: int
    raw: int
    opcode: int
    op: str
    opd0: int
    opd1: int
    full_operand: int
    tile_rows: int = 1
    chain_id: int = 0
    raw_instruction_idx: int = -1
    expanded_idx: int = 0
    inc_parent_opcode: int | None = None
    target_unit: str = "control"
    defs: list[tuple] = field(default_factory=list)
    uses: list[tuple] = field(default_factory=list)
    memory: MemoryAccess | None = None
    source: SourceLocation = field(default_factory=SourceLocation)
    cpu_cycle: int | None = None

    def to_event(self) -> dict[str, Any]:
        return {
            "idx": self.command_id,
            "command_id": self.command_id,
            "op": self.op,
            "opcode": self.opcode,
            "raw": self.raw,
            "opd0": self.opd0,
            "opd1": self.opd1,
            "full_operand": self.full_operand,
            "tile_rows": self.tile_rows,
            "raw_instruction_idx": self.raw_instruction_idx,
            "expanded_idx": self.expanded_idx,
            "inc_parent_opcode": self.inc_parent_opcode,
            "chain_id": self.chain_id,
            "target_unit": self.target_unit,
            "defs": self.defs,
            "uses": self.uses,
            "memory": self.memory.to_dict() if self.memory else None,
            "source": self.source.to_dict(),
            "cpu_cycle": self.cpu_cycle,
            "tensor_reads": list(self.memory.tensors)
                if self.memory and self.memory.direction == "read" else [],
            "tensor_writes": list(self.memory.tensors)
                if self.memory and self.memory.direction == "write" else [],
        }


def opcode_name(opcode: int, opd0: int = 0) -> str:
    if opcode == OP_V_FUNC:
        if opd0 == SUB_SOFTMAX:
            return "V_FUNC/SOFTMAX"
        if opd0 == SUB_LAYERNORM:
            return "V_FUNC/LAYERNORM"
        return f"V_FUNC/sub{opd0}"
    return OPCODE_NAMES.get(opcode, f"OP_{opcode}")


def target_unit(opcode: int, opd0: int = 0) -> str:
    if opcode in (OP_M_RD, OP_M_WR, OP_M_RD_DRAM, OP_M_WR_DRAM):
        return "mmm"
    if opcode in (OP_MV_MUL, OP_MV_MUL_INC):
        return "mvu"
    if opcode in (OP_SS_ADD, OP_S_RECIP, OP_S_SQRT, OP_SS_MUL):
        return "spu"
    if opcode in (OP_S_WR, OP_S_RD, OP_INST_ISSUE):
        return "control"
    return "vmm"


def resolve_defs_uses(opcode: int, opd0: int, opd1: int,
                      full_operand: int) -> tuple[list[tuple], list[tuple]]:
    defs: list[tuple] = []
    uses: list[tuple] = []
    if opcode == OP_S_WR:
        defs.append(("REG", opd0))
    elif opcode == OP_S_RD:
        uses.append(("REG", opd0))
    elif opcode == OP_V_RD:
        if opd0 == MEM_SPU_BROADCAST:
            uses.append(("SRF", opd1))
        elif opd0 != MEM_FILL:
            uses.append(("VRF", opd0, opd1))
        defs.extend([("pipe",), ("vpipe_a",)])
    elif opcode == OP_M_RD:
        defs.append(("MRF",) if opd0 == MEM_VEC_TO_MAT_ROW else ("pipe",))
        if opd0 != MEM_VEC_TO_MAT_ROW:
            defs.append(("vpipe_a",))
    elif opcode == OP_V_WR:
        uses.append(("pipe",))
        if opd0 == MEM_VEC_TO_MAT_ROW:
            defs.append(("VRF", MEM_VEC_TO_MAT_ROW, 0))
        elif opd0 in (MEM_SPU_ADD_REDUCE, MEM_SPU_MAX_REDUCE,
                      MEM_SPU_ABSMAX_REDUCE):
            defs.append(("SRF", opd1))
        else:
            defs.append(("VRF", opd0, opd1))
    elif opcode == OP_M_WR:
        uses.append(("MRF",))
    elif opcode == OP_MV_MUL:
        uses.extend([("pipe",), ("MRF",)])
        defs.append(("pipe",))
    elif opcode in (OP_VV_ADD, OP_VV_A_SUB_B, OP_VV_B_SUB_A,
                    OP_VV_MIN, OP_VV_MAX, OP_VV_MUL):
        uses.extend([("pipe",), ("vpipe_a",)])
        defs.append(("pipe",))
    elif opcode in (OP_V_SIGM, OP_V_TANH, OP_V_RELU, OP_V_GELU, OP_V_EXP):
        uses.append(("pipe",)); defs.append(("pipe",))
    elif opcode == OP_V_FUNC:
        uses.append(("pipe",)); defs.append(("pipe",))
        if opd0 == SUB_LAYERNORM:
            uses.extend([("VRF", 5, 0), ("VRF", 7, 0)])
    elif opcode == OP_V_RD_DRAM:
        uses.append(("DRAM", full_operand))
        defs.extend([("pipe",), ("vpipe_a",)])
    elif opcode == OP_V_WR_DRAM:
        uses.append(("pipe",)); defs.append(("DRAM", full_operand))
    elif opcode == OP_M_RD_DRAM:
        uses.extend([("DRAM", full_operand), ("REG", REG_TILE_ROWS)])
        defs.append(("MRF",))
    elif opcode == OP_M_WR_DRAM:
        uses.extend([("MRF",), ("REG", REG_TILE_ROWS)])
        defs.append(("DRAM", full_operand))
    elif opcode == OP_SS_ADD:
        uses.append(("SRF", opd0)); defs.append(("SRF", opd1))
    elif opcode in (OP_S_RECIP, OP_S_SQRT, OP_SS_MUL):
        uses.append(("SRF", opd0)); defs.append(("SRF", opd1))
    return defs, uses


def memory_access(opcode: int, full_operand: int, native_dim: int,
                  tile_rows: int = 1,
                  manifest: WorkloadManifest | None = None) -> MemoryAccess | None:
    if opcode in (OP_V_RD_DRAM, OP_V_WR_DRAM):
        elements = native_dim
    elif opcode in (OP_M_RD_DRAM, OP_M_WR_DRAM):
        matrix_dim = max(1, int(tile_rows)) * native_dim
        elements = matrix_dim * matrix_dim
    else:
        return None
    direction = "read" if opcode in (OP_V_RD_DRAM, OP_M_RD_DRAM) else "write"
    tensors = tuple(region.name for region in manifest.classify(full_operand, elements)) \
        if manifest else ()
    return MemoryAccess(direction, int(full_operand), int(elements), tensors)


def decode_executed(*, command_id: int, raw: int, opcode: int, opd0: int,
                    opd1: int, full_operand: int, native_dim: int,
                    tile_rows: int = 1,
                    chain_id: int = 0, raw_instruction_idx: int = -1,
                    expanded_idx: int = 0,
                    source: SourceLocation | None = None,
                    cpu_cycle: int | None = None,
                    manifest: WorkloadManifest | None = None) -> NpuCommand:
    defs, uses = resolve_defs_uses(opcode, opd0, opd1, full_operand)
    raw_opcode = (raw >> 24) & 0xFF
    return NpuCommand(
        command_id=command_id, raw=raw, opcode=opcode,
        op=opcode_name(opcode, opd0), opd0=opd0, opd1=opd1,
        full_operand=full_operand, tile_rows=max(1, int(tile_rows)),
        chain_id=chain_id,
        raw_instruction_idx=raw_instruction_idx, expanded_idx=expanded_idx,
        inc_parent_opcode=raw_opcode if raw_opcode in INC_OPCODES else None,
        target_unit=target_unit(opcode, opd0), defs=defs, uses=uses,
        memory=memory_access(
            opcode, full_operand, native_dim, tile_rows, manifest
        ),
        source=source or SourceLocation(), cpu_cycle=cpu_cycle,
    )


def decode_raw(raw: int, *, command_id: int = 0, chain_id: int = 0,
               native_dim: int = 1, tile_rows: int = 1,
               source: SourceLocation | None = None,
               cpu_cycle: int | None = None,
               manifest: WorkloadManifest | None = None) -> NpuCommand:
    raw_opcode = (raw >> 24) & 0xFF
    opcode = INC_BASE_OPCODES.get(raw_opcode, raw_opcode)
    opd0 = (raw >> 16) & 0xFF
    opd1 = raw & 0xFFFF
    full_operand = raw & 0xFFFFFF if raw_opcode >= 20 else 0
    command = decode_executed(
        command_id=command_id, raw=raw, opcode=opcode, opd0=opd0,
        opd1=opd1, full_operand=full_operand, native_dim=native_dim,
        tile_rows=tile_rows,
        chain_id=chain_id, raw_instruction_idx=command_id,
        source=source, cpu_cycle=cpu_cycle, manifest=manifest,
    )
    if raw_opcode in INC_OPCODES:
        command.op = opcode_name(raw_opcode, opd0)
        command.inc_parent_opcode = raw_opcode
    return command
