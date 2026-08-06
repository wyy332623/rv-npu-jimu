"""NPU — Event Tracer for Computation Graph Derivation.

Hooks into NpuDeviceMini._execute() to record a flat event log with
def-use information per instruction.  Each event carries explicit
`defs` (resources written) and `uses` (resources read) on named
resources: DRAM addresses, VRF banks, pipeline registers, MRF, SRF,
and scalar regs.

Design: patch _execute as the single interception point so that
INC-expanded iterations (which recursively call _execute with base
opcodes and resolved addresses) are captured individually with
concrete DRAM addresses.
"""

from typing import List, Tuple

# ── Opcode constants (copied from npu_device_mini.py) ───────────────
OP_S_WR   = 0;  OP_S_RD   = 1
OP_V_RD   = 2;  OP_M_RD   = 3
OP_V_WR   = 5;  OP_M_WR   = 6
OP_MV_MUL = 7
OP_VV_ADD = 8;  OP_VV_A_SUB_B = 9;  OP_VV_B_SUB_A = 10
OP_VV_MUL = 11
OP_V_SIGM = 12; OP_V_TANH = 13; OP_V_RELU = 14; OP_VV_MAX = 15
OP_V_RD_INC = 16; OP_V_WR_INC = 17
OP_VV_ADD_INC = 18; OP_VV_MAX_INC = 19
OP_V_RD_DRAM = 20; OP_V_WR_DRAM = 21
OP_V_RD_DRAM_INC = 22; OP_V_WR_DRAM_INC = 23
OP_M_RD_DRAM = 24; OP_M_WR_DRAM = 25
OP_V_RD_3D = 26
OP_MV_MUL_INC = 27
OP_VV_MIN = 30
OP_VV_MUL_INC = 31
OP_VV_A_SUB_B_INC = 32; OP_VV_B_SUB_A_INC = 33
OP_V_EXP  = 37
OP_S_RECIP = 35; OP_S_SQRT = 38; OP_SS_MUL = 40
OP_V_GELU = 42
OP_V_FUNC = 43
OP_SS_ADD = 44
OP_INST_ISSUE = 45

INC_OPCODES = {
    OP_V_RD_INC, OP_V_WR_INC, OP_V_RD_DRAM_INC, OP_V_WR_DRAM_INC,
    OP_MV_MUL_INC, OP_VV_ADD_INC, OP_VV_MAX_INC, OP_VV_MUL_INC,
    OP_VV_A_SUB_B_INC, OP_VV_B_SUB_A_INC,
}

SUB_SOFTMAX   = 0
SUB_LAYERNORM = 1

# ── Memory targets (for V_RD / V_WR special cases) ──────────────
MEM_FILL            = 12
MEM_SPU_ADD_REDUCE  = 14
MEM_SPU_MAX_REDUCE  = 15
MEM_SPU_ABSMAX_REDUCE = 16
MEM_SPU_BROADCAST   = 17
MEM_VEC_TO_MAT_ROW  = 18

# ── Opcode → name mapping ──────────────────────────────────────────
OPCODE_NAMES: dict[int, str] = {
    OP_S_WR: "S_WR",           OP_S_RD: "S_RD",
    OP_V_RD: "V_RD",           OP_M_RD: "M_RD",
    OP_V_WR: "V_WR",           OP_M_WR: "M_WR",
    OP_MV_MUL: "MV_MUL",
    OP_VV_ADD: "VV_ADD",       OP_VV_A_SUB_B: "VV_A_SUB_B",
    OP_VV_B_SUB_A: "VV_B_SUB_A", OP_VV_MUL: "VV_MUL",
    OP_V_SIGM: "V_SIGM",       OP_V_TANH: "V_TANH",
    OP_V_RELU: "V_RELU",       OP_VV_MAX: "VV_MAX",
    OP_V_RD_INC: "V_RD_INC",   OP_V_WR_INC: "V_WR_INC",
    OP_VV_ADD_INC: "VV_ADD_INC", OP_VV_MAX_INC: "VV_MAX_INC",
    OP_V_RD_DRAM: "V_RD_DRAM", OP_V_WR_DRAM: "V_WR_DRAM",
    OP_V_RD_DRAM_INC: "V_RD_DRAM_INC",
    OP_V_WR_DRAM_INC: "V_WR_DRAM_INC",
    OP_M_RD_DRAM: "M_RD_DRAM", OP_M_WR_DRAM: "M_WR_DRAM",
    OP_V_RD_3D: "V_RD_3D",
    OP_MV_MUL_INC: "MV_MUL_INC",
    OP_VV_MIN: "VV_MIN",
    OP_VV_MUL_INC: "VV_MUL_INC",
    OP_VV_A_SUB_B_INC: "VV_A_SUB_B_INC",
    OP_VV_B_SUB_A_INC: "VV_B_SUB_A_INC",
    OP_V_EXP: "V_EXP",
    OP_S_RECIP: "S_RECIP",     OP_S_SQRT: "S_SQRT", OP_SS_MUL: "SS_MUL",
    OP_V_GELU: "V_GELU",
    OP_V_FUNC: "V_FUNC",
    OP_SS_ADD: "SS_ADD",
    OP_INST_ISSUE: "INST_ISSUE",
}


def _opcode_name(opcode: int, opd0: int = 0) -> str:
    """Return human-readable opcode name, resolving V_FUNC sub-opcodes."""
    if opcode == OP_V_FUNC:
        if opd0 == SUB_SOFTMAX:
            return "V_FUNC/SOFTMAX"
        elif opd0 == SUB_LAYERNORM:
            return "V_FUNC/LAYERNORM"
        return f"V_FUNC/sub{opd0}"
    return OPCODE_NAMES.get(opcode, f"OP_{opcode}")


def _resolve_defs_uses(opcode: int, opd0: int, opd1: int,
                       full_operand: int) -> Tuple[list, list]:
    """Return (defs, uses) for a given instruction dispatch.

    This is a static table based on the emulator's _execute() semantics.
    INC variants are handled here too — when the tracer hooks _execute
    directly, it sees the base opcode with resolved addresses from
    INC expansion.
    """
    defs: list[tuple] = []
    uses: list[tuple] = []

    if opcode == OP_S_WR:
        defs.append(("REG", opd0))

    elif opcode == OP_S_RD:
        uses.append(("REG", opd0))

    elif opcode == OP_V_RD:
        if opd0 == MEM_FILL:
            # FILL: loads a scalar constant into every pipeline element
            # No VRF read; the scalar is encoded in opd1 (FP16 bits).
            # Model as a def of pipe+vpipe_a with no VRF use.
            pass
        elif opd0 == MEM_SPU_BROADCAST:
            # BROADCAST: reads one SRF element, fills pipeline with it
            uses.append(("SRF", opd1))
        else:
            uses.append(("VRF", opd0, opd1))
        defs.append(("pipe",))
        defs.append(("vpipe_a",))

    elif opcode == OP_M_RD:
        if opd0 == MEM_VEC_TO_MAT_ROW:
            # Reads accumulated row buffer → MRF
            defs.append(("MRF",))
        else:
            # Generic M_RD: sets pipeline and vpipe_a from row buffer
            defs.append(("pipe",))
            defs.append(("vpipe_a",))

    elif opcode == OP_V_WR:
        uses.append(("pipe",))
        # Determine def target based on opd0 (mem_target)
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
        uses.append(("pipe",))
        uses.append(("MRF",))
        defs.append(("pipe",))

    elif opcode in (OP_VV_ADD, OP_VV_A_SUB_B, OP_VV_B_SUB_A,
                    OP_VV_MIN, OP_VV_MAX):
        uses.append(("pipe",))
        uses.append(("vpipe_a",))
        defs.append(("pipe",))

    elif opcode == OP_VV_MUL:
        uses.append(("pipe",))
        uses.append(("vpipe_a",))
        defs.append(("pipe",))

    elif opcode in (OP_V_SIGM, OP_V_TANH, OP_V_RELU, OP_V_GELU, OP_V_EXP):
        uses.append(("pipe",))
        defs.append(("pipe",))

    elif opcode == OP_V_FUNC:
        sub = opd0
        if sub == SUB_SOFTMAX:
            uses.append(("pipe",))
            defs.append(("pipe",))
        elif sub == SUB_LAYERNORM:
            uses.append(("pipe",))
            uses.append(("VRF", 5, 0))  # IVRF = gamma
            uses.append(("VRF", 7, 0))  # AS0 = beta
            defs.append(("pipe",))

    elif opcode == OP_V_RD_DRAM:
        uses.append(("DRAM", full_operand))
        defs.append(("pipe",))
        defs.append(("vpipe_a",))

    elif opcode == OP_V_WR_DRAM:
        uses.append(("pipe",))
        defs.append(("DRAM", full_operand))

    elif opcode == OP_M_RD_DRAM:
        uses.append(("DRAM", full_operand))
        defs.append(("MRF",))

    elif opcode == OP_M_WR_DRAM:
        uses.append(("MRF",))
        defs.append(("DRAM", full_operand))

    elif opcode == OP_INST_ISSUE:
        pass  # no defs/uses

    elif opcode == OP_SS_ADD:
        uses.append(("SRF", opd0))
        defs.append(("SRF", opd1))

    elif opcode in (OP_S_RECIP, OP_S_SQRT):
        uses.append(("SRF", opd0))
        defs.append(("SRF", opd1))

    elif opcode == OP_SS_MUL:
        uses.append(("SRF", opd0))
        defs.append(("SRF", opd1))

    # INC variants are expanded by the emulator into base-opcode calls。
    # When the tracer hooks _execute, it sees base opcodes (V_RD_DRAM,。。
    # V_WR_DRAM, etc.) with resolved full_operand — so the INC-specific。
    # branches above are unreachable but kept as defensive fallbacks.。

    return defs, uses


def _memory_access(opcode: int, full_operand: int, native_dim: int):
    """Return an element-addressed DRAM access descriptor, if any."""
    if opcode in (OP_V_RD_DRAM, OP_V_WR_DRAM):
        elements = native_dim
    elif opcode in (OP_M_RD_DRAM, OP_M_WR_DRAM):
        elements = native_dim * native_dim
    else:
        return None
    direction = "read" if opcode in (OP_V_RD_DRAM, OP_M_RD_DRAM) else "write"
    return {
        "direction": direction,
        "address": int(full_operand),
        "elements": int(elements),
        "end_address": int(full_operand) + int(elements),
    }


class EventTracer:
    """Wraps NpuDeviceMini and records per-instruction def-use events.

    Patches _execute (not _push_instruction) so that INC-expanded
    iterations appear as individual events with resolved concrete
    addresses.

    Usage:
        npu = NpuDeviceMini(native_dim=8)
        tracer = EventTracer(npu)
        # ... run firmware via MMIO ...
        for ev in tracer.events:
            print(ev)
        tracer.unpatch()
    """

    def __init__(self, inner_device):
        self._inner = inner_device
        self.events: list[dict] = []
        self._event_idx = 0
        self._raw_inst = 0  # stores the raw instruction from _push_instruction
        self._raw_instruction_idx = -1
        self._expanded_idx = 0
        self._chain_id = 0

        # Save originals
        self._original_push = inner_device._push_instruction
        self._original_execute = inner_device._execute

        # Patch _push_instruction to capture the raw instruction word
        def patched_push(inst: int):
            self._raw_inst = inst
            self._raw_instruction_idx += 1
            self._expanded_idx = 0
            self._original_push(inst)

        # Patch _execute to record events.
        # New signature accepts pipeline/vpipe_a kwargs for chain threading.
        def patched_execute(opcode: int, opd0: int, opd1: int,
                            full_operand: int = 0,
                            pipeline=None, vpipe_a=None):
            # Record event BEFORE execution so we capture the
            # instruction dispatch, not the side-effects
            op_name = _opcode_name(opcode, opd0)
            defs, uses = _resolve_defs_uses(opcode, opd0, opd1,
                                            full_operand)
            raw_opcode = (self._raw_inst >> 24) & 0xFF
            event = {
                "idx": self._event_idx,
                "op": op_name,
                "opcode": opcode,
                "raw": self._raw_inst,
                "raw_instruction_idx": self._raw_instruction_idx,
                "expanded_idx": self._expanded_idx,
                "inc_parent_opcode": raw_opcode if raw_opcode in INC_OPCODES else None,
                "chain_id": self._chain_id,
                "defs": defs,
                "uses": uses,
                "memory": _memory_access(
                    opcode, full_operand, int(inner_device.native_dim)
                ),
            }
            self.events.append(event)
            self._event_idx += 1
            self._expanded_idx += 1

            # Delegate to original with pipeline/vpipe_a
            result = self._original_execute(
                opcode, opd0, opd1, full_operand,
                pipeline=pipeline, vpipe_a=vpipe_a)
            if opcode == OP_INST_ISSUE:
                self._chain_id += 1
            return result

        inner_device._push_instruction = patched_push
        inner_device._execute = patched_execute

    def unpatch(self):
        """Restore original _push_instruction and _execute methods."""
        self._inner._push_instruction = self._original_push
        self._inner._execute = self._original_execute

    def clear(self):
        """Reset events list and counter."""
        self.events.clear()
        self._event_idx = 0
        self._raw_instruction_idx = -1
        self._expanded_idx = 0
        self._chain_id = 0
