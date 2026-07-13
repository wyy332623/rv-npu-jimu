"""NPU — Micro-Op DAG: collapsed instruction-level events into semantic micro-ops.

Groups consecutive instructions into meaningful micro-operations by recognizing
common NPU instruction idioms:

  DRAM_LOAD:   V_RD_DRAM, V_WR            — load vector from DRAM → VRF
  MAT_LOAD:    M_RD_DRAM, M_WR            — load matrix row from DRAM → MRF
  DRAM_STORE:  V_WR_DRAM                  — write vector from pipeline → DRAM
  VREG_LD:     V_RD(fill/broadcast), V_WR — init pipeline from imm/SPU → VRF
  VREG_MOVE:   V_RD, V_WR                 — move VRF bank → bank
  MV_MUL:      V_RD, [V_RD,] MV_MUL, V_WR — matrix-vector multiply tile
  VV_BINOP:    V_RD, V_RD, VV_*, V_WR    — vector binary arithmetic
  V_UNARY:     V_RD, V_*, ..., V_WR       — vector unary (sigmoid, GELU, etc.)
  V_FUNC:      ..., V_FUNC/*, ..., V_WR   — complex micro-coded (softmax, LN)
  S_WR:        S_WR                        — scalar register write (config)
  M_ACC:       M_RD(vec_to_mat), M_WR     — accumulator row → MRF

The pipe/vpipe_a edges are hidden inside each micro-op node.  Only VRF, MRF,
SRF, and DRAM resource edges remain between micro-ops, making the dataflow
much more readable.
"""

from dataclasses import dataclass, field
from typing import Optional

from emulator.npu_event_trace import (
    OP_S_WR, OP_S_RD, OP_V_RD, OP_M_RD, OP_V_WR, OP_M_WR,
    OP_MV_MUL,
    OP_VV_ADD, OP_VV_A_SUB_B, OP_VV_B_SUB_A, OP_VV_MUL, OP_VV_MIN, OP_VV_MAX,
    OP_V_SIGM, OP_V_TANH, OP_V_RELU, OP_V_GELU, OP_V_EXP,
    OP_V_FUNC, OP_V_WR_DRAM, OP_V_RD_DRAM,
    OP_M_RD_DRAM, OP_M_WR_DRAM,
    OP_V_RD_DRAM_INC, OP_V_WR_DRAM_INC,
    OP_V_RD_INC, OP_V_WR_INC,
    OP_MV_MUL_INC,
    OP_VV_ADD_INC, OP_VV_MAX_INC, OP_VV_MUL_INC,
    OP_VV_A_SUB_B_INC, OP_VV_B_SUB_A_INC,
    MEM_FILL, MEM_SPU_BROADCAST, MEM_VEC_TO_MAT_ROW,
    SUB_SOFTMAX, SUB_LAYERNORM,
)

# ── Classification helpers ────────────────────────────────────────

_IS_BINARY_STR = frozenset({
    "VV_ADD", "VV_A_SUB_B", "VV_B_SUB_A", "VV_MUL", "VV_MIN", "VV_MAX",
    "VV_ADD_INC", "VV_MAX_INC", "VV_MUL_INC",
    "VV_A_SUB_B_INC", "VV_B_SUB_A_INC",
})

_IS_UNARY_STR = frozenset({
    "V_SIGM", "V_TANH", "V_RELU", "V_GELU", "V_EXP",
})

_BINARY_DISPLAY = {
    "VV_ADD": "VV_ADD", "VV_A_SUB_B": "VV_A_SUB_B",
    "VV_B_SUB_A": "VV_B_SUB_A", "VV_MUL": "VV_MUL",
    "VV_MIN": "VV_MIN", "VV_MAX": "VV_MAX",
    "VV_ADD_INC": "VV_ADD", "VV_MAX_INC": "VV_MAX",
    "VV_MUL_INC": "VV_MUL",
    "VV_A_SUB_B_INC": "VV_A_SUB_B",
    "VV_B_SUB_A_INC": "VV_B_SUB_A",
}

_UNARY_NAMES = {
    "V_SIGM": "V_SIGM", "V_TANH": "V_TANH",
    "V_RELU": "V_RELU", "V_GELU": "V_GELU", "V_EXP": "V_EXP",
}


# ── Micro-op node ─────────────────────────────────────────────────

@dataclass
class MicroOp:
    """One semantic micro-operation (collapsed instruction group)."""
    kind: str                           # "DRAM_LOAD", "MV_MUL", etc.
    name: str                           # human label, e.g. "MV_MUL [tile 2, row 1]"
    event_indices: list[int]            # indices into original event list
    defs: list[tuple]                   # resources defined (VRF/MRF/DRAM/SRF only)
    uses: list[tuple]                   # resources used  (VRF/MRF/DRAM/SRF only)
    detail: str = ""                    # optional extra annotation


# ── Collapser ─────────────────────────────────────────────────────

def collapse_to_micro_ops(events: list[dict]) -> list[MicroOp]:
    """Collapse instruction-level events into semantic micro-operations.

    Args:
        events: flat event list from EventTracer.

    Returns:
        List of MicroOp nodes with pipe/vpipe_a edges hidden.
    """
    ops: list[MicroOp] = []
    i = 0
    n = len(events)

    def _raw(i: int) -> int:
        """Extract raw instruction word from event."""
        return events[i].get("raw", 0)

    def _opd0(i: int) -> int:
        return (_raw(i) >> 20) & 0x3F

    def _opd1(i: int) -> int:
        return (_raw(i) >> 14) & 0x3F

    def _opcode(i: int) -> int:
        # Extract opcode from the event's 'op' name reverse-lookup is fragile,
        # so we store it during tracing via raw. Use the defs/uses structure.
        # Actually, we need the numeric opcode. Let's get it from the event.
        # The event stores 'op' as a string — we need the numeric code.
        # We'll infer from the structure instead.
        pass  # unused, we match on event['op'] string

    # Track MRF row state to produce precise ('MRF', row_idx) defs/uses.
    # The NPU has a single MRF; M_WR loads it, MV_MUL reads it.
    # Without row tracking, the MRF edge is invisible because
    # _resolve_defs_uses emits ('MRF',) (no coordinate) for M_WR
    # while MV_MUL lists MRF as a pipe-side use that gets filtered.
    mrf_row: int = -1  # last row loaded into MRF

    def _non_pipe(resources: list[tuple]) -> list[tuple]:
        """Filter out pipe/vpipe_a — only keep persistent resources."""
        return [r for r in resources
                if r[0] not in ("pipe", "vpipe_a")]

    while i < n:
        ev = events[i]
        op_str = ev["op"]

        # ── DRAM_LOAD: V_RD_DRAM, V_WR ───────────────────────────
        if op_str == "V_RD_DRAM" and i + 1 < n and events[i + 1]["op"] == "V_WR":
            dram_src = [r for r in ev["uses"] if r[0] == "DRAM"]
            vrf_dst = _non_pipe(events[i + 1]["defs"])
            ops.append(MicroOp(
                kind="DRAM_LOAD",
                name=f"DRAM_LOAD",
                event_indices=[i, i + 1],
                defs=vrf_dst,
                uses=dram_src,
                detail=_dram_detail(dram_src),
            ))
            i += 2
            continue

        # ── MAT_LOAD: M_RD_DRAM, M_WR ────────────────────────────
        if op_str == "M_RD_DRAM" and i + 1 < n and events[i + 1]["op"] == "M_WR":
            dram_src = [r for r in ev["uses"] if r[0] == "DRAM"]
            # Track MRF row state: compute row index from DRAM address
            # relative to projector base.
            dram_addr = dram_src[0][1] if dram_src else 0
            # Simple row heuristic: divide by (native_dim * native_dim * 4 bytes)
            # We just use a monotonic counter; what matters is that each
            # MAT_LOAD gets a unique row coordinate.
            mrf_row += 1
            mrf_key = ("MRF", mrf_row)
            ops.append(MicroOp(
                kind="MAT_LOAD",
                name="MAT_LOAD",
                event_indices=[i, i + 1],
                defs=[mrf_key],
                uses=dram_src,
                detail=_dram_detail(dram_src),
            ))
            i += 2
            continue

        # ── DRAM_STORE: V_WR_DRAM ────────────────────────────────
        if op_str == "V_WR_DRAM":
            dram_dst = [r for r in ev["defs"] if r[0] == "DRAM"]
            ops.append(MicroOp(
                kind="DRAM_STORE",
                name="DRAM_STORE",
                event_indices=[i],
                defs=dram_dst,
                uses=[],  # pipe input, hidden
                detail=_dram_detail(dram_dst),
            ))
            i += 1
            continue

        # ── DRAM_STORE: V_RD, V_WR_DRAM  (V_RD stages pipeline for store) ─
        if op_str == "V_RD" and i + 1 < n and events[i + 1]["op"] == "V_WR_DRAM":
            uses = _non_pipe(ev["uses"])
            dram_dst = [r for r in events[i + 1]["defs"] if r[0] == "DRAM"]
            ops.append(MicroOp(
                kind="DRAM_STORE",
                name="DRAM_STORE",
                event_indices=[i, i + 1],
                defs=dram_dst,
                uses=uses,
                detail=_dram_detail(dram_dst),
            ))
            i += 2
            continue

        # ── DRAM_LOAD standalone (V_RD_DRAM not followed by V_WR) ──
        if op_str == "V_RD_DRAM" and (i + 1 >= n or events[i + 1]["op"] != "V_WR"):
            dram_src = [r for r in ev["uses"] if r[0] == "DRAM"]
            ops.append(MicroOp(
                kind="DRAM_LOAD",
                name="DRAM_LOAD",
                event_indices=[i],
                defs=[],  # goes to pipe, consumed by next op
                uses=dram_src,
                detail=_dram_detail(dram_src),
            ))
            i += 1
            continue

        # ── VREG_LD: V_RD(fill), V_WR, V_WR  (fill + SPU write) ─────
        if (op_str == "V_RD" and not ev["uses"]
                and i + 2 < n and events[i + 1]["op"] == "V_WR"
                and events[i + 2]["op"] == "V_WR"):
            vrf_dst1 = _non_pipe(events[i + 1]["defs"])
            vrf_dst2 = _non_pipe(events[i + 2]["defs"])
            ops.append(MicroOp(
                kind="VREG_LD",
                name="VREG_LD(fill+spu)",
                event_indices=[i, i + 1, i + 2],
                defs=vrf_dst1 + vrf_dst2,
                uses=[],
            ))
            i += 3
            continue

        # ── VREG_LD: V_RD(fill), V_WR ────────────────────────────
        #    V_RD with no uses (= MEM_FILL) → initialise pipeline with immediate
        if (op_str == "V_RD" and not ev["uses"]
                and i + 1 < n and events[i + 1]["op"] == "V_WR"):
            vrf_dst = _non_pipe(events[i + 1]["defs"])
            ops.append(MicroOp(
                kind="VREG_LD",
                name="VREG_LD(fill)",
                event_indices=[i, i + 1],
                defs=vrf_dst,
                uses=[],
            ))
            i += 2
            continue

        # ── VREG_LD: V_RD(broadcast from SRF), V_WR ─────────────
        if (op_str == "V_RD" and ev["uses"] and ev["uses"][0][0] == "SRF"
                and i + 1 < n and events[i + 1]["op"] == "V_WR"):
            srf_src = ev["uses"]
            vrf_dst = _non_pipe(events[i + 1]["defs"])
            ops.append(MicroOp(
                kind="VREG_LD",
                name="VREG_LD(bcast)",
                event_indices=[i, i + 1],
                defs=vrf_dst,
                uses=srf_src,
            ))
            i += 2
            continue

        # ── M_ACC: M_RD(vec_to_mat_row), M_WR ────────────────────
        #    Accumulator row → MRF. Updates mrf_row tracker.
        if (op_str == "M_RD" and i + 1 < n and events[i + 1]["op"] == "M_WR"
                and any(d[0] == "VRF" and d[1] == MEM_VEC_TO_MAT_ROW
                        for d in events[i + 1].get("defs", _non_pipe(events[i + 1]["defs"])))):
            ops.append(MicroOp(
                kind="M_ACC",
                name="M_ACC",
                event_indices=[i, i + 1],
                defs=[("MRF", mrf_row)],  # reuses whatever row was last loaded
                uses=[],
            ))
            i += 2
            continue

        # ── MV_MUL: [V_RD,] MV_MUL, [V_WR+, ATTN_PAD] ────────────────
        #    The compute core. V_RD loads the vector, MV_MUL multiplies
        #    pipeline vector × MRF, then optional V_WR stores result and
        #    ATTN_PAD signals attention padding boundary.
        #    Handles these firmware patterns:
        #      V_RD, V_RD, MV_MUL, V_WR       (standard tile)
        #      V_RD, MV_MUL, V_WR             (single staging)
        #      MV_MUL, V_WR                   (pipeline already loaded)
        #      V_RD, MV_MUL                   (ATTN_PAD, no trailing WR)
        #      MV_MUL                         (ATTN_PAD, pipeline loaded)
        absorbed_prefix = False
        if op_str == "V_RD" and i + 1 < n and events[i + 1]["op"] in ("MV_MUL", "MV_MULINC"):
            absorbed_prefix = True
            j_start = i
        elif op_str in ("MV_MUL", "MV_MULINC"):
            j_start = i
        elif op_str == "V_RD" and i + 2 < n:
            j = i + 1
            while j < n and events[j]["op"] == "V_RD":
                j += 1
            if j < n and events[j]["op"] in ("MV_MUL", "MV_MULINC"):
                absorbed_prefix = True
                j_start = i
            else:
                j_start = None
        else:
            j_start = None

        if j_start is not None:
            j = j_start
            # Skip past V_RDs to find MV_MUL
            while j < n and events[j]["op"] == "V_RD":
                j += 1
            if j < n and events[j]["op"] in ("MV_MUL", "MV_MULINC"):
                mul_idx = j
                j += 1
                # Collect trailing V_WR(s) and optional ATTN_PAD
                # ATTN_PAD is a MV_MULINC branch that has no V_WR after it.
                # We absorb V_WR up to (but not including) the next
                # V_RD_DRAM, M_RD_DRAM, V_FUNC, or a V_RD that starts
                # a new operand load.
                while j < n and events[j]["op"] == "V_WR":
                    j += 1
                instr_range = list(range(j_start, j))
                uses = []
                for k in instr_range:
                    if events[k]["op"].startswith("V_RD"):
                        uses.extend(_non_pipe(events[k]["uses"]))
                defs = []
                for k in instr_range:
                    if events[k]["op"] == "V_WR":
                        defs.extend(_non_pipe(events[k]["defs"]))

                # Add MRF use — MV_MUL reads the MRF row that was
                # last loaded by the most recent MAT_LOAD or M_ACC.
                if mrf_row >= 0:
                    uses.append(("MRF", mrf_row))

                ops.append(MicroOp(
                    kind="MV_MUL",
                    name="MV_MUL",
                    event_indices=instr_range,
                    defs=defs,
                    uses=uses,
                ))
                i = j
                continue

        # ── VV_BINOP: V_RD*, VV_*, [V_WR]+ ────────────────────
        #    Generalized: V_RD or V_RD_DRAM are both valid operand loads.
        #    Handles patterns like:
        #      V_RD, V_RD, VV_ADD, V_WR
        #      V_RD, V_RD_DRAM, VV_ADD, V_WR  (attention residual add)
        if op_str in ("V_RD", "V_RD_DRAM") and i + 2 < n:
            j = i + 1
            while j < n and events[j]["op"] in ("V_RD", "V_RD_DRAM"):
                j += 1
            if j < n and events[j]["op"] in _IS_BINARY_STR:
                binop = events[j]["op"]
                binop_name = _BINARY_DISPLAY.get(binop, "VV_BINOP")
                j += 1
                while j < n and events[j]["op"] == "V_WR":
                    j += 1
                if j > i + 2:  # at least load, load, op, wr
                    instr_range = list(range(i, j))
                    uses = []
                    for k in instr_range:
                        if events[k]["op"].startswith("V_RD"):
                            uses.extend(_non_pipe(events[k]["uses"]))
                    defs = []
                    for k in instr_range:
                        if events[k]["op"] == "V_WR":
                            defs.extend(_non_pipe(events[k]["defs"]))

                    ops.append(MicroOp(
                        kind="VV_BINOP",
                        name=binop_name,
                        event_indices=instr_range,
                        defs=defs,
                        uses=uses,
                    ))
                    i = j
                    continue

        # ── V_UNARY: V_RD, V_*, ..., V_WR ────────────────────────
        #    Single-operand vector operation.
        if op_str == "V_RD" and i + 2 < n:
            j = i + 1
            while j < n and events[j]["op"] in _IS_UNARY_STR:
                j += 1
            if j > i + 1:  # found at least one unary op
                while j < n and events[j]["op"] == "V_WR":
                    j += 1
                if j > i + 2:  # at least V_RD, V_OP, V_WR
                    instr_range = list(range(i, j))
                    unop = events[i + 1]["op"]
                    unop_name = _UNARY_NAMES.get(events[i + 1]["op"], "V_UNARY")

                    uses = []
                    for k in instr_range:
                        if events[k]["op"].startswith("V_RD"):
                            uses.extend(_non_pipe(events[k]["uses"]))
                    defs = []
                    for k in instr_range:
                        if events[k]["op"] == "V_WR":
                            defs.extend(_non_pipe(events[k]["defs"]))

                    ops.append(MicroOp(
                        kind="V_UNARY",
                        name=unop_name,
                        event_indices=instr_range,
                        defs=defs,
                        uses=uses,
                    ))
                    i = j
                    continue

        # ── V_FUNC: complex micro-coded ops (softmax, layernorm) ──
        #    These have internal V_RD/V_WR sequences between V_FUNC calls.
        #    We group from the first V_RD until we see a V_WR_DRAM or a new
        #    pattern starts.  Simpler approach: group consecutive V_FUNC/*,
        #    V_RD, V_WR sequences that don't match any other pattern.
        if op_str.startswith("V_FUNC"):
            func_name = op_str  # e.g. "V_FUNC/LAYERNORM" or "V_FUNC/SOFTMAX"
            j = i + 1
            # Consume the V_RD/V_WR sequences that follow
            while j < n and events[j]["op"] in ("V_RD", "V_WR", "V_GELU",
                                                  "V_SIGM", "V_TANH", "V_RELU",
                                                  "V_EXP", "V_FUNC/SOFTMAX",
                                                  "V_FUNC/LAYERNORM",
                                                  "VV_ADD", "VV_MUL",
                                                  "S_WR", "S_RD"):
                # Stop if we hit DRAM access (start of a new logical op)
                if events[j]["op"] in ("V_RD_DRAM", "V_WR_DRAM",
                                        "M_RD_DRAM", "M_WR_DRAM"):
                    break
                j += 1

            instr_range = list(range(i, j))
            uses = []
            defs = []
            for k in instr_range:
                uses.extend(_non_pipe(events[k]["uses"]))
                defs.extend(_non_pipe(events[k]["defs"]))

            kind = "LAYERNORM" if "LAYERNORM" in func_name else \
                   "SOFTMAX" if "SOFTMAX" in func_name else "V_FUNC"

            ops.append(MicroOp(
                kind=kind,
                name=func_name,
                event_indices=instr_range,
                defs=defs,
                uses=uses,
            ))
            i = j
            continue

        # ── V_GELU standalone (sometimes appears without V_RD prefix) ──
        if op_str == "V_GELU":
            j = i + 1
            while j < n and events[j]["op"] in ("V_WR", "V_RD"):
                j += 1
            instr_range = list(range(i, j))
            uses = []
            defs = []
            for k in instr_range:
                uses.extend(_non_pipe(events[k]["uses"]))
                defs.extend(_non_pipe(events[k]["defs"]))
            ops.append(MicroOp(
                kind="V_UNARY",
                name="V_GELU",
                event_indices=instr_range,
                defs=defs,
                uses=uses,
            ))
            i = j
            continue

        # ── VREG_MOVE: V_RD, V_WR (simple bank-to-bank move) ─────
        if op_str == "V_RD" and i + 1 < n and events[i + 1]["op"] == "V_WR":
            uses = _non_pipe(ev["uses"])
            defs = _non_pipe(events[i + 1]["defs"])
            ops.append(MicroOp(
                kind="VREG_MOVE",
                name="VREG_MOVE",
                event_indices=[i, i + 1],
                defs=defs,
                uses=uses,
            ))
            i += 2
            continue

        # ── Standalone V_WR (pipe result stored outside of pattern) ───
        #    Occurs after MV_MUL in ATTN_PAD branches where the
        #    firmware writes MV_MUL output to VRF separately.
        if op_str == "V_WR":
            ops.append(MicroOp(
                kind="V_WR",
                name="V_WR",
                event_indices=[i],
                defs=_non_pipe(ev["defs"]),
                uses=[],  # pipe input, hidden
            ))
            i += 1
            continue

        # ── M_RD: standalone M_RD (vec_to_mat_row) → MRF load ──────
        #    After loading K/V rows into VRF[18], the firmware uses M_RD
        #    to load the accumulator result into MRF for the next MV_MUL.
        #    The fallback would emit bare ("MRF",) with no row coordinate,
        #    which never matches the ("MRF", row_idx) from MAT_LOAD/M_ACC.
        if op_str == "M_RD" and ev["defs"]:
            # Only handle M_RD with MRF def (vec_to_mat_row variant).
            # Generic M_RD (defs pipe/vpipe_a) falls through to fallback.
            if any(d[0] == "MRF" for d in ev["defs"]):
                mrf_row += 1
                # Add VRF[18][0] as an implicit use — the M_RD reads the
                # row buffer built by preceding V_WR(VRF[18]) writes.
                # This connects the standalone DRAM_LOAD(→VRF[18]) nodes
                # to this M_RD via the def-use chain, eliminating the
                # appearance of orphaned row-buffer loads.
                ops.append(MicroOp(
                    kind="M_RD",
                    name="M_RD",
                    event_indices=[i],
                    defs=[("MRF", mrf_row)],
                    uses=[("VRF", MEM_VEC_TO_MAT_ROW, 0)],
                ))
                i += 1
                continue

        # ── S_WR: scalar register write (config) ─────────────────
        if op_str == "S_WR":
            reg_dst = ev["defs"]
            ops.append(MicroOp(
                kind="S_WR",
                name="S_WR",
                event_indices=[i],
                defs=_non_pipe(reg_dst),
                uses=[],
                detail=f"REG[{reg_dst[0][1]}]" if reg_dst else "",
            ))
            i += 1
            continue

        # ── Fallback: ungrouped single instruction ───────────────
        ops.append(MicroOp(
            kind=op_str,
            name=op_str,
            event_indices=[i],
            defs=_non_pipe(ev["defs"]),
            uses=_non_pipe(ev["uses"]),
        ))
        i += 1

    return ops


def _dram_detail(dram_resources: list[tuple]) -> str:
    """Format DRAM address for detail annotation."""
    if not dram_resources:
        return ""
    r = dram_resources[0]
    if r[0] == "DRAM" and len(r) > 1:
        return f"DRAM[{r[1]:#x}]"
    return ""


# ── Micro-op DAG builder ──────────────────────────────────────────

def build_micro_op_dag(
    micro_ops: list[MicroOp],
    *,
    pipe_edges: bool = True,
) -> tuple[list[MicroOp], list[tuple[int, int, tuple]]]:
    """Build def-use DAG from micro-ops.

    Tracks VRF, MRF, SRF, DRAM resources via standard def-use.
    If *pipe_edges* is True (default), also adds implicit edges
    for pipe/vpipe_a values that flow between consecutive micro-ops.
    Without pipe edges, computation chains (MV_MUL -> SOFTMAX -> STORE)
    appear disconnected because pipe is filtered out of uses/defs.

    Returns:
        (nodes, edges) where edges are (src_idx, dst_idx, resource).
    """
    last_def: dict[tuple, int] = {}
    edges: list[tuple[int, int, tuple]] = []
    seen: set[tuple[int, int, tuple]] = set()

    def _add_edge(src: int, dst: int, res: tuple) -> None:
        key = (src, dst, res)
        if key not in seen and src != dst:
            edges.append(key)
            seen.add(key)

    for mop_idx, mop in enumerate(micro_ops):
        for use in mop.uses:
            if use in last_def:
                _add_edge(last_def[use], mop_idx, use)
        for df in mop.defs:
            last_def[df] = mop_idx

    # ── Phase 2: implicit pipe/vpipe_a edges ───────────────────────
    #
    # The pipe register carries the live computation vector between
    # consecutive micro-ops.  When a micro-op defines pipe and the
    # next micro-op uses pipe (but neither side appears in the
    # non-pipe uses/defs), there is no edge in Phase 1.
    #
    # We scan the original events to find pipe writers and readers,
    # then bridge across micro-op boundaries.
    #
    if pipe_edges:
        pipe_last_writer: dict[str, int] = {}  # 'pipe' or 'vpipe_a' -> mop_idx
        # Build event -> micro-op index map
        event_to_mop: dict[int, int] = {}
        for mi, m in enumerate(micro_ops):
            for ei in m.event_indices:
                event_to_mop[ei] = mi

        # Scan ALL events to track pipe state
        # We need the events list — get it from the micro-op detail
        # Actually, we already consumed events in collapse_to_micro_ops.
        # Instead, reconstruct from micro-ops by scanning their implicit defs/uses.
        #
        # Simpler approach: run a second pass over micro-ops tracking
        # which micro-op last defined pipe or vpipe_a, and which
        # micro-op next needs it.
        #
        # A micro-op "provides pipe" if any of its constituent events
        # defines pipe and no later event in the SAME micro-op
        # overwrites pipe.  A micro-op "needs pipe" if any of its
        # constituent events uses pipe and no earlier event in the
        # SAME micro-op defines pipe.
        #
        # But we don't have the raw events here.  Instead, we can
        # use a heuristic based on micro-op kinds:
        #
        # "Provides pipe" kinds (write pipe as side effect):
        #   DRAM_LOAD, VREG_LD, VREG_MOVE (the V_RD part),
        #   MV_MUL (result), VV_BINOP (result), V_UNARY (result),
        #   SOFTMAX, LAYERNORM, DRAM_STORE (the V_RD part)
        #
        # "Needs pipe" kinds (read pipe as input):
        #   MV_MUL (input), VV_BINOP (both inputs), V_UNARY (input),
        #   SOFTMAX, LAYERNORM, DRAM_STORE
        #
        # Actually, let's just track pipe by scanning micro-ops
        # sequentially.  If mop A defines something visible (VRF, DRAM)
        # and mop B is the very next computation micro-op (not S_WR),
        # then B depends on A's output via pipe.
        #
        # Simplest correct approach: each non-config micro-op depends
        # on the micro-op that last "produced" a pipe value.
        #
        PIPE_PRODUCERS = frozenset({
            "DRAM_LOAD", "VREG_LD", "VREG_MOVE", "M_RD",
            "MV_MUL", "VV_BINOP", "V_UNARY", "SOFTMAX", "LAYERNORM",
            "V_WR",  # V_WR reads pipe then writes VRF (consumer + producer)
        })
        PIPE_CONSUMERS = frozenset({
            "MV_MUL", "VV_BINOP", "V_UNARY", "SOFTMAX", "LAYERNORM",
            "DRAM_STORE", "VREG_MOVE", "V_WR",
            "M_RD",
        })

        last_producer: int | None = None
        for mop_idx, mop in enumerate(micro_ops):
            if mop.kind == "S_WR":
                continue  # config, doesn't affect pipe
            if mop.kind in PIPE_CONSUMERS and last_producer is not None:
                # This micro-op needs pipe input from the last producer
                _add_edge(last_producer, mop_idx, ("pipe",))
            if mop.kind in PIPE_PRODUCERS:
                last_producer = mop_idx

    return micro_ops, edges


# ── Renderers ─────────────────────────────────────────────────────

def micro_op_dag_to_text(
    nodes: list[MicroOp],
    edges: list[tuple[int, int, tuple]],
    max_nodes: int | None = None,
) -> str:
    """Render micro-op DAG as human-readable text."""
    preds: dict[int, list[tuple[int, tuple]]] = {}
    for src, dst, res in edges:
        preds.setdefault(dst, []).append((src, res))

    display = nodes[:max_nodes] if max_nodes else nodes
    lines = []
    for mop_idx, mop in enumerate(display):
        detail = f"  {mop.detail}" if mop.detail else ""
        instr_range = f"[{mop.event_indices[0]}-{mop.event_indices[-1]}]" if len(mop.event_indices) > 1 else f"[{mop.event_indices[0]}]"
        lines.append(
            f"{mop_idx:3d} {mop.kind:16s} {mop.name:20s} {instr_range:12s}"
            f" uses={_fmt_res(mop.uses)} defs={_fmt_res(mop.defs)}{detail}"
        )
        for src_idx, res in sorted(preds.get(mop_idx, [])):
            src_name = nodes[src_idx].name if src_idx < len(nodes) else "?"
            lines.append(
                f"    <- [{src_idx} {src_name}] via {_resource_str(res)}"
            )

    total = len(nodes)
    shown = len(display)
    lines.append(f"\nTotal: {total} micro-ops, {len(edges)} edges"
                 + (f" (showing first {shown})" if max_nodes and max_nodes < total else ""))
    return "\n".join(lines)


def micro_op_dag_to_dot(
    nodes: list[MicroOp],
    edges: list[tuple[int, int, tuple]],
    path: str | None = None,
    max_nodes: int | None = None,
) -> str:
    """Render micro-op DAG as Graphviz DOT.

    Node coloring by kind:
      DRAM_LOAD/DRAM_STORE → lightblue
      MAT_LOAD             → wheat
      MV_MUL               → gold
      VV_BINOP             → lightyellow
      V_UNARY              → honeydew
      LAYERNORM/SOFTMAX    → thistle
      S_WR                 → lightgray
      default              → white
    """
    COLORS = {
        "DRAM_LOAD": "lightblue",
        "DRAM_STORE": "lightblue",
        "MAT_LOAD": "wheat",
        "MV_MUL": "gold",
        "VV_BINOP": "lightyellow",
        "V_UNARY": "honeydew",
        "LAYERNORM": "thistle",
        "SOFTMAX": "thistle",
        "V_FUNC": "thistle",
        "S_WR": "lightgray",
        "VREG_LD": "mistyrose",
        "VREG_MOVE": "mistyrose",
        "M_ACC": "wheat",
    }

    display = nodes[:max_nodes] if max_nodes else nodes
    idx_set = {i for i in range(len(display))}

    # Deduplicate edges for DOT
    edge_labels: dict[tuple[int, int], list[str]] = {}
    for src, dst, res in edges:
        if src in idx_set and dst in idx_set:
            key = (src, dst)
            label = _resource_str(res)
            if key not in edge_labels:
                edge_labels[key] = []
            if label not in edge_labels[key]:
                edge_labels[key].append(label)

    lines = ['digraph micro_op_dag {']
    lines.append('  node [shape=box style=filled];')
    lines.append('  rankdir=TB;')

    for i, mop in enumerate(display):
        color = COLORS.get(mop.kind, "white")
        detail = f"\\n{mop.detail}" if mop.detail else ""
        instr_info = f" ({len(mop.event_indices)} instr)" if len(mop.event_indices) > 1 else ""
        label = f"{i}: {mop.name}{instr_info}{detail}"
        label = label.replace('"', '\\"')
        lines.append(f'  n{i} [label="{label}" fillcolor="{color}"];')

    for (src, dst), labels in sorted(edge_labels.items()):
        label = ", ".join(labels).replace('"', '\\"')
        lines.append(f'  n{src} -> n{dst} [label="{label}"];')

    lines.append('}')
    dot_str = "\n".join(lines) + "\n"

    if path is not None:
        with open(path, "w") as f:
            f.write(dot_str)

    return dot_str


def _resource_str(res: tuple) -> str:
    """Format a resource tuple as a readable string."""
    if res[0] == "DRAM":
        return f"DRAM[{res[1]:#x}]"
    elif res[0] == "VRF":
        return f"VRF[{res[1]}][{res[2]}]"
    elif res[0] == "MRF":
        return "MRF"
    elif res[0] == "SRF":
        return f"SRF[{res[1]}]"
    elif res[0] == "REG":
        return f"REG[{res[1]}]"
    else:
        return str(res)


def _fmt_res(resources: list[tuple]) -> str:
    """Format a list of resources compactly."""
    if not resources:
        return "-"
    return ",".join(_resource_str(r) for r in resources)


# ── DAG connectivity diagnostics ──────────────────────────────────

def check_dag_connectivity(
    micro_ops: list[MicroOp],
    edges: list[tuple[int, int, tuple]],
    dim: int = 2,
    hidden_size: int = 4,
    seq_len: int = 2,
) -> str:
    """Analyze and classify all disconnected or orphaned nodes.

    Scans the micro-op DAG and categorizes nodes with missing
    incoming or outgoing edges.  Produces a human-readable report
    explaining why each category is or isn't a concern.

    Args:
        micro_ops: list of MicroOp nodes.
        edges: edge list from build_micro_op_dag().
        dim: native dimension (for address classification).
        hidden_size: model hidden size.
        seq_len: sequence length.

    Returns:
        Multi-line report string.
    """
    from collections import Counter
    num_tiles = hidden_size // dim
    total = len(micro_ops)
    has_in = [any(e[1] == i for e in edges) for i in range(total)]
    has_out = [any(e[0] == i for e in edges) for i in range(total)]

    lines = []
    lines.append(f"DAG Connectivity Report: {total} nodes, {len(edges)} edges")
    lines.append("=" * 70)

    # ── Fully disconnected ──
    fully = [(i, micro_ops[i]) for i in range(total)
             if not has_in[i] and not has_out[i]]
    kinds = Counter(m.kind for _, m in fully)
    lines.append(f"\n1. Fully disconnected: {len(fully)}")
    for k, c in sorted(kinds.items(), key=lambda x: -x[1]):
        lines.append(f"   {k:20s}: {c}")
    lines.append("   → S_WR: configuration registers, no data deps (expected)")
    lines.append("   → VREG_LD: fill constants, produce VRF values (expected)")

    # ── DRAM_LOAD with no incoming edges ──
    dram_root = []
    for i in range(total):
        m = micro_ops[i]
        if m.kind == "DRAM_LOAD" and not has_in[i]:
            addr = next((r[1] for r in m.uses if r[0] == "DRAM"), None)
            if addr is not None:
                region, pos = _classify_dram_addr(addr, num_tiles, dim)
                dram_root.append((i, addr, region, pos, m.kind))

    lines.append(f"\n2. DRAM_LOAD roots (no writer, pre-loaded data): {len(dram_root)}")
    regions = Counter(r for _, _, r, _, _ in dram_root)
    for reg, c in sorted(regions.items(), key=lambda x: -x[1]):
        lines.append(f"   {reg:12s}: {c}")
    lines.append("   → X: input data loaded by test harness (expected)")
    lines.append("   → UNIT_VEC: identity matrix rows (expected)")
    lines.append("   → LN params: gamma/beta for layernorm (expected)")
    lines.append("   → K/V beyond seq_len: TILE_BUILD backward scan reads")
    lines.append("     padding rows past saved data (known limitation)")

    # ── MAT_LOAD with no incoming edges ──
    mat_root = sum(1 for i, m in enumerate(micro_ops)
                   if m.kind == "MAT_LOAD" and not has_in[i])
    lines.append(f"\n3. MAT_LOAD roots (no writer, pre-loaded weights): {mat_root}")
    lines.append("   → Weight tiles loaded by test harness (expected)")

    # ── DRAM_STORE with no outgoing edges ──
    store_leaves = []
    for i in range(total):
        m = micro_ops[i]
        if m.kind == "DRAM_STORE" and not has_out[i]:
            addr = next((r[1] for r in m.defs if r[0] == "DRAM"), None)
            if addr is not None:
                region, pos = _classify_dram_addr(addr, num_tiles, dim)
                store_leaves.append((i, addr, region, pos))

    lines.append(f"\n4. DRAM_STORE leaves (written but never read): {len(store_leaves)}")
    for i, addr, reg, pos in store_leaves:
        lines.append(f"   n{i}: {reg}[{pos}] at 0x{addr:x}")
    lines.append("   → OUT: final output written at end of layer (expected)")

    # ── Summary ──
    connected = sum(1 for i in range(total) if has_in[i] and has_out[i])
    roots = sum(1 for i in range(total) if has_out[i] and not has_in[i])
    leaves = sum(1 for i in range(total) if has_in[i] and not has_out[i])
    lines.append(f"\n5. Summary")
    lines.append(f"   Connected (in+out): {connected}")
    lines.append(f"   Roots (out only):   {roots}")
    lines.append(f"   Leaves (in only):   {leaves}")
    lines.append(f"   Isolated (no edges): {len(fully)}")
    lines.append(f"   Total:               {total}")

    return "\n".join(lines)


# ── DRAM region classification (from npu_op_graph) ────────────────
# Duplicated here to avoid circular import.
_DRAM_REGIONS: list[tuple[int, str, int]] = [
    (0x200, "Q",         0x100),
    (0x300, "K",         0x100),
    (0x400, "V",         0x100),
    (0x500, "SCRATCH",   0x080),
    (0x580, "SO_SCRATCH",0x080),
    (0x600, "Z",         0x020),
    (0x620, "LN1",       0x020),
    (0x640, "GELU",      0x020),
    (0x700, "RES",       0x100),
    (0x800, "OUT",       0x100),
    (0x900, "UNIT_VEC",  0x080),
]


def _classify_dram_addr(addr: int, num_tiles: int, native_dim: int = 8) -> tuple[str, int]:
    """Classify DRAM address into (region_name, position).

    The firmware saves tile-row data at:
      SAVE_*_BASE + pos * num_tiles * 8 + tr * 8
    where 8 = 4 elements * 2 bytes (fp16) per tile row in DRAM addresses.

    Returns (region_name, position).  Position is -1 if the address
    doesn't fall into a known tensor save region (e.g. weight area,
    input X area).  Region is "WEIGHT" for weight matrix area between
    X data and save regions, "X" for input data area before 0x200.
    """
    # Input X area: 0 to first save region base
    if addr < 0x200:
        return ("X", addr // (num_tiles * native_dim))
    stride = num_tiles * 8  # bytes per position in DRAM address space
    for i, (base, name, _span) in enumerate(reversed(_DRAM_REGIONS)):
        if addr >= base:
            if i > 0:
                upper = list(reversed(_DRAM_REGIONS))[i - 1][0]
            else:
                upper = base + _DRAM_REGIONS[-1][2] * native_dim
            if addr < upper:
                pos = (addr - base) // stride if stride > 0 else 0
                return (name, pos)
    return ("UNKNOWN", addr)


# ── DRAM-flow clustering ─────────────────────────────────────────

@dataclass
class TensorCluster:
    """A cluster of micro-ops bounded by DRAM load/store edges.

    Represents a "tensor lifecycle": a DRAM load brings data on-chip,
    a series of compute micro-ops process it, and a DRAM store writes
    the result back.
    """
    # Unique identifier
    id: str                             # e.g. "K[0]_proj"
    # Semantic name
    label: str                          # e.g. "K[0] Projection"
    # Micro-op indices in this cluster
    member_indices: list[int]
    # DRAM region being produced (written at the end)
    produced_region: str                # e.g. "K"
    produced_position: int              # e.g. 0
    # DRAM regions consumed (read at the start)
    consumed_regions: set[str]          # e.g. {"X"} or {"K", "Q"}
    # Summary stats
    dram_load_bytes: int = 0
    dram_store_bytes: int = 0
    compute_flops: int = 0
    # First and last micro-op index
    first_idx: int = 0
    last_idx: int = 0


# ── Cluster naming helpers ────────────────────────────────────────

_TENSOR_NAMES: dict[str, str] = {
    "X": "X",
    "Q": "Q",
    "K": "K",
    "V": "V",
    "Z": "Z",
    "SCRATCH": "Scratch",
    "SO_SCRATCH": "SOut",
    "LN1": "LN1",
    "GELU": "GELU",
    "RES": "Res",
    "OUT": "Out",
    "UNIT_VEC": "Uvec",
    "WEIGHT": "Wt",
}

# DRAM save regions that use per-position indexing vs transient scratch buffers.
_POSITIONAL_SAVE_REGIONS = {"Q", "K", "V", "RES", "OUT"}
_TRANSIENT_REGIONS = {"SCRATCH", "Z", "LN1", "GELU", "SO_SCRATCH"}


def _cluster_label(region: str, pos: int, is_store: bool,
                   consumed_regions: Optional[set[str]] = None) -> str:
    """Generate a human-readable cluster label using BERT model naming."""
    cons = consumed_regions or set()
    pos_str = f" p{pos}" if pos >= 0 else ""

    # Projections
    if region == "K" and "WEIGHT" in cons:
        return f"K Proj{pos_str}"
    if region == "V" and "WEIGHT" in cons:
        return f"V Proj{pos_str}"
    if region == "Q" and "WEIGHT" in cons:
        return f"Q Proj{pos_str}"

    # Attention: K.T + Score + Softmax + V.T → writes prob to SCRATCH
    if region == "SCRATCH" and "V" in cons and "K" in cons:
        return f"Attn Score+Softmax{pos_str}"
    # Attention: V.T + Context → writes Z
    if region == "Z" and "V" in cons:
        return f"Attn Context{pos_str}"

    # Self-Output projection
    if region == "SO_SCRATCH" and "WEIGHT" in cons:
        return f"Self-Output{pos_str}"

    # Skip-connection save
    if region == "RES" and "X" in cons:
        return f"Save X→RES{pos_str}"

    # Residual add 1: SO_SCRATCH + X (from RES save)
    if region == "SCRATCH" and "SO_SCRATCH" in cons and "X" not in cons:
        return f"Residual Add1{pos_str}"
    # Residual add 1 — second tile row
    if region == "SCRATCH" and "SO_SCRATCH" not in cons and "X" in cons and "WEIGHT" not in cons and not cons == {"SCRATCH"}:
        return f"Residual Add1 t1{pos_str}"

    # LayerNorm 1
    if region == "LN1" and "SCRATCH" in cons:
        return f"LayerNorm 1{pos_str}"
    # LayerNorm scratch saves (no inputs, just staging)
    if region == "SCRATCH" and not cons:
        return f"LN scratch{pos_str}"

    # Residual add 2: FFN output + saved RES → before LN2
    if region == "SCRATCH" and "RES" in cons:
        return f"Residual Add2{pos_str}"
    # FFN Output — second tile row (no RES, just FFN output)
    if region == "SCRATCH" and "X" in cons and "RES" not in cons and "SO_SCRATCH" not in cons:
        return f"FFN Out+Res Add2 t1{pos_str}"

    # FFN
    if region == "GELU" and "WEIGHT" in cons:
        return f"FFN Inter+GELU{pos_str}"
    if region == "SCRATCH" and "GELU" in cons and "WEIGHT" in cons:
        return f"FFN Output{pos_str}"

    # LayerNorm 2 → output
    if region == "OUT" and "SCRATCH" in cons:
        return f"LayerNorm 2 → Out{pos_str}"

    # Fallback
    name = _TENSOR_NAMES.get(region, region)
    verb = "Save" if is_store else "Load"
    return f"{verb} {name}{pos_str}"


def _is_weight_addr(addr: int, num_tiles: int, native_dim: int,
                    hidden_size: int) -> bool:
    """Check if a DRAM address is in the weight matrix area
    (after input data, before save regions at 0x200)."""
    input_end = hidden_size  # X data occupies 0..hidden_size-1
    save_start = 0x200
    return input_end <= addr < save_start


def extract_clusters(
    micro_ops: list[MicroOp],
    edges: list[tuple[int, int, tuple]],
    dim: int = 2,
    hidden_size: int = 4,
    seq_len: int = 2,
) -> list[TensorCluster]:
    """Partition the micro-op DAG into DRAM-flow clusters.

    Each cluster corresponds to one "tensor producer" — a DRAM_STORE
    plus all the compute and DRAM_LOAD ops that feed into it.
    Clusters are identified by scanning forward: each DRAM_STORE
    ends a cluster; the content of the cluster is all micro-ops
    since the previous DRAM_STORE (or start).

    Returns:
        List of TensorCluster objects in execution order.
    """
    num_tiles = hidden_size // dim

    # Identify DRAM_STORE positions as cluster boundaries
    store_indices = []
    for i, m in enumerate(micro_ops):
        if m.kind == "DRAM_STORE":
            store_indices.append(i)

    # Build clusters: each store ends a cluster; content spans from
    # the previous store (or start) up to and including this store.
    # Exclude nodes that are fully disconnected (no in AND no out edges)
    # — these are S_WR config writes and dead VREG_LD constants.
    total = len(micro_ops)
    has_in = [any(e[1] == i for e in edges) for i in range(total)]
    has_out = [any(e[0] == i for e in edges) for i in range(total)]
    def _is_connected(i: int) -> bool:
        return has_in[i] or has_out[i]

    clusters: list[TensorCluster] = []
    range_start = 0
    for store_idx in store_indices:
        # Collect member indices in [range_start, store_idx] inclusive,
        # but exclude fully disconnected nodes
        member_indices = [i for i in range(range_start, store_idx + 1)
                          if _is_connected(i)]

        # Extract DRAM load/stores and compute stats
        load_regions: set[str] = set()
        store_region = ""
        store_pos = -1
        load_bytes = 0
        store_bytes = 0
        flops = 0

        for mi in member_indices:
            m = micro_ops[mi]

            # Collect DRAM loads from ALL nodes (not just DRAM_LOAD,
            # since VV_BINOP, etc. may have absorbed DRAM reads)
            if m.kind == "DRAM_LOAD":
                for r in m.uses:
                    if r[0] == "DRAM":
                        region, pos = _classify_dram_addr(r[1], num_tiles, dim)
                        if region != "UNKNOWN":
                            load_regions.add(region)
                        load_bytes += dim * 2
            elif m.kind == "VV_BINOP":
                # VV_BINOP may have absorbed a V_RD_DRAM
                for r in m.uses:
                    if r[0] == "DRAM":
                        region, pos = _classify_dram_addr(r[1], num_tiles, dim)
                        if region != "UNKNOWN":
                            load_regions.add(region)
                        load_bytes += dim * 2
            elif m.kind == "V_UNARY":
                for r in m.uses:
                    if r[0] == "DRAM":
                        region, pos = _classify_dram_addr(r[1], num_tiles, dim)
                        if region != "UNKNOWN":
                            load_regions.add(region)
                        load_bytes += dim * 2

            # Collect MAT_LOAD (weight tile)
            if m.kind == "MAT_LOAD":
                load_regions.add("WEIGHT")
                for r in m.uses:
                    if r[0] == "DRAM":
                        load_bytes += dim * dim * 2  # full tile × fp16

            # Collect DRAM stores
            if m.kind == "DRAM_STORE":
                for r in m.defs:
                    if r[0] == "DRAM":
                        region, pos = _classify_dram_addr(r[1], num_tiles, dim)
                        if region != "UNKNOWN":
                            store_region = region
                            store_pos = pos
                            store_bytes += dim * 2

            # FLOPs estimation
            if m.kind == "MV_MUL":
                flops += 2 * dim * dim
            elif m.kind == "VV_BINOP":
                flops += 2 * dim
            elif m.kind == "SOFTMAX":
                flops += 5 * dim
            elif m.kind == "LAYERNORM":
                flops += 4 * dim
            elif m.kind in ("V_UNARY", "V_GELU"):
                flops += 1 * dim

        # Build label
        if store_region:
            label = _cluster_label(store_region, store_pos, is_store=True,
                                   consumed_regions=load_regions)
        else:
            label = f"Compute {range_start}-{store_idx}"

        cid = f"{store_region}[{store_pos}]" if store_region else f"c{len(clusters)}"

        tc = TensorCluster(
            id=cid,
            label=label,
            member_indices=member_indices,
            produced_region=store_region,
            produced_position=store_pos,
            consumed_regions=load_regions,
            dram_load_bytes=load_bytes,
            dram_store_bytes=store_bytes,
            compute_flops=flops,
            first_idx=range_start,
            last_idx=store_idx,
        )
        clusters.append(tc)
        range_start = store_idx + 1

    # Any remaining ops after last store
    if range_start < len(micro_ops):
        member_indices = [i for i in range(range_start, len(micro_ops))
                          if _is_connected(i)]
        if member_indices:
            clusters.append(TensorCluster(
            id="tail",
            label=f"Tail {range_start}-{len(micro_ops)-1}",
            member_indices=member_indices,
            produced_region="",
            produced_position=-1,
            consumed_regions=set(),
            first_idx=range_start,
            last_idx=len(micro_ops)-1,
        ))

    # Merge consecutive clusters that write the same (region, position)
    # and have no DRAM reads between them.
    if len(clusters) > 1:
        merged = [clusters[0]]
        for c in clusters[1:]:
            prev = merged[-1]
            same_producer = (prev.produced_region == c.produced_region and
                             prev.produced_position == c.produced_position and
                             prev.produced_region != "")
            # Check if there's a DRAM read between prev.member_indices[-1]
            # and c.member_indices[0] (looking at the raw micro-op range,
            # not just member indices since disconnected nodes live between)
            gap_start = max(prev.member_indices[-1] if prev.member_indices else 0, 0)
            gap_end = min(c.member_indices[0] if c.member_indices else 0, total)
            has_dram_read = False
            for mi in range(gap_start + 1, gap_end):
                m = micro_ops[mi]
                if _is_connected(mi) and m.kind in ("DRAM_LOAD", "MAT_LOAD") and m.uses:
                    for r in m.uses:
                        if r[0] == "DRAM":
                            has_dram_read = True
                            break
                    if has_dram_read:
                        break
            if same_producer and not has_dram_read:
                # Merge c into prev
                prev.member_indices.extend(c.member_indices)
                prev.member_indices.sort()
                prev.last_idx = c.last_idx
                prev.dram_load_bytes += c.dram_load_bytes
                prev.dram_store_bytes += c.dram_store_bytes
                prev.compute_flops += c.compute_flops
                prev.consumed_regions |= c.consumed_regions
            else:
                merged.append(c)
        clusters = merged

    # Re-label after merge — labels depend on full consumed_regions set.
    # Determine seq-pos from execution order: Q[1] Proj marks the boundary.
    seq_pos1_start = None
    for ci, c in enumerate(clusters):
        if c.produced_region == "Q" and c.produced_position == 1:
            seq_pos1_start = ci
            break

    for ci, c in enumerate(clusters):
        if c.produced_region:
            # K and V are pre-computed for ALL positions upfront — they
            # serve both p0 and p1.  Don't add a p0/p1 suffix.
            # K and V are pre-computed for ALL positions — no p0/p1 suffix.
            if c.produced_region in ("K", "V"):
                c.label = _cluster_label(
                    c.produced_region, -1,
                    is_store=True, consumed_regions=c.consumed_regions)
            else:
                # For Q, RES, OUT and transient regions, seq-pos is
                # determined by execution order (before/after Q[1] Proj).
                seq_pos = 0
                if seq_pos1_start is not None and ci >= seq_pos1_start:
                    seq_pos = 1
                c.label = _cluster_label(
                    c.produced_region, seq_pos, is_store=True,
                    consumed_regions=c.consumed_regions)

    return clusters


def clusters_to_text(clusters: list[TensorCluster]) -> str:
    """Render clusters as a human-readable table."""
    lines = []
    lines.append(f"{'#':>3} {'Label':30s} {'Members':>20s} {'LoadB':>6} {'StoreB':>7} {'FLOPs':>6} {'AI':>6}")
    lines.append("-" * 90)
    for i, c in enumerate(clusters):
        mem_str = f"{c.first_idx}-{c.last_idx}" if len(c.member_indices) > 1 else str(c.first_idx)
        ai = c.compute_flops / (c.dram_load_bytes + c.dram_store_bytes) if (c.dram_load_bytes + c.dram_store_bytes) > 0 else 0
        lines.append(
            f"{i:3d} {c.label:30s} {mem_str:>20s}"
            f" {c.dram_load_bytes:6d} {c.dram_store_bytes:7d}"
            f" {c.compute_flops:6d} {ai:6.1f}"
        )
        # Show consumed regions
        if c.consumed_regions:
            lines.append(f"     reads: {', '.join(sorted(c.consumed_regions))}")
        if c.produced_region:
            lines.append(f"     writes: {c.produced_region}[{c.produced_position}]")
    lines.append(f"\nTotal: {len(clusters)} clusters")
    return "\n".join(lines)


def clusters_to_dot(
    clusters: list[TensorCluster],
    micro_ops: list[MicroOp],
    edges: list[tuple[int, int, tuple]],
    path: Optional[str] = None,
) -> str:
    """Render clusters as a Graphviz DOT graph with subgraph clusters.

    Each TensorCluster becomes a graphviz subgraph with:
      - A summary header bar showing label, FLOPs, bytes, AI
      - All member micro-op nodes inside
    Inter-cluster DRAM edges are drawn between cluster boundaries.

    Args:
        clusters: list of TensorCluster objects.
        micro_ops: original micro-op list.
        edges: original edge list.
        path: if given, write DOT to this file.

    Returns:
        DOT string.
    """
    # Build micro-op index → cluster index map
    mop_to_cluster: dict[int, int] = {}
    for ci, c in enumerate(clusters):
        for mi in c.member_indices:
            mop_to_cluster[mi] = ci

    # Build cluster → cluster edges from DRAM edges that cross boundaries
    inter_cluster_edges: dict[tuple[int, int], list[str]] = {}
    for src, dst, res in edges:
        c_src = mop_to_cluster.get(src)
        c_dst = mop_to_cluster.get(dst)
        if c_src is not None and c_dst is not None and c_src != c_dst:
            key = (c_src, c_dst)
            label = _resource_str(res)
            if key not in inter_cluster_edges:
                inter_cluster_edges[key] = []
            if label not in inter_cluster_edges[key]:
                inter_cluster_edges[key].append(label)

    # Color palette for clusters
    COLORS = [
        "#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
        "#FCAE91", "#FB6A4A", "#EF3B2C", "#CB181D",
    ]

    lines = ["digraph dram_clusters {"]
    lines.append('  rankdir=TB;')
    lines.append('  compound=true;')
    lines.append('  node [shape=box style=filled fillcolor=white];')

    # ── Step 1: emit all micro-op nodes inside cluster subgraphs ──
    for ci, c in enumerate(clusters):
        color = COLORS[ci % len(COLORS)]
        ai_val = c.compute_flops / max(c.dram_load_bytes + c.dram_store_bytes, 1)
        lines.append('  subgraph cluster_' + str(ci) + ' {')
        lines.append('    label=<<B>' + c.label + '</B><BR/>'
                     'FLOPs=' + str(c.compute_flops) + '  '
                     'Load=' + str(c.dram_load_bytes) + 'B  '
                     'Store=' + str(c.dram_store_bytes) + 'B  '
                     'AI=' + f'{ai_val:.1f}' + '>;')
        lines.append('    labeljust=l;')
        lines.append('    style=filled;')
        lines.append('    fillcolor="' + color + '";')
        lines.append('    color="#888888";')

        # Emit ALL member micro-op nodes inside the cluster
        for mi in c.member_indices:
            m = micro_ops[mi]
            # Build a concise label
            label = str(mi) + ': ' + m.kind
            if m.kind in ("DRAM_LOAD", "DRAM_STORE", "MAT_LOAD") and m.detail:
                label += '\\n' + m.detail
            elif m.kind == "MV_MUL":
                pass  # plain label
            lines.append('    n' + str(mi) + ' [label="' + label + '"];')

        lines.append('  }')

    # ── Step 2: emit ALL edges, deduplicating multiple edges between same nodes ──
    edge_labels: dict[tuple[int, int, bool], list[str]] = {}
    for src, dst, res in edges:
        c_src = mop_to_cluster.get(src)
        c_dst = mop_to_cluster.get(dst)
        if c_src is None or c_dst is None:
            continue
        is_inter = c_src != c_dst
        key = (src, dst, is_inter)
        label = _resource_str(res)
        if key not in edge_labels:
            edge_labels[key] = []
        if label not in edge_labels[key]:
            edge_labels[key].append(label)

    for (src, dst, is_inter), labels in edge_labels.items():
        label = '\\n'.join(labels)
        if not is_inter:
            lines.append('  n' + str(src) + ' -> n' + str(dst) +
                         ' [label="' + label + '"];')
        else:
            lines.append('  n' + str(src) + ' -> n' + str(dst) +
                         ' [label="' + label +
                         '" ltail="cluster_' + str(mop_to_cluster.get(src)) +
                         '" lhead="cluster_' + str(mop_to_cluster.get(dst)) + '"];')

    lines.append('}')
    dot_str = "\n".join(lines) + "\n"

    if path is not None:
        with open(path, "w") as f:
            f.write(dot_str)

    return dot_str

# DRAM save regions that use per-position indexing vs transient scratch buffers.
_POSITIONAL_SAVE_REGIONS = {"Q", "K", "V", "RES", "OUT"}
_TRANSIENT_REGIONS = {"SCRATCH", "Z", "LN1", "GELU", "SO_SCRATCH"}
