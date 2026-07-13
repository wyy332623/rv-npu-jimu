"""NPU — Operator-Level Graph with DRAM-Region Grouping.

Collapses the flat instruction-level event trace into an operator-level
graph.  Each node (OpNode) represents one logical operator on a full
tensor (e.g. MVM_Q, ATTN_SCORE, LN1); edges are tensor-level data
dependencies derived from DRAM region classification.

Phase 2 of the computation-graph derivation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

# ── DRAM region layout (from firmware/bert/bert_layer.c) ─────────────
# Each entry: (base_addr, region_name, max_span)
# max_span is generous — actual region size is seq_len * num_tiles * NATIVE_DIM
DRAM_REGIONS: list[tuple[int, str, int]] = [
    (0x200, "Q",         0x100),   # SAVE_Q_BASE, 256 elements span
    (0x300, "K",         0x100),   # SAVE_K_BASE
    (0x400, "V",         0x100),   # SAVE_V_BASE
    (0x500, "SCRATCH",   0x080),   # SCRATCH_ADDR, 128 elements span
    (0x580, "SO_SCRATCH",0x080),   # SO_SCRATCH, 128 elements
    (0x600, "Z",         0x020),   # SCRATCH_Z, 32 elements
    (0x620, "LN1",       0x020),   # SCRATCH_LN1, 32 elements
    (0x640, "GELU",      0x020),   # SCRATCH_GELU, 32 elements
    (0x700, "RES",       0x100),   # SAVE_RES_BASE, 256 elements
    (0x800, "OUT",       0x100),   # SAVE_OUT_BASE, 256 elements
    (0x900, "UNIT_VEC",  0x080),   # UNIT_VEC_BASE, 128 elements
]


def classify_addr(addr: int, num_tiles: int,
                  native_dim: int = 8) -> tuple[str, int]:
    """Classify a DRAM address into (region_name, position).

    Position is computed as (addr - base) // (num_tiles * native_dim).
    Addresses not falling in any known region return ("UNKNOWN", addr).

    Upper bound for each region is the base of the next region, ensuring
    no overlaps regardless of span values.
    """
    stride = num_tiles * native_dim
    # Build upper bounds from next region base (or a generous sentinel)
    for i, (base, name, _span) in enumerate(reversed(DRAM_REGIONS)):
        rev_idx = len(DRAM_REGIONS) - 1 - i
        if addr >= base:
            # Upper bound = base of next-lower region (already checked)
            # or for the lowest region (Q at 0x200), any addr < next region base.
            # Since we iterate reversed, higher bases are checked first.
            # The upper bound is simply the base of the preceding entry in
            # the reversed list (i.e., the next-higher base).
            if i > 0:
                upper = list(reversed(DRAM_REGIONS))[i - 1][0]
            else:
                # Highest region (UNIT_VEC at 0x900): generous upper = base + span*8
                upper = base + DRAM_REGIONS[-1][2] * native_dim
            if addr < upper:
                pos = (addr - base) // stride if stride > 0 else 0
                return (name, pos)
    return ("UNKNOWN", addr)


# ── Data structures ────────────────────────────────────────────────

@dataclass
class OpNode:
    """One logical operator in the operator-level graph."""
    batch_idx: int           # which npu_wait_done() batch
    name: str                # e.g. "MVM_Q", "ATTN_SCORE"
    produced_tensors: set    # set of str like {"Q"} or {"Z", "LN1"}
    consumed_tensors: set    # set of str like {"SCRATCH", "RES"}
    opcode_counts: dict      # opcode_name → count within this batch
    instr_count: int         # total number of instructions in this batch
    first_idx: int           # first event index in the batch
    last_idx: int            # last event index in the batch


@dataclass
class OpGraph:
    """Operator-level computation graph."""
    nodes: list[OpNode]
    edges: list[tuple[int, int, str]]  # (src_node_idx, dst_node_idx, tensor)


# ── Opcode classification helpers ──────────────────────────────────

# Opcodes that read from DRAM (via full_operand LO-format)
_DRAM_READ_OPS = frozenset({
    "V_RD_DRAM", "M_RD_DRAM",
    "V_RD_DRAM_INC",
})

# Opcodes that write to DRAM
_DRAM_WRITE_OPS = frozenset({
    "V_WR_DRAM", "M_WR_DRAM",
    "V_WR_DRAM_INC",
})

# Opcodes that involve matrix-vector multiplication
_MVM_OPS = frozenset({
    "MV_MUL", "MV_MUL_INC",
})

# Opcodes that are activation functions
_ACTIVATION_OPS = frozenset({
    "V_SIGM", "V_TANH", "V_RELU", "V_GELU", "V_EXP",
})

# Opcodes that are elementwise binary ops
_BIN_OPS = frozenset({
    "VV_ADD", "VV_A_SUB_B", "VV_B_SUB_A", "VV_MUL", "VV_MIN", "VV_MAX",
    "VV_ADD_INC", "VV_MAX_INC", "VV_MUL_INC",
    "VV_A_SUB_B_INC", "VV_B_SUB_A_INC",
})

# SPU ops
_SPU_OPS = frozenset({
    "S_RECIP", "S_SQRT", "SS_MUL", "SS_ADD",
})

# V_FUNC sub-ops
_SOFTMAX_OPS = frozenset({"V_FUNC/SOFTMAX"})
_LAYERNORM_OPS = frozenset({"V_FUNC/LAYERNORM"})


def _classify_opcodes(opcode_counts: dict) -> str:
    """Derive operator name from opcode mix within a batch.

    Naming heuristic based on the scope doc:
    - MV_MUL present → MVM_<first_produced_tensor>
    - VV_MUL + softmax → ATTN_SCORE
    - VV_MUL (second per position) → ATTN_CONTEXT
    - V_FUNC/LAYERNORM → LN1 or LN2 (resolved by caller using position context)
    - V_GELU → FFN_GELU
    - VV_ADD + no MV_MUL → RESIDUAL_ADD
    - V_RD_DRAM and V_WR_DRAM only → SAVE_<tensor> or LOAD_<tensor>
    - Else → OP_<batch_idx>  (caller replaces batch_idx)
    """
    has_mvm = any(o in opcode_counts for o in _MVM_OPS)
    has_vv_mul = "VV_MUL" in opcode_counts
    has_softmax = any(o in opcode_counts for o in _SOFTMAX_OPS)
    has_layernorm = any(o in opcode_counts for o in _LAYERNORM_OPS)
    has_gelu = "V_GELU" in opcode_counts
    has_vv_add = "VV_ADD" in opcode_counts
    has_vv_add_inc = "VV_ADD_INC" in opcode_counts

    if has_softmax and has_vv_mul:
        return "ATTN_SCORE"
    if has_layernorm:
        return "LN"
    if has_gelu:
        return "FFN_GELU"
    if has_mvm:
        return "MVM"
    if has_vv_mul:
        return "ATTN_CONTEXT"
    if (has_vv_add or has_vv_add_inc) and not has_mvm:
        return "RESIDUAL_ADD"

    # Pure save/load ops
    dram_reads = sum(opcode_counts.get(o, 0) for o in _DRAM_READ_OPS)
    dram_writes = sum(opcode_counts.get(o, 0) for o in _DRAM_WRITE_OPS)
    total_compute = sum(v for k, v in opcode_counts.items()
                        if k not in _DRAM_READ_OPS and k not in _DRAM_WRITE_OPS
                        and k not in ("S_WR", "S_RD", "INST_ISSUE",
                                      "V_RD", "V_WR", "V_WR_INC", "V_RD_INC",
                                      "M_RD", "M_WR",
                                      "VREG_MOVE", "VREG_LD"))
    if total_compute == 0:
        # Pure data-movement: reads and/or writes without any real compute
        if dram_writes > 0 and dram_reads == 0:
            return "SAVE"
        if dram_reads > 0 and dram_writes == 0:
            return "LOAD"
        if dram_reads > 0 and dram_writes > 0:
            # Read-modify-write: reads from one region and writes to another
            # (e.g. SO_SCRATCH load-add-store for residual)
            return "SAVE"  # name after the produced tensor

    return "OP"


# ── Batch-splitting ────────────────────────────────────────────────

def split_events_into_batches(
    events: list[dict],
    batch_sizes: Optional[list[int]] = None,
) -> list[list[dict]]:
    """Split events into batches.

    If batch_sizes is provided (from TraceRecorder.extract_batches()),
    events are split by instruction count.  Otherwise, each event is
    its own batch (no batch info available — caller should provide
    batch_sizes for meaningful results).

    Args:
        events: flat event list from EventTracer.
        batch_sizes: list of instruction counts per batch.

    Returns:
        List of event sub-lists, one per batch.
    """
    if not events:
        return []

    if batch_sizes is None:
        # Without batch info, each event is its own "batch"
        return [[ev] for ev in events]

    batches = []
    offset = 0
    for size in batch_sizes:
        batch = events[offset:offset + size]
        batches.append(batch)
        offset += size

    # Any remaining events go into a trailing batch
    if offset < len(events):
        batches.append(events[offset:])

    return batches


# ── Build operator graph ──────────────────────────────────────────

def _split_batch_into_sub_ops(
    batch: list[dict],
    num_tiles: int,
    native_dim: int,
) -> list[list[dict]]:
    """Split a single batch into sub-operators using DRAM-write boundaries.

    When firmware sends all instructions in one npu_wait_done() batch,
    we still want per-operator granularity.  The heuristic: each time
    the event trace writes to DRAM and the written region changes
    (or a new region appears), we close the current sub-op and start
    a new one.

    Consecutive DRAM writes to the SAME region (e.g. tile columns of
    a matrix) are kept together in one sub-op — the operator is
    finished when we see a write to a DIFFERENT region.

    Non-DRAM events (VRF ops, pipeline arithmetic) are assigned to
    the current sub-op.
    """
    if not batch:
        return []

    sub_ops: list[list[dict]] = []
    current: list[dict] = [batch[0]]
    current_written_regions: set[str] = set()

    for ev in batch[1:]:
        # Find DRAM regions defined by this event
        ev_written_regions: set[str] = set()
        for df in ev["defs"]:
            if df[0] == "DRAM":
                region, _ = classify_addr(df[1], num_tiles, native_dim)
                if region != "UNKNOWN":
                    ev_written_regions.add(region)

        # If this event writes to a region that is NOT in the current
        # sub-op's written regions AND the current sub-op already
        # has written some region, start a new sub-op.
        new_region = ev_written_regions - current_written_regions
        if new_region and current_written_regions:
            sub_ops.append(current)
            current = [ev]
            current_written_regions = ev_written_regions
        else:
            current.append(ev)
            current_written_regions |= ev_written_regions

    if current:
        sub_ops.append(current)

    return sub_ops


def build_op_graph(
    events: list[dict],
    dim: int,
    hidden_size: int,
    seq_len: int = 1,
    batch_sizes: Optional[list[int]] = None,
) -> OpGraph:
    """Build operator-level graph from flat event trace.

    If a batch is very large (typical when firmware sends all ops in
    one npu_wait_done()), the batch is automatically sub-divided
    using DRAM-write boundaries to recover per-operator granularity.

    Args:
        events: flat event list from EventTracer.
        dim: native_dim (NATIVE_DIM) of the NPU.
        hidden_size: hidden_size of the model.
        seq_len: sequence length (number of positions).
        batch_sizes: instruction counts per batch (from
            TraceRecorder.extract_batches()). If None, each event
            forms its own batch — not very useful but valid.

    Returns:
        OpGraph with nodes and edges.
    """
    num_tiles = hidden_size // dim
    native_dim = dim

    mega_batches = split_events_into_batches(events, batch_sizes)

    # Sub-divide each mega-batch into per-operator sub-batches
    batches: list[list[dict]] = []
    for mb in mega_batches:
        subs = _split_batch_into_sub_ops(mb, num_tiles, native_dim)
        if subs:
            batches.extend(subs)

    nodes: list[OpNode] = []
    # Track context for disambiguation
    ln_count = 0  # how many LN ops we've emitted per position
    vv_mul_count = 0  # total VV_MUL ops across all batches

    for batch_idx, batch in enumerate(batches):
        produced_tensors: set[str] = set()
        consumed_tensors: set[str] = set()
        opcode_counts: dict[str, int] = {}
        first_idx = batch[0]["idx"] if batch else 0
        last_idx = batch[-1]["idx"] if batch else 0

        for ev in batch:
            op = ev["op"]
            opcode_counts[op] = opcode_counts.get(op, 0) + 1

            # Classify DRAM defs
            for df in ev["defs"]:
                if df[0] == "DRAM":
                    region, pos = classify_addr(df[1], num_tiles, native_dim)
                    if region != "UNKNOWN":
                        produced_tensors.add(region)

            # Classify DRAM uses
            for us in ev["uses"]:
                if us[0] == "DRAM":
                    region, pos = classify_addr(us[1], num_tiles, native_dim)
                    if region != "UNKNOWN":
                        consumed_tensors.add(region)

        # Derive base name from opcode mix
        base_name = _classify_opcodes(opcode_counts)

        # Disambiguate
        name = _disambiguate_name(
            base_name, opcode_counts, produced_tensors, consumed_tensors,
            batch_idx, ln_count, vv_mul_count, num_tiles, seq_len,
        )

        # Update context counters
        vv_mul_count += opcode_counts.get("VV_MUL", 0)
        if base_name == "LN":
            ln_count += 1

        node = OpNode(
            batch_idx=batch_idx,
            name=name,
            produced_tensors=produced_tensors,
            consumed_tensors=consumed_tensors,
            opcode_counts=opcode_counts,
            instr_count=len(batch),
            first_idx=first_idx,
            last_idx=last_idx,
        )
        nodes.append(node)

    # Build edges
    edges = _build_edges(nodes)
    return OpGraph(nodes=nodes, edges=edges)


def _disambiguate_name(
    base_name: str,
    opcode_counts: dict,
    produced: set[str],
    consumed: set[str],
    batch_idx: int,
    ln_count: int,
    vv_mul_count: int,
    num_tiles: int,
    seq_len: int,
) -> str:
    """Refine a base operator name with context information."""
    if base_name == "MVM":
        # Name after the primary produced tensor
        tensor = _first_produced_tensor(produced)
        return f"MVM_{tensor}" if tensor else f"MVM_{batch_idx}"

    if base_name == "LN":
        # Alternate between LN1 and LN2 based on ln_count
        suffix = "1" if ln_count % 2 == 0 else "2"
        tensor = _first_produced_tensor(produced)
        return f"LN{suffix}" if not tensor else f"LN{suffix}"

    if base_name == "ATTN_SCORE":
        return "ATTN_SCORE"

    if base_name == "ATTN_CONTEXT":
        return "ATTN_CONTEXT"

    if base_name == "FFN_GELU":
        return "FFN_GELU"

    if base_name == "RESIDUAL_ADD":
        return "RESIDUAL_ADD"

    if base_name == "SAVE":
        tensor = _first_produced_tensor(produced)
        return f"SAVE_{tensor}" if tensor else f"SAVE_{batch_idx}"

    if base_name == "LOAD":
        tensor = _first_consumed_tensor(consumed)
        return f"LOAD_{tensor}" if tensor else f"LOAD_{batch_idx}"

    return f"OP_{batch_idx}"


def _first_produced_tensor(tensors: set[str]) -> str:
    """Return the first non-scratch produced tensor, preferring Q/K/V/OUT."""
    priority = ["Q", "K", "V", "Z", "LN1", "GELU", "RES", "OUT",
                "SCRATCH", "SO_SCRATCH", "UNIT_VEC"]
    for p in priority:
        if p in tensors:
            return p
    # Return any available
    return next(iter(tensors)) if tensors else ""


def _first_consumed_tensor(tensors: set[str]) -> str:
    """Return the first non-scratch consumed tensor."""
    priority = ["Q", "K", "V", "Z", "LN1", "GELU", "RES", "OUT",
                "SCRATCH", "SO_SCRATCH", "UNIT_VEC"]
    for p in priority:
        if p in tensors:
            return p
    return next(iter(tensors)) if tensors else ""


def _build_edges(nodes: list[OpNode]) -> list[tuple[int, int, str]]:
    """Build tensor-level edges between operator nodes.

    For each tensor, connect producers to the nearest preceding consumer.
    Multiple producers of the same tensor (e.g., SCRATCH reuse) —
    connect consumer to the **nearest preceding** producer.
    """
    # last_producer[tensor] → node index of most recent producer
    last_producer: dict[str, int] = {}
    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()

    for dst_idx, node in enumerate(nodes):
        for tensor in node.consumed_tensors:
            if tensor in last_producer:
                src_idx = last_producer[tensor]
                edge = (src_idx, dst_idx, tensor)
                if edge not in seen:
                    edges.append(edge)
                    seen.add(edge)
        for tensor in node.produced_tensors:
            last_producer[tensor] = dst_idx

    return edges


# ── Renderers ────────────────────────────────────────────────────

def op_graph_to_text(graph: OpGraph) -> str:
    """Render operator graph as human-readable text table."""
    # Build predecessor map
    preds: dict[int, list[tuple[int, str]]] = {}
    for src, dst, tensor in graph.edges:
        preds.setdefault(dst, []).append((src, tensor))

    lines = []
    lines.append(f"{'#':>3} {'Name':20s} {'Prod':16s} {'Cons':16s} {'Instr':>5} {'Ids':>9}")
    lines.append("-" * 75)

    for i, node in enumerate(graph.nodes):
        prod_str = ",".join(sorted(node.produced_tensors)) if node.produced_tensors else "-"
        cons_str = ",".join(sorted(node.consumed_tensors)) if node.consumed_tensors else "-"
        ids_str = f"{node.first_idx}-{node.last_idx}"
        lines.append(
            f"{i:3d} {node.name:20s} {prod_str:16s} {cons_str:16s}"
            f" {node.instr_count:5d} {ids_str:>9}"
        )
        for src_idx, tensor in sorted(preds.get(i, [])):
            src_name = graph.nodes[src_idx].name if src_idx < len(graph.nodes) else "?"
            lines.append(f"    <- [{src_idx} {src_name}] via {tensor}")

    lines.append(f"\nTotal: {len(graph.nodes)} operators, {len(graph.edges)} edges")
    return "\n".join(lines)


def op_graph_to_dot(graph: OpGraph,
                    path: Optional[str] = None) -> str:
    """Render operator graph as Graphviz DOT.

    Args:
        graph: OpGraph to render.
        path: if given, write DOT to this file.

    Returns:
        DOT string.
    """
    # Deduplicate edges: same (src, dst) may have multiple tensors
    edge_labels: dict[tuple[int, int], list[str]] = {}
    for src, dst, tensor in graph.edges:
        key = (src, dst)
        if key not in edge_labels:
            edge_labels[key] = []
        if tensor not in edge_labels[key]:
            edge_labels[key].append(tensor)

    lines = ['digraph op_graph {']
    lines.append('  node [shape=box, style=filled, fillcolor=lightyellow];')
    lines.append('  rankdir=TB;')

    for i, node in enumerate(graph.nodes):
        label = f"{node.name}".replace('"', '\\"')
        prod = ",".join(sorted(node.produced_tensors))
        if prod:
            label += f"\\n→ {prod}"
        lines.append(f'  n{i} [label="{label}"];')

    for (src, dst), labels in sorted(edge_labels.items()):
        label = ", ".join(sorted(labels)).replace('"', '\\"')
        lines.append(f'  n{src} -> n{dst} [label="{label}"];')

    lines.append('}')
    dot_str = "\n".join(lines) + "\n"

    if path is not None:
        with open(path, "w") as f:
            f.write(dot_str)

    return dot_str
