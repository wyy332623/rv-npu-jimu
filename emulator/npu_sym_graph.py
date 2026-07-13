"""NPU — Symbolic Parametric Operator Graph.

Derives a parametric (symbolic) operator graph template from multiple
concrete OpGraph instances at different seq_len values.  The symbolic
graph can be instantiated for any (dim, hidden_size, seq_len).

Phase 3 of the computation-graph derivation pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

from emulator.npu_op_graph import OpGraph, OpNode


# ── Symbolic address ──────────────────────────────────────────────

@dataclass(frozen=True)
class SymAddr:
    """Symbolic DRAM tensor address.

    Equality is based on region + coord keys ONLY (base_expr is for
    human readability and does not participate in equality / hashing).
    """
    region: str                       # "Q", "K", "V", etc.
    coord: dict                       # {"pos": "pos"} or {}
    base_expr: str = ""               # e.g. "SAVE_K_BASE + pos * num_tiles * 8"

    def __eq__(self, other):
        if not isinstance(other, SymAddr):
            return NotImplemented
        return self.region == other.region and self.coord == other.coord

    def __hash__(self):
        return hash((self.region, tuple(sorted(self.coord.items()))))


# ── Symbolic operator node ────────────────────────────────────────

@dataclass
class SymOpNode:
    """One operator in the symbolic (parametric) graph."""
    name: str                         # "MVM_Q_FOR_pos" or "ATTN_SCORE"
    loop_vars: list[str]              # ["pos"] or []
    produced_tensors: list[SymAddr]
    consumed_tensors: list[SymAddr]
    instr_count_expr: str             # "num_tiles * 4 + 2" or "42"


# ── Symbolic operator graph ───────────────────────────────────────

@dataclass
class SymOpGraph:
    """Parametric operator graph template."""
    nodes: list[SymOpNode]
    edges: list[tuple[int, int, str]]  # (src_idx, dst_idx, region_name)


# ── Derivation ────────────────────────────────────────────────────

def derive_sym_graph(
    graphs: dict[int, OpGraph],
    dim: int,
    hidden_size: int,
) -> SymOpGraph:
    """Derive a symbolic graph from concrete graphs at different seq_lens.

    Algorithm:
    1. For each concrete graph, normalize node names by stripping any
       positional suffixes (e.g. "MVM_Q" stays "MVM_Q", "SAVE_K" stays
       "SAVE_K").
    2. Detect operators whose count scales with seq_len — these become
       loop_vars=["pos"].
    3. Operators with constant count across seq_len values become
       loop_vars=[].
    4. For each template operator, build SymAddr for produced/consumed
       tensors: if the address shifts by stride per position, mark coord
       with the loop variable; otherwise mark as constant.

    Args:
        graphs: dict mapping seq_len → concrete OpGraph (from Phase 2).
        dim: native dimension.
        hidden_size: model hidden size.

    Returns:
        SymOpGraph template.
    """
    if not graphs:
        return SymOpGraph(nodes=[], edges=[])

    seq_lens = sorted(graphs.keys())
    if len(seq_lens) < 2:
        # Single graph: treat everything as loop_vars=[]
        return _single_graph_template(graphs[seq_lens[0]])

    # Count nodes per normalized name at each seq_len
    name_counts: dict[str, dict[int, int]] = {}  # name → {seq_len: count}
    node_by_name: dict[str, list[tuple[int, OpNode]]] = {}

    for sl in seq_lens:
        g = graphs[sl]
        for node in g.nodes:
            norm = _normalize_name(node.name)
            name_counts.setdefault(norm, {})[sl] = name_counts.get(norm, {}).get(sl, 0) + 1
            node_by_name.setdefault(norm, []).append((sl, node))

    # Determine loop vars
    num_tiles = hidden_size // dim
    sym_nodes: list[SymOpNode] = []
    name_to_idx: dict[str, int] = {}

    for norm_name, counts_by_sl in name_counts.items():
        # Check if count scales linearly with seq_len
        counts = [counts_by_sl.get(sl, 0) for sl in seq_lens]
        is_positional = _scales_with_seq_len(counts, seq_lens)

        loop_vars = ["pos"] if is_positional else []

        # Choose representative node (from largest seq_len for most data)
        rep_nodes = [(sl, n) for sl, n in node_by_name[norm_name] if sl == seq_lens[-1]]
        if not rep_nodes:
            rep_nodes = node_by_name[norm_name]
        _, rep_node = rep_nodes[0]

        # Build SymAddrs
        prod_sym = _tensors_to_sym(rep_node.produced_tensors, loop_vars, num_tiles, dim)
        cons_sym = _tensors_to_sym(rep_node.consumed_tensors, loop_vars, num_tiles, dim)

        # Instruction count expression
        if is_positional and len(seq_lens) >= 2:
            # Try to derive formula: count = a * seq_len + b
            c0 = counts[0]
            c1 = counts[1]
            if c1 > c0 and (c1 - c0) % (seq_lens[1] - seq_lens[0]) == 0:
                a = (c1 - c0) // (seq_lens[1] - seq_lens[0])
                b = c0 - a * seq_lens[0]
                if b == 0:
                    instr_expr = f"{a} * seq_len"
                else:
                    instr_expr = f"{a} * seq_len + {b}"
            else:
                instr_expr = str(rep_node.instr_count)
        else:
            instr_expr = str(rep_node.instr_count)

        # Name with loop suffix
        if is_positional:
            sym_name = f"{norm_name}_FOR_{'_'.join(loop_vars)}"
        else:
            sym_name = norm_name

        sym_node = SymOpNode(
            name=sym_name,
            loop_vars=loop_vars,
            produced_tensors=prod_sym,
            consumed_tensors=cons_sym,
            instr_count_expr=instr_expr,
        )
        name_to_idx[norm_name] = len(sym_nodes)
        sym_nodes.append(sym_node)

    # Build edges — from the largest seq_len graph
    largest_g = graphs[seq_lens[-1]]
    sym_edges = _derive_edges(largest_g, name_to_idx)

    return SymOpGraph(nodes=sym_nodes, edges=sym_edges)


def _normalize_name(name: str) -> str:
    """Strip positional numeric suffixes from operator names.

    E.g. "MVM_Q" → "MVM_Q", "SAVE_K" → "SAVE_K"
    Names without numeric suffixes are unchanged.
    """
    # Our naming convention already avoids positional suffixes
    # (the position is encoded in the tensor, not the op name).
    # Just return as-is for now.
    return name


def _scales_with_seq_len(counts: list[int], seq_lens: list[int]) -> bool:
    """Check if node count scales roughly linearly with seq_len."""
    if len(counts) < 2:
        return False
    # At simplest: if count at highest seq_len > count at lowest
    # and the ratio is roughly proportional
    if counts[-1] > counts[0] and seq_lens[-1] > seq_lens[0]:
        # Check approximate proportionality
        if counts[0] == 0:
            # count went from 0 to >0 — treat as scaling
            return True
        ratio = counts[-1] / counts[0]
        sl_ratio = seq_lens[-1] / seq_lens[0]
        # Within 50% tolerance (generous for small counts)
        return 0.5 * sl_ratio < ratio < 2.0 * sl_ratio
    return False


def _tensors_to_sym(
    tensors: set[str],
    loop_vars: list[str],
    num_tiles: int,
    dim: int,
) -> list[SymAddr]:
    """Convert concrete tensor names to symbolic addresses."""
    result = []
    for t in sorted(tensors):
        if loop_vars and _is_positional_tensor(t):
            coord = {lv: lv for lv in loop_vars}
            base = f"{t}_BASE + {' + '.join(f'{lv} * num_tiles * {dim}' for lv in loop_vars)}"
        else:
            coord = {}
            base = f"{t}_BASE"
        result.append(SymAddr(region=t, coord=coord, base_expr=base))
    return result


def _is_positional_tensor(t: str) -> bool:
    """Check if a tensor name represents position-dependent data.

    Q, K, V are position-dependent (saved per position).
    SCRATCH, Z, LN1, GELU are reused across positions (constant).
    RES, OUT are position-dependent.
    """
    positional = {"Q", "K", "V", "RES", "OUT"}
    return t in positional


def _derive_edges(
    graph: OpGraph,
    name_to_idx: dict[str, int],
) -> list[tuple[int, int, str]]:
    """Derive symbolic edges from the largest concrete graph."""
    # Map concrete node indices to symbolic node indices
    node_to_sym: dict[int, int] = {}
    for i, node in enumerate(graph.nodes):
        norm = _normalize_name(node.name)
        if norm in name_to_idx:
            node_to_sym[i] = name_to_idx[norm]

    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()

    for src, dst, tensor in graph.edges:
        sym_src = node_to_sym.get(src)
        sym_dst = node_to_sym.get(dst)
        if sym_src is not None and sym_dst is not None:
            edge = (sym_src, sym_dst, tensor)
            if edge not in seen:
                edges.append(edge)
                seen.add(edge)

    return edges


def _single_graph_template(graph: OpGraph) -> SymOpGraph:
    """Create a trivial template from a single concrete graph."""
    sym_nodes: list[SymOpNode] = []
    for node in graph.nodes:
        prod_sym = [SymAddr(region=t, coord={}, base_expr=f"{t}_BASE")
                    for t in sorted(node.produced_tensors)]
        cons_sym = [SymAddr(region=t, coord={}, base_expr=f"{t}_BASE")
                    for t in sorted(node.consumed_tensors)]
        sym_nodes.append(SymOpNode(
            name=node.name,
            loop_vars=[],
            produced_tensors=prod_sym,
            consumed_tensors=cons_sym,
            instr_count_expr=str(node.instr_count),
        ))
    return SymOpGraph(nodes=sym_nodes, edges=list(graph.edges))


# ── Instantiation ─────────────────────────────────────────────────

def instantiate(
    sym_graph: SymOpGraph,
    dim: int,
    hidden_size: int,
    seq_len: int,
) -> OpGraph:
    """Instantiate a concrete OpGraph from a symbolic template.

    For each SymOpNode with loop_vars=["pos"], emit seq_len concrete
    OpNodes, one per position.  For non-parametric nodes, emit one.

    Args:
        sym_graph: Symbolic template from derive_sym_graph.
        dim: native dimension.
        hidden_size: model hidden size.
        seq_len: sequence length.

    Returns:
        Concrete OpGraph.
    """
    num_tiles = hidden_size // dim
    nodes: list[OpNode] = []
    # Track which concrete nodes came from which sym node
    sym_edges = sym_graph.edges

    for sym_node in sym_graph.nodes:
        if "pos" in sym_node.loop_vars:
            for pos in range(seq_len):
                concrete = _expand_node(sym_node, pos, num_tiles, dim)
                nodes.append(concrete)
        else:
            concrete = _expand_node(sym_node, None, num_tiles, dim)
            nodes.append(concrete)

    # Rebuild concrete edges from symbolic edges
    edges = _build_concrete_edges(nodes, sym_graph, seq_len)

    return OpGraph(nodes=nodes, edges=edges)


# ── Renderers ─────────────────────────────────────────────────────

def sym_graph_to_text(graph: SymOpGraph) -> str:
    """Render symbolic graph as human-readable text."""
    lines = []
    lines.append(f"  # {'Name':<28} {'Loops':<10} {'Prod':<14} {'Cons':<14} InstrExpr")
    lines.append("-" * 85)
    for i, n in enumerate(graph.nodes):
        prod = ",".join(sa.region for sa in n.produced_tensors) or "-"
        cons = ",".join(sa.region for sa in n.consumed_tensors) or "-"
        loops = ",".join(n.loop_vars) or "-"
        lines.append(
            f"  {i:<2} {n.name:<28} {loops:<10} {prod:<14} {cons:<14} {n.instr_count_expr}"
        )
        for src, dst, t in graph.edges:
            if dst == i:
                lines.append(f"    <- [{src} {graph.nodes[src].name}] via {t}")
    lines.append(f"\nTotal: {len(graph.nodes)} symbolic ops, {len(graph.edges)} edges")
    return "\n".join(lines)


def sym_graph_to_dot(
    graph: SymOpGraph,
    path: str | None = None,
) -> str:
    """Render symbolic graph as Graphviz DOT with parametric structure.

    Nodes with loop_vars are rendered as <TABLE> with a header row
    showing the parametric structure (e.g. "for pos in 0..seq_len-1").
    Constant (non-parametric) nodes are simple boxes.

    Edge labels show the tensor being passed and, for positional
    tensors, the coordinate expression (e.g. "K[pos]").

    Args:
        graph: Symbolic operator graph.
        path: If given, write DOT to this file.

    Returns:
        DOT source string.
    """
    lines = ["digraph sym_graph {",
             '  rankdir=LR;',
             '  node [shape=plaintext style=filled fillcolor=white];',
             '  edge [fontsize=10];',
             '  fontname="monospace";']

    for i, n in enumerate(graph.nodes):
        prod = ",".join(sa.region for sa in n.produced_tensors) or "\u2014"
        cons = ",".join(sa.region for sa in n.consumed_tensors) or "\u2014"
        instr = n.instr_count_expr
        # Clean up the name: remove _FOR_pos suffix for display
        display_name = n.name.replace('_FOR_pos', '')

        if n.loop_vars:
            color = "#FFF8DC"  # cornsilk for parametric
            lines.append(f'  n{i} [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2">'
                         f'<TR><TD COLSPAN="2"><B>{display_name}</B></TD></TR>'
                         f'<TR><TD COLSPAN="2"><FONT POINT-SIZE="9">'
                         f'x seq_len</FONT></TD></TR>'
                         f'<TR><TD>in: </TD><TD>{cons}</TD></TR>'
                         f'<TR><TD>out: </TD><TD>{prod}</TD></TR>'
                         f'<TR><TD COLSPAN="2"><FONT POINT-SIZE="9">'
                         f'instr = {instr}</FONT></TD></TR>'
                         f'</TABLE>>];')
        else:
            color = "#E0F0FF"  # light blue for constant
            lines.append(f'  n{i} [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2">'
                         f'<TR><TD COLSPAN="2"><B>{display_name}</B></TD></TR>'
                         f'<TR><TD COLSPAN="2"><FONT POINT-SIZE="9">constant</FONT></TD></TR>'
                         f'<TR><TD>in: </TD><TD>{cons}</TD></TR>'
                         f'<TR><TD>out: </TD><TD>{prod}</TD></TR>'
                         f'<TR><TD COLSPAN="2"><FONT POINT-SIZE="9">'
                         f'instr = {instr}</FONT></TD></TR>'
                         f'</TABLE>>];')

    for src, dst, tensor in graph.edges:
        lines.append(f'  n{src} -> n{dst} [label="{tensor}"];')

    lines.append("}")
    dot = "\n".join(lines)

    if path:
        with open(path, "w") as f:
            f.write(dot)

    return dot


def _expand_node(
    sym: SymOpNode,
    pos: Optional[int],
    num_tiles: int,
    dim: int,
) -> OpNode:
    """Expand one symbolic node into a concrete node."""
    prod = set()
    cons = set()
    for sa in sym.produced_tensors:
        prod.add(sa.region)
    for sa in sym.consumed_tensors:
        cons.add(sa.region)

    name = sym.name
    if pos is not None:
        name = f"{sym.name.replace('_FOR_pos', '')}_{pos}"

    # Estimate instruction count
    try:
        count = eval(sym.instr_count_expr,
                      {"seq_len": (pos + 1 if pos is not None else 1),
                       "num_tiles": num_tiles})
    except Exception:
        count = 0

    return OpNode(
        batch_idx=0,
        name=name,
        produced_tensors=prod,
        consumed_tensors=cons,
        opcode_counts={},
        instr_count=max(1, int(count)),
        first_idx=0,
        last_idx=0,
    )


def _build_concrete_edges(
    nodes: list[OpNode],
    sym_graph: SymOpGraph,
    seq_len: int,
) -> list[tuple[int, int, str]]:
    """Build concrete edges from the symbolic edge list."""
    # Map symbolic node index → list of concrete node indices
    sym_to_concrete: dict[int, list[int]] = {}
    concrete_idx = 0
    for si, sn in enumerate(sym_graph.nodes):
        if "pos" in sn.loop_vars:
            sym_to_concrete[si] = list(range(concrete_idx, concrete_idx + seq_len))
            concrete_idx += seq_len
        else:
            sym_to_concrete[si] = [concrete_idx]
            concrete_idx += 1

    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()

    for sym_src, sym_dst, tensor in sym_graph.edges:
        for c_src in sym_to_concrete.get(sym_src, []):
            for c_dst in sym_to_concrete.get(sym_dst, []):
                # Only connect if source precedes destination
                if c_src < c_dst:
                    edge = (c_src, c_dst, tensor)
                    if edge not in seen:
                        edges.append(edge)
                        seen.add(edge)

    return edges
