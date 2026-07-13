"""NPU — DAG Builder and Renderers for the Flat Event Trace.

Builds a concrete def-use DAG from flat events emitted by EventTracer.
Each event is a node; edges connect uses to their reaching definitions.

Provides text and Graphviz DOT renderers.
"""

from typing import List, Tuple, Optional

# Re-export opcode names for downstream convenience
from emulator.npu_event_trace import OPCODE_NAMES  # noqa: F401


def build_dag(
    events: list[dict],
) -> Tuple[list[dict], list[Tuple[int, int, tuple]]]:
    """Build a concrete def-use DAG from flat events.

    Forward-pass reaching-definition chaining: each use connects to
    the most recent def of the same resource.  Deduplicates edges
    that would appear multiple times when an event uses the same
    resource via multiple uses.

    Args:
        events: list of event dicts with keys idx, op, raw, defs, uses.

    Returns:
        (nodes, edges) where nodes == events and edges are
        (src_idx, dst_idx, resource) tuples.
    """
    last_def: dict[tuple, int] = {}  # resource → event_idx
    edges: list[tuple[int, int, tuple]] = []
    seen_edges: set[tuple[int, int, tuple]] = set()

    for ev in events:
        for use in ev["uses"]:
            if use in last_def:
                edge = (last_def[use], ev["idx"], use)
                if edge not in seen_edges:
                    edges.append(edge)
                    seen_edges.add(edge)
        for df in ev["defs"]:
            last_def[df] = ev["idx"]

    return events, edges


def dag_to_text(nodes: list[dict],
                edges: list[Tuple[int, int, tuple]]) -> str:
    """Render DAG as human-readable text.

    One line per node, followed by indented predecessor list.
    """
    # Build predecessor map: dst → [(src, resource), ...]
    preds: dict[int, list[tuple[int, tuple]]] = {}
    for src, dst, res in edges:
        preds.setdefault(dst, []).append((src, res))

    lines = []
    for node in nodes:
        idx = node["idx"]
        op = node["op"]
        defs = node["defs"]
        uses = node["uses"]
        lines.append(
            f"{idx:3d} {op:20s} defs={defs} uses={uses}"
        )
        for src_idx, res in sorted(preds.get(idx, [])):
            src_op = nodes[src_idx]["op"] if src_idx < len(nodes) else "?"
            # Format resource as a readable string
            res_str = _resource_str(res)
            lines.append(
                f"    <- [{src_idx} {src_op}] via {res_str}"
            )

    return "\n".join(lines)


def _resource_str(res: tuple) -> str:
    """Format a resource tuple as a readable string."""
    if res[0] == "DRAM":
        return f"DRAM[{res[1]:#x}]"
    elif res[0] == "VRF":
        return f"VRF[{res[1]}][{res[2]}]"
    elif res[0] == "REG":
        return f"REG[{res[1]}]"
    elif res[0] == "SRF":
        return f"SRF[{res[1]}]"
    elif res[0] == "pipe":
        return "pipe"
    elif res[0] == "vpipe_a":
        return "vpipe_a"
    elif res[0] == "MRF":
        return "MRF"
    else:
        return str(res)


def dag_to_dot(nodes: list[dict],
               edges: list[Tuple[int, int, tuple]],
               path: Optional[str] = None) -> str:
    """Render DAG as Graphviz DOT.

    Node labels include index and opcode.  Edge labels show the
    shared resource name.

    Args:
        nodes: event list (each has idx, op, ...).
        edges: list of (src, dst, resource) tuples.
        path: if given, write DOT to this file.

    Returns:
        DOT string.
    """
    # Deduplicate edges for DOT: same (src, dst) may appear with
    # multiple resources — combine labels.
    edge_labels: dict[tuple[int, int], list[str]] = {}
    for src, dst, res in edges:
        key = (src, dst)
        label = _resource_str(res)
        if key not in edge_labels:
            edge_labels[key] = []
        if label not in edge_labels[key]:
            edge_labels[key].append(label)

    lines = ['digraph dag {']
    lines.append('  node [shape=box];')
    lines.append('  rankdir=TB;')

    for node in nodes:
        idx = node["idx"]
        op = node["op"]
        label = f"{idx}: {op}".replace('"', '\\"')
        lines.append(f'  n{idx} [label="{label}"];')

    for (src, dst), labels in sorted(edge_labels.items()):
        label = ", ".join(labels).replace('"', '\\"')
        lines.append(f'  n{src} -> n{dst} [label="{label}"];')

    lines.append('}')
    dot_str = "\n".join(lines) + "\n"

    if path is not None:
        with open(path, "w") as f:
            f.write(dot_str)

    return dot_str


# ── Convenience: trace firmware and build operator graph ────────────

def trace_and_build_op_graph(elf_path: str, dim: int, hidden_size: int,
                               seq_len: int = 1) -> 'OpGraph':
    """Run emulator on ELF, capture events, build operator graph.

    This convenience function:
      1. Creates NpuDeviceMini with the given dim
      2. Creates both TraceRecorder and EventTracer
      3. Runs the ISS with the ELF
      4. Extracts batch sizes from TraceRecorder
      5. Calls build_op_graph with events and batch sizes

    Requires:
      - libnpukernels.so in _build/kernels/ (for NpuDeviceMini)
      - pyelftools installed (for MiniRV64)

    Args:
        elf_path: path to firmware ELF
        dim: native_dim (NATIVE_DIM)
        hidden_size: model hidden_size
        seq_len: sequence length

    Returns:
        OpGraph
    """
    from emulator.npu_device_mini import NpuDeviceMini
    from emulator.npu_event_trace import EventTracer
    from emulator.trace_recorder import TraceRecorder
    from emulator.npu_op_graph import build_op_graph
    from iss.mini_rv64 import MiniRV64

    npu = NpuDeviceMini(native_dim=dim)
    npu.set_hidden_size(hidden_size)
    npu.set_seq_len(seq_len)

    # EventTracer must wrap the NPU device directly so it can patch
    # _push_instruction and _execute.  TraceRecorder wraps the NPU
    # via MMIO delegation — it sees store()/load() but NOT the
    # internal _execute method.
    tracer = EventTracer(npu)
    rec = TraceRecorder(npu)       # records MMIO for batch extraction

    cpu = MiniRV64()
    cpu.set_mmio_device(rec)
    cpu.load_elf(elf_path)
    cpu.run(cycles=50000)

    batch_sizes = [len(b) for b in rec.extract_batches()]
    graph = build_op_graph(tracer.events, dim=dim, hidden_size=hidden_size,
                           seq_len=seq_len, batch_sizes=batch_sizes)

    tracer.unpatch()
    return graph
