#!/usr/bin/env python3
"""Generate graph visualizations from BERT firmware running on the NPU emulator.

Usage:
    python jimu-dse/scripts/visualize_graph.py                    # all graphs, default params
    python jimu-dse/scripts/visualize_graph.py --phase op          # operator graph only
    python jimu-dse/scripts/visualize_graph.py --phase dag          # instruction DAG only
    python jimu-dse/scripts/visualize_graph.py --phase sym          # symbolic graph (runs 2 seq_lens)
    python jimu-dse/scripts/visualize_graph.py --dim 4 --seq-len 2 --hidden 8
    python jimu-dse/scripts/visualize_graph.py -o _out             # custom output dir
    python jimu-dse/scripts/visualize_graph.py --no-render          # generate .dot only, skip graphviz

Requires (auto-skipped if missing):
    - _build/kernels/libnpukernels.so   (C kernel library for emulator)
    - pyelftools                        (for MiniRV64 ISS; pip install pyelftools)
    - graphviz `dot` binary             (for PNG/SVG rendering; --no-render skips this)

Firmware is automatically rebuilt with matching DRAM layout before emulation.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ── Repo-root-relative imports ─────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def check_deps(need_sym: bool = False) -> list[str]:
    """Return list of missing dependency descriptions; empty = all good."""
    missing = []
    lib_paths = [
        REPO_ROOT / "_build/kernels/libnpukernels.so",
    ]
    if not any(p.exists() for p in lib_paths):
        missing.append("_build/kernels/libnpukernels.so  (run: make kernels)")
    try:
        from iss.mini_rv64 import MiniRV64  # noqa: F401
    except ImportError:
        missing.append("pyelftools  (pip install pyelftools)")
    return missing


def build_firmware(dim, hidden_size, seq_len, num_head) -> str:
    """Build firmware ELF with matching DRAM layout macros.

    Returns path to the built ELF file.
    """
    import os

    num_tiles = hidden_size // dim
    proj_base = hidden_size * seq_len + 4
    mat_size = hidden_size * hidden_size
    stride = mat_size + hidden_size
    proj_end = proj_base + 6 * stride
    ln_base = proj_end
    ln_size = num_tiles * 8
    scratch_addr = 0x500

    build_dir = f"build_dim{dim}"
    env = {
        'NATIVE_DIM': str(dim),
        'SEQ_LEN': str(seq_len),
        '_HIDDEN_SIZE': str(hidden_size),
        '_PROJ_BASE': str(proj_base),
        '_MAT_SIZE': str(mat_size),
        '_STRIDE': str(stride),
        '_NUM_TILES': str(num_tiles),
        '_LN1_GAMMA': str(ln_base),
        '_LN1_BETA': str(ln_base + ln_size),
        '_LN2_GAMMA': str(ln_base + 2 * ln_size),
        '_LN2_BETA': str(ln_base + 3 * ln_size),
        '_SCRATCH': str(scratch_addr),
        'NUM_HEAD': str(num_head),
    }
    full_env = {**os.environ, **env}
    fw_root = str(REPO_ROOT / "firmware")
    r = subprocess.run(
        ['make', '-C', fw_root, f'BUILD_DIR={build_dir}', 'clean', 'all'],
        capture_output=True, text=True, env=full_env)
    if r.returncode != 0:
        print(f"  ✗  Firmware build failed (RC={r.returncode})")
        print(r.stderr[-300:] if r.stderr else r.stdout[-300:])
        sys.exit(1)
    elf = str(REPO_ROOT / "firmware" / build_dir / "bert.elf")
    print(f"  ✓  Firmware built: {elf}")
    return elf


def load_weights(npu, params, dim, hidden_size):
    """Load BERT weights and inputs into NPU DRAM (mirrors test_bert_e2e logic)."""
    import numpy as np
    from emulator.npu_device_mini import MEM_DRAM

    num_tiles = hidden_size // dim

    # Input X — use zeros (graph structure doesn't depend on values)
    npu._vrf[MEM_DRAM][0:hidden_size] = np.zeros(hidden_size, dtype=np.float32)

    proj_base = hidden_size + 4
    mat_size = hidden_size * hidden_size
    stride = mat_size + hidden_size

    def tiled_copy(dst_off, W):
        tile_blocks = []
        for tr in range(W.shape[0] // dim):
            for tc in range(W.shape[1] // dim):
                tile = W[tr * dim:(tr + 1) * dim, tc * dim:(tc + 1) * dim]
                tile_blocks.append(tile.flatten())
        td = np.concatenate(tile_blocks)
        npu._vrf[MEM_DRAM][dst_off:dst_off + len(td)] = td

    def bias_copy(dst_off, b):
        npu._vrf[MEM_DRAM][dst_off:dst_off + len(b)] = (
            b.astype(np.float32).flatten()[:hidden_size])

    for i, pname in enumerate(['Q', 'K', 'V', 'selfoutput']):
        off = proj_base + i * stride
        tiled_copy(off, params[pname]['W'].astype(np.float32))
        bias_copy(off + mat_size, params[pname]['b'])

    for i, (wname, bname) in enumerate(
            [('W_intmfc', 'b_intmfc'), ('W_outfc', 'b_outfc')]):
        off = proj_base + (4 + i) * stride
        tiled_copy(off, params[wname].astype(np.float32))
        bias_copy(off + mat_size, params[bname])

    # Unit vectors
    UNIT_VEC_BASE = 0x900
    for j in range(dim):
        e_j = np.zeros(dim, dtype=np.float32)
        e_j[j] = 1.0
        npu._vrf[MEM_DRAM][UNIT_VEC_BASE + j * dim:
                            UNIT_VEC_BASE + j * dim + dim] = e_j

    # LayerNorm params
    ln_base = proj_base + 6 * stride
    ln_size = num_tiles * 8
    ln_vals = [params['LayerNorm']['W'][0], params['LayerNorm']['b'][0],
               params['LayerNorm']['W'][1], params['LayerNorm']['b'][1]]
    for li, vec in enumerate(ln_vals):
        dst = ln_base + li * ln_size
        flat = vec.astype(np.float32).flatten()
        for tr in range(num_tiles):
            chunk = flat[tr * dim:(tr + 1) * dim]
            npu._vrf[MEM_DRAM][dst + tr * 8:
                                dst + tr * 8 + len(chunk)] = chunk

    # SCRATCH zero-fill
    SCRATCH_ADDR = 0x500
    npu._vrf[MEM_DRAM][SCRATCH_ADDR:
                        SCRATCH_ADDR + num_tiles * 8] = (
        np.zeros(num_tiles * 8, dtype=np.float32))


def run_emulator(dim, hidden_size, seq_len, num_head, params, elf_path=None):
    """Run firmware through emulator with EventTracer. Returns (tracer, recorder)."""
    from emulator.npu_device_mini import NpuDeviceMini
    from emulator.npu_event_trace import EventTracer
    from emulator.trace_recorder import TraceRecorder
    from iss.mini_rv64 import MiniRV64

    npu = NpuDeviceMini(native_dim=dim)
    npu.set_hidden_size(hidden_size)
    npu.set_seq_len(seq_len)
    load_weights(npu, params, dim, hidden_size)

    tracer = EventTracer(npu)
    rec = TraceRecorder(npu)

    cpu = MiniRV64()
    cpu.set_mmio_device(rec)
    if elf_path is None:
        elf_path = str((REPO_ROOT / f"firmware/build_dim{dim}/bert.elf").resolve())
    cpu.load_elf(elf_path)
    cpu.run(cycles=300_000)

    return tracer, rec


def generate_bert_params(dim, hidden_size, seq_len, num_head):
    """Generate BERT golden params (weights/biases)."""
    from tests.gen_golden_bert import bert_encoder_layer
    head_size = hidden_size // num_head
    golden, params = bert_encoder_layer(
        add_mask=False, num_head=num_head, head_size=head_size,
        hidden_size=hidden_size, seq_len=seq_len,
        native_dim=dim, precision='emulator_float32', seed=42)
    return params


def render_dot(dot_path: Path, out_dir: Path, no_render: bool):
    """Render .dot to .svg if graphviz is available."""
    if no_render:
        return
    dot_bin = shutil.which("dot")
    if not dot_bin:
        print(f"  ⚠  graphviz 'dot' not found; skipping render of {dot_path.name}")
        return
    for fmt in ("svg",):
        out = out_dir / dot_path.with_suffix(f".{fmt}").name
        cmd = [dot_bin, f"-T{fmt}", f"-o{out}", str(dot_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✓  {out}")
        else:
            print(f"  ✗  {cmd} → {r.stderr.strip()[:120]}")


def main():
    ap = argparse.ArgumentParser(description="Generate NPU graph visualizations from firmware emulation")
    ap.add_argument("--phase", choices=["dag", "micro", "op", "sym", "cluster", "all"], default="all",
                    help="Which graph phase to generate (default: all)")
    ap.add_argument("--dim", type=int, default=2, help="NPU native dimension (default: 2)")
    ap.add_argument("--hidden", type=int, default=4, help="Hidden size (default: 4)")
    ap.add_argument("--seq-len", type=int, default=1, help="Sequence length (default: 1)")
    ap.add_argument("--num-head", type=int, default=2, help="Number of attention heads (default: 2)")
    ap.add_argument("-o", "--output", type=str, default="_out", help="Output directory (default: _out)")
    ap.add_argument("--no-render", action="store_true", help="Only write .dot files, skip graphviz rendering")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase = args.phase
    dim = args.dim
    hidden_size = args.hidden
    seq_len = args.seq_len
    num_head = args.num_head

    # ── Dependency check ───────────────────────────────────────────
    need_sym = phase in ("sym", "all")
    missing = check_deps(need_sym=need_sym)
    if missing:
        print("Missing dependencies — skipping:")
        for m in missing:
            print(f"  • {m}")
        sys.exit(1)

    # ── Build firmware ─────────────────────────────────────────────
    print(f"\n══ Building firmware: dim={dim}, hidden={hidden_size}, seq_len={seq_len}, num_head={num_head} ══")
    elf_path = build_firmware(dim, hidden_size, seq_len, num_head)

    # ── Phase 1: Instruction DAG ───────────────────────────────────
    if phase in ("dag", "all"):
        print("\n══ Phase 1: Instruction DAG ══")
        params = generate_bert_params(dim, hidden_size, seq_len, num_head)
        tracer, rec = run_emulator(dim, hidden_size, seq_len, num_head, params, elf_path)

        from emulator.npu_dag import build_dag, dag_to_dot, dag_to_text

        nodes, edges = build_dag(tracer.events)
        print(f"  {len(nodes)} nodes, {len(edges)} edges")

        # Full DAG
        dot_path = out_dir / "instr_dag.dot"
        dag_to_dot(nodes, edges, path=str(dot_path))
        print(f"  ✓  {dot_path}")
        render_dot(dot_path, out_dir, args.no_render)

        # Text summary
        txt_path = out_dir / "instr_dag.txt"
        txt_path.write_text(dag_to_text(nodes, edges))
        print(f"  ✓  {txt_path}")

        tracer.unpatch()

    # ── Phase 1.5: Micro-Op DAG ─────────────────────────────────
    if phase in ("dag", "micro", "all"):
        print("\n══ Phase 1.5: Micro-Op DAG ══")
        params = generate_bert_params(dim, hidden_size, seq_len, num_head)
        tracer, rec = run_emulator(dim, hidden_size, seq_len, num_head, params, elf_path)

        from emulator.npu_micro_op_dag import (
            collapse_to_micro_ops, build_micro_op_dag,
            micro_op_dag_to_text, micro_op_dag_to_dot,
        )

        micro_ops = collapse_to_micro_ops(tracer.events)
        nodes, edges = build_micro_op_dag(micro_ops)
        total_events = len(tracer.events)
        ratio = total_events / len(nodes) if nodes else 0
        print(f"  {total_events} instructions -> {len(nodes)} micro-ops ({ratio:.1f}x compression)")
        print(f"  {len(edges)} dataflow edges")

        dot_path = out_dir / "micro_op_dag.dot"
        micro_op_dag_to_dot(nodes, edges, path=str(dot_path))
        print(f"  ✓  {dot_path}")
        render_dot(dot_path, out_dir, args.no_render)

        txt_path = out_dir / "micro_op_dag.txt"
        txt_path.write_text(micro_op_dag_to_text(nodes, edges))
        print(f"  ✓  {txt_path}")

        # Connectivity diagnostics
        from emulator.npu_micro_op_dag import check_dag_connectivity
        diag = check_dag_connectivity(
            nodes, edges, dim=dim, hidden_size=hidden_size, seq_len=seq_len)
        print()
        for line in diag.split("\n"):
            print(f"  {line}")

        tracer.unpatch()

    # ── Phase 1.7: DRAM-flow cluster graph ──────────────────────────
    if phase in ("cluster", "all"):
        print("\n══ Phase 1.7: DRAM-Flow Clusters ══")
        params = generate_bert_params(dim, hidden_size, seq_len, num_head)
        tracer, rec = run_emulator(dim, hidden_size, seq_len, num_head, params, elf_path)

        from emulator.npu_micro_op_dag import (
            collapse_to_micro_ops, build_micro_op_dag,
            extract_clusters, clusters_to_text, clusters_to_dot,
        )

        micro_ops = collapse_to_micro_ops(tracer.events)
        nodes, edges = build_micro_op_dag(micro_ops)
        clusters = extract_clusters(nodes, edges, dim=dim, hidden_size=hidden_size, seq_len=seq_len)
        print(f"  {len(clusters)} clusters, {len(nodes)} micro-ops, {len(edges)} edges")

        # Text
        txt_path = out_dir / "dram_clusters.txt"
        txt_path.write_text(clusters_to_text(clusters))
        print(f"  ✓  {txt_path}")

        # DOT + SVG
        dot_path = out_dir / "dram_clusters.dot"
        clusters_to_dot(clusters, nodes, edges, path=str(dot_path))
        print(f"  ✓  {dot_path}")
        render_dot(dot_path, out_dir, args.no_render)

        tracer.unpatch()

    # ── Phase 2: Operator graph ────────────────────────────────────
    if phase in ("op", "all"):
        print("\n══ Phase 2: Operator Graph ══")
        params = generate_bert_params(dim, hidden_size, seq_len, num_head)
        tracer, rec = run_emulator(dim, hidden_size, seq_len, num_head, params, elf_path)

        from emulator.npu_op_graph import build_op_graph, op_graph_to_dot, op_graph_to_text

        batch_sizes = [len(b) for b in rec.extract_batches()]
        graph = build_op_graph(
            tracer.events, dim=dim, hidden_size=hidden_size,
            seq_len=seq_len, batch_sizes=batch_sizes)
        print(f"  {len(graph.nodes)} operators, {len(graph.edges)} edges")

        dot_path = out_dir / "op_graph.dot"
        op_graph_to_dot(graph, path=str(dot_path))
        print(f"  ✓  {dot_path}")
        render_dot(dot_path, out_dir, args.no_render)

        txt_path = out_dir / "op_graph.txt"
        txt_path.write_text(op_graph_to_text(graph))
        print(f"  ✓  {txt_path}")

        tracer.unpatch()

    # ── Phase 3: Symbolic graph ────────────────────────────────────
    if phase in ("sym", "all"):
        print("\n══ Phase 3: Symbolic Graph ══")
        from emulator.npu_op_graph import build_op_graph, op_graph_to_dot, op_graph_to_text
        from emulator.npu_sym_graph import derive_sym_graph, sym_graph_to_dot, sym_graph_to_text, instantiate

        seq_lens = [args.seq_len, args.seq_len * 2]
        graphs = {}
        for sl in seq_lens:
            # Rebuild firmware for this seq_len
            elfi = build_firmware(dim, hidden_size, sl, num_head)
            params = generate_bert_params(dim, hidden_size, sl, num_head)
            tracer, rec = run_emulator(dim, hidden_size, sl, num_head, params, elfi)
            batch_sizes = [len(b) for b in rec.extract_batches()]
            g = build_op_graph(
                tracer.events, dim=dim, hidden_size=hidden_size,
                seq_len=sl, batch_sizes=batch_sizes)
            graphs[sl] = g
            tracer.unpatch()

        sym = derive_sym_graph(graphs, dim=dim, hidden_size=hidden_size)
        positional = [n for n in sym.nodes if "pos" in n.loop_vars]
        constant = [n for n in sym.nodes if not n.loop_vars]
        print(f"  {len(sym.nodes)} symbolic ops ({len(positional)} positional, {len(constant)} constant)")
        print(f"  {len(sym.edges)} symbolic edges")

        dot_path = out_dir / "sym_graph.dot"
        sym_graph_to_dot(sym, path=str(dot_path))
        print(f"  ✓  {dot_path}")
        render_dot(dot_path, out_dir, args.no_render)

        txt_path = out_dir / "sym_graph.txt"
        txt_path.write_text(sym_graph_to_text(sym))
        print(f"  ✓  {txt_path}")

        # Instantiate at original seq_len and show diff
        concrete = instantiate(sym, dim=dim, hidden_size=hidden_size, seq_len=args.seq_len)
        dot_inst = out_dir / "sym_graph_instantiated.dot"
        from emulator.npu_op_graph import op_graph_to_dot as og_to_dot
        og_to_dot(concrete, path=str(dot_inst))
        print(f"  ✓  {dot_inst}  (instantiated at seq_len={args.seq_len}: {len(concrete.nodes)} ops)")
        render_dot(dot_inst, out_dir, args.no_render)

    print("\nDone. Output in", out_dir)


if __name__ == "__main__":
    main()
