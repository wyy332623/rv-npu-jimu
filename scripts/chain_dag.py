#!/usr/bin/env python3
"""
Generate DAG diagrams for the single-chain and multi-chain examples.

Emits event-level and micro-op DAGs showing the flow of tensors through
the pipeline, MRF, and VRF — matching the diagrams in firmware/examples/.

Usage:
    python3 scripts/chain_dag.py [--output dir]

Output:
    chain_example_events.dot/png  — flat event-level DAG
    chain_example_microops.dot/png — collapsed micro-op DAG
"""

import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from emulator.npu_device_mini import (
    NpuDeviceMini, MEM_DRAM,
    MEM_MULTIPLY_VRF, MEM_MVM_INITIAL_VRF, MEM_MATRIX_RF,
    MEM_FILL, MEM_SPU_BROADCAST, MEM_VEC_TO_MAT_ROW,
    OP_V_RD_DRAM, OP_V_WR_DRAM,
    OP_V_RD, OP_V_WR,
    OP_M_RD_DRAM, OP_M_WR, OP_M_RD, OP_MV_MUL,
    OP_VV_ADD, OP_VV_B_SUB_A, OP_VV_MUL,
    OP_V_SIGM, OP_V_EXP,
    OP_INST_ISSUE, OP_S_WR, OP_S_RECIP,
    REG_TILE_ROWS, REG_TILE_COLS, REG_ITERATIONS,
)
from emulator.npu_event_trace import EventTracer
from emulator.npu_dag import build_dag, dag_to_text, dag_to_dot
from emulator.npu_micro_op_dag import (
    collapse_to_micro_ops, build_micro_op_dag,
    micro_op_dag_to_text, micro_op_dag_to_dot,
)


def si(op, opd0, opd1):
    return ((op & 0xFF) << 24) | ((opd0 & 0xFF) << 16) | (opd1 & 0xFFFF)

def lo(op, addr):
    return ((op & 0xFF) << 24) | (addr & 0xFFFFFF)


def run_single_chain(tracer):
    """Mirrors firmware/examples/01_single_chain.c"""
    dev = tracer._inner
    WEIGHT = 0x400
    VECTOR = 0x2000
    RESULT = 0x2100

    def push(inst):
        dev._push_instruction(inst)

    # ── Configuration (not part of chain) ──
    push(si(OP_S_WR, REG_TILE_ROWS, 1))
    push(si(OP_S_WR, REG_TILE_COLS, 1))
    push(si(OP_S_WR, REG_ITERATIONS, 1))

    # ── Chain: MVM ──
    push(lo(OP_M_RD_DRAM, WEIGHT))                # [1] M_RD_DRAM  → MRF
    push(si(OP_M_WR, MEM_MATRIX_RF, 0))           # [2] M_WR        commit
    push(lo(OP_V_RD_DRAM, VECTOR))                # [3] V_RD_DRAM  → pipeline
    push(si(OP_MV_MUL, 0, 0))                     # [4] MV_MUL     MRF × pipe
    push(si(OP_V_WR, MEM_MULTIPLY_VRF, 0))        # [5] V_WR       pipe → VRF
    push(si(OP_INST_ISSUE, 0, 0))                 # [6] INST_ISSUE commit

    # ── Second chain: VRF → DRAM ──
    push(si(OP_V_RD, MEM_MULTIPLY_VRF, 0))        # [7] V_RD       VRF → pipe
    push(lo(OP_V_WR_DRAM, RESULT))                # [8] V_WR_DRAM  pipe → DRAM
    push(si(OP_INST_ISSUE, 0, 0))                 # [9] INST_ISSUE commit


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != '--output' else '.'
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_dir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else '.'
    os.makedirs(output_dir, exist_ok=True)

    # Run single chain through EventTracer
    npu = NpuDeviceMini(native_dim=4)
    tracer = EventTracer(npu)
    run_single_chain(tracer)

    # ── Event-level DAG ──
    events = tracer.events
    nodes, edges = build_dag(events)

    text = dag_to_text(nodes, edges)
    print("=== Event-level DAG (text) ===")
    print(text)
    print()

    dot = dag_to_dot(nodes, edges)
    dot_path = os.path.join(output_dir, 'chain_example_events.dot')
    with open(dot_path, 'w') as f:
        f.write(dot)
    print(f"Wrote {dot_path}")

    # ── Micro-op DAG ──
    micro_ops = collapse_to_micro_ops(events)
    micro_ops, micro_edges = build_micro_op_dag(micro_ops)

    text = micro_op_dag_to_text(micro_ops, micro_edges)
    print("=== Micro-op DAG (text) ===")
    print(text)
    print()

    dot = micro_op_dag_to_dot(micro_ops, micro_edges)
    dot_path = os.path.join(output_dir, 'chain_example_microops.dot')
    with open(dot_path, 'w') as f:
        f.write(dot)
    print(f"Wrote {dot_path}")

    # ── Multi-chain (silu_mvm_residual pattern) ──
    npu2 = NpuDeviceMini(native_dim=4)
    tracer2 = EventTracer(npu2)
    dev2 = tracer2._inner
    GATE = 0x3000
    UP   = 0x3004
    RES  = 0x3008
    OUT  = 0x300C
    W_DOWN = 0xA00

    def push2(inst):
        dev2._push_instruction(inst)

    # Configuration
    push2(si(OP_S_WR, REG_TILE_ROWS, 1))
    push2(si(OP_S_WR, REG_TILE_COLS, 1))
    push2(si(OP_S_WR, REG_ITERATIONS, 1))

    # SiLU × up → W_down → residual chain
    push2(lo(OP_V_RD_DRAM, GATE))
    push2(si(OP_V_SIGM, 0, 0))
    push2(si(OP_V_WR, MEM_MULTIPLY_VRF, 0))
    push2(lo(OP_V_RD_DRAM, GATE))
    push2(si(OP_V_RD, MEM_MULTIPLY_VRF, 0))
    push2(si(OP_VV_MUL, 0, 0))
    push2(si(OP_V_WR, MEM_MULTIPLY_VRF, 0))
    push2(lo(OP_V_RD_DRAM, UP))
    push2(si(OP_V_RD, MEM_MULTIPLY_VRF, 0))
    push2(si(OP_VV_MUL, 0, 0))
    push2(si(OP_V_WR, MEM_MVM_INITIAL_VRF, 0))
    push2(lo(OP_M_RD_DRAM, W_DOWN))
    push2(si(OP_M_WR, MEM_MATRIX_RF, 0))
    push2(si(OP_V_RD, MEM_MVM_INITIAL_VRF, 0))
    push2(si(OP_MV_MUL, 0, 0))
    push2(si(OP_V_WR, MEM_MULTIPLY_VRF, 0))
    push2(lo(OP_V_RD_DRAM, RES))
    push2(si(OP_V_RD, MEM_MULTIPLY_VRF, 0))
    push2(si(OP_VV_ADD, 0, 0))
    push2(lo(OP_V_WR_DRAM, OUT))
    push2(si(OP_INST_ISSUE, 0, 0))

    events2 = tracer2.events
    nodes2, edges2 = build_dag(events2)
    micro_ops2 = collapse_to_micro_ops(events2)
    micro_ops2, micro_edges2 = build_micro_op_dag(micro_ops2)

    dot = micro_op_dag_to_dot(micro_ops2, micro_edges2)
    dot_path = os.path.join(output_dir, 'silu_chain_microops.dot')
    with open(dot_path, 'w') as f:
        f.write(dot)
    print(f"Wrote {dot_path}")

    print("\nTo render: dot -Tpng <dotfile> -o <pngfile>")
    try:
        import subprocess
        for f in os.listdir(output_dir):
            if f.endswith('.dot'):
                dot_path = os.path.join(output_dir, f)
                png_path = dot_path.replace('.dot', '.png')
                subprocess.run(['dot', '-Tpng', dot_path, '-o', png_path],
                              capture_output=True)
                if os.path.exists(png_path):
                    print(f"  Rendered {png_path}")
    except FileNotFoundError:
        print("  (install graphviz for PNG rendering)")


if __name__ == '__main__':
    main()
