# NPU Architecture

## Overview

The NPU is a firmware-driven SIMD accelerator for transformer inference.
A RISC-V control processor sends 32-bit MMIO instruction words to the NPU;
the NPU executes them via its functional units.
The architecture is optimized for the patterns found in transformer models:
matrix-vector multiply, softmax, layer normalization, GELU, and residual adds.

### Two Instruction Issuing Modes

| Mode | Mechanism | Use Case |
|------|-----------|----------|
| **Per-instruction** | Each write to `INST_FIFO` stalls until `STATUS=DONE`. Firmware polls `NPU_STATUS` after every instruction. | Configuration (S_WR), simple sequences, debug. `npu_send_inst()` in `npu_driver.c`. |
| **Chain-based** | Multiple instructions are written without per-instruction stalls, then committed atomically via `INST_ISSUE`. The `CHAIN_STATUS` register tracks per-unit busy state. | Compute-heavy sequences (MVM, softmax, residual). `npu_chain_begin()` + `npu_chain_commit()` + `npu_wait_chain()`. |

In per-instruction mode, each instruction must complete before the next
begins.  In chain mode, instructions within a chain flow through the
pipeline register without DRAM save-load roundtrips, and independent
chains can overlap in hardware (SMC — Simultaneous Multi-Chaining).

### Key Design Principles

- **Firmware-driven**: all computation is expressed as MMIO writes from a
  RISC-V CPU. The NPU has no instruction cache or sequencer.
- **Synchronous execution (per-instruction mode)**: each instruction runs
  to completion before the next starts. No pipelining, no parallel dispatch.
- **Asynchronous execution (chain mode)**: instructions within a chain
  sequence through the pipeline register; multiple chains can run
  concurrently on different functional units.
- **Data-parallel**: all units operate on vectors of `NATIVE_DIM` elements.
- **Transformer-optimized**: ISA directly supports the ops needed for
  attention, feed-forward, and normalization layers.

### System Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RISC-V CPU (firmware)                         │
│                                                                          │
│    Mode A: Per-instruction                          Mode B: Chain-based │
│    ──────────────────────                          ──────────────────── │
│    send(SI(OP_M_RD_DRAM))                          npu_chain_begin()    │
│    while(STATUS != DONE);     ← stal               send(...)            │
│    send(SI(OP_MV_MUL, 0, 0))                       send(...)            │
│    while(STATUS != DONE);     ← stal               npu_chain_commit()   │
│    send(SI(OP_V_WR, ...))                          while(CHAIN_STATUS)  │
│    while(STATUS != DONE);     ← stal               ← per-unit busy      │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │ MMIO (32-bit instruction words)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Instruction Decoder                          │
│                                                                  │
│  Per-instruction path:  decode → execute → STATUS=DONE           │
│  Chain path:            decode → chain FIFO → INST_ISSUE commits │
│                                                                  │
│  Pipeline register threads values between instructions:          │
│    V_RD → pipe, MV_MUL → pipe, V_FUNC → pipe, V_WR captures    │
│                                                                  │
│  CHAIN_STATUS tracks per-unit busy (bit0=VMM, bit1=MMM, bit2=   │
│  MVU).  No scoreboard, no hazard checking in Python emulator.    │
└──────┬─────────────────────┬────────────────┬──────────┬──────────┘
       │                     │                │          │
       ▼                     ▼                ▼          ▼
┌──────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐
│  TMM         │   │  MVU         │   │  MFU     │   │  SLU         │
│  (DRAM ↔     │   │  (matrix-    │   │  (GELU,  │   │  (softmax,   │
│   register)  │   │   vector)    │   │   add,   │   │   layernorm) │
│  ┌─────────┐ │   │  ┌─────────┐ │   │  ┌─────┐ │   │  ┌─────────┐ │
│  │ V_RD    │ │   │  │ MV_MUL  │ │   │  │GELU │ │   │  │ Max     │ │
│  │ V_WR    │ │   │  │         │ │   │  └─────┘ │   │  │ │       │ │
│  │         │ │   │  │ (C lib  │ │   │  ┌─────┐ │   │  │ Exp    │ │
│  │ M_RD    │ │   │  │  call)  │ │   │  │add/ │ │   │  │ │       │ │
│  │ M_WR    │ │   │  └─────────┘ │   │  │sub  │ │   │  │ Sum    │ │
│  └─────────┘ │   │              │   │  └─────┘ │   │  │ │       │ │
│              │   │              │   │  ┌─────┐ │   │  │Normal  │ │
│              │   │              │   │  │tanh/ │ │   │  └─────────┘ │
└──────────────┘   │              │   │  │mul  │ │   │              │
                    └──────────────┘   │  └─────┘ │   └──────────────┘
                                       └──────────┘

Register Files (VRF):
┌────────────┬─────┬──────────┬────────────────────────────┐
│ Bank       │ ID  │ Elements │ Connected To               │
├────────────┼─────┼──────────┼────────────────────────────┤
│ DRAM       │  0  │  524288  │ Off-chip memory            │
│ MULTIPLY   │  1  │      64  │ Temporary multiply storage │
│ MATRIX_RF  │  4  │ 128×128  │ Weight tile (MRF)          │
│ MVM_INIT   │  5  │   20480  │ MVU input vector           │
│ MFU_INIT   │  6  │    4096  │ MFU input / VRF cache      │
│ ADDSUB_0   │  7  │    1024  │ MFU AddSub operand A       │
│ ADDSUB_1   │  8  │    4096  │ MFU AddSub operand B       │
│ MVM_ACC    │ 13  │     256  │ MVM accumulator            │
│ VEC_TO_MAT │ 18  │       —  │ Vector → matrix row buffer │
└────────────┴─────┴──────────┴────────────────────────────┘

Data Flow:
  DRAM ──TMM──► VRF │ MRF ──MVU──► pipeline ──MFU/SLU──► VRF ──TMM──► DRAM
                     │                  │
                     └── VREG_MOVE ─────┘  (VRF-to-VRF moves)
```

### The Pipeline Register

"Pipeline register" (or `pipe` / `vpipe_a`) is a **single vector-width
register** that carries the live computation result between instructions.
Every compute instruction writes its output to the pipeline register.
The next instruction reads from it as one of its operands:

```
MV_MUL:  pipeline = Σ(MRF[row][c] × VRF[input][c])
VV_ADD:  pipeline = vpipe_a + pipeline   (reads previous pipeline output)
SOFTMAX: pipeline = softmax(pipeline)
V_WR:    VRF[dst] = pipeline             (capture result to VRF)
```

### SerDes Mode (SLU)

SerDes (Serializer/Deserializer) mode is used when `NATIVE_DIM > LANES`.
The hardware has only `LANES` parallel FP16 data paths, but a vector may
contain more elements. In SerDes mode, `V_FUNC` divides the vector into
chunks of `LANES` elements. For **softmax**: it accumulates max and sum
across all chunks, then does one final normalize pass. For **layernorm**: it
accumulates sum and sum-of-squares across chunks, computes the global
variance, then normalizes all chunks.

## Functional Units

### TMM — Tensor Memory Manager

Two independent sub-units move data between DRAM and register files:

- **VMM**: vector transfers (`V_RD_DRAM`, `V_WR_DRAM`, INC variants).
  Supports multi-vector streaming via tile count registers.
- **MMM**: matrix tile transfers (`M_RD_DRAM`, `M_WR_DRAM`).

### MVU — Matrix-Vector Multiply Unit

Computes dot products between an MRF row and the input vector.
Supports accumulating mode (`MV_MUL_INC`) for tiled matmul across
multiple weight tile columns. The MVM accumulator VRF (`MEM_MVM_ACC_VRF`)
holds partial sums between tile column iterations.

Numerical computation is delegated to `libnpukernels.so::mv_mul()`.
The Python code sets up the operands and calls C via ctypes.

### MFU — Multifunction Unit

Three MFUs with capability-based routing:
- MFU 0: GELU (LUT-based)
- MFU 1: vector add/subtract
- MFU 2: tanh, add/sub, multiply

Arithmetic is delegated to C library functions (`gelu`, `vec_add`,
`vec_sub`, `vec_mul`). The Python model selects the MFU and routes
operands.

### SLU — Softmax / LayerNorm Unit

Fused reduction and normalization in a single `V_FUNC` dispatch:
- **Softmax**: max reduction → exp(x - max) → sum exp → normalize
- **LayerNorm**: sum → mean → variance → inv_sqrt → scale + shift

Supports SerDes mode when `NATIVE_DIM > LANES`, accumulating intermediate
statistics across chunks before the final normalization pass.

Arithmetic is delegated to C library functions (`softmax`, `layernorm`).

## Register Files

### Vector Register Files (VRF)

| Bank | ID | Elements | Purpose |
|------|----|----------|---------|
| `MEM_DRAM` | 0 | 524288 | External DRAM |
| `MEM_MULTIPLY_VRF` | 1 | 64 | Temporary multiply storage |
| `MEM_MATRIX_RF` | 4 | 128×128 | Weight matrix tile (MRF) |
| `MEM_MVM_INITIAL_VRF` | 5 | 20480 | MVU input vector |
| `MEM_MFU_INITIAL_VRF` | 6 | 4096 | MFU input / VRF cache |
| `MEM_ADDSUB_VRF_0` | 7 | 1024 | MFU AddSub operand A |
| `MEM_ADDSUB_VRF_1` | 8 | 4096 | MFU AddSub operand B |
| `MEM_MVM_ACC_VRF` | 13 | 256 | MVM accumulator |
| `MEM_VEC_TO_MAT_ROW` | 18 | — | Vector → matrix row buffer |

### Matrix Register File (MRF)

Single `NATIVE_DIM × NATIVE_DIM` tile. Loaded from DRAM via `M_RD_DRAM`.
Only one tile can be resident at a time.

## Execution Model (Python Emulator)

The Python emulator implements both instruction issuing modes:

### Per-Instruction Mode

1. Firmware writes instruction word to `INST_FIFO` (MMIO offset 0x00).
2. Emulator decodes the 32-bit opcode and operands.
3. Emulator calls the appropriate functional unit method, which may
   call into `libnpukernels.so` via ctypes for arithmetic.
4. Instruction completes synchronously.
5. `STATUS` (offset 0x04) becomes `DONE`.
6. Firmware reads `STATUS` to confirm completion before the next write.

### Chain Mode

1. Firmware writes multiple instruction words to `INST_FIFO` without
   polling `STATUS` between them.
2. Each instruction executes synchronously in sequence (the emulator has
   no parallelism), with the pipeline register threading values between
   consecutive instructions.
3. Firmware writes `INST_ISSUE` to commit the chain.
4. Firmware polls `CHAIN_STATUS` (offset 0x0C) until all bits are 0,
   indicating all functional units have completed.

### Differences from Hardware Model

The functional Python emulator (`NpuDeviceMini`) is single-threaded and
non-pipelined. Each instruction runs to completion before the next starts.
There is no cycle counter, hazard detection, or parallel dispatch in that
model. Chain mode exists there for correctness verification only. Timing-aware
execution is provided separately by `NpuDeviceTimed` and the Verilator RTL
backend described below (see specification.md §3.3 for chain semantics).

## Implementation: Python vs C

| Component | Language | Notes |
|-----------|----------|-------|
| **RISC-V ISS** (`iss/mini_rv64.py`) | Python | ~1500 lines. Runs the RV64IM ELF binary one instruction per `step()`. |
| **NPU device model** (`emulator/npu_device_mini.py`) | Python | Functional emulator. Handles MMIO, register files, instruction decode, DRAM transfers. |
| **Timed device** (`emulator/npu_device_timed.py`) | Python | Lock-step FIFO, scoreboard, execution-unit, and DRAM timing around the functional emulator. |
| **RTL timing core** (`rtl/jimu_npu_timing_core.sv`) | SystemVerilog | Finite ROB, independent controllers, semantic dependencies, DRAM serialization, and bank conflicts. |
| **RTL replay adapter** (`emulator/npu_rtl_sim.py`) | Python + C++ | Encodes trace resources and drives the Verilator harness in `sim/jimu_rtl_harness.cpp`. |
| **Instruction decoder** | Python | Decodes 32-bit instructions and dispatches to the appropriate functional unit method. |
| **TMM** (VMM + MMM) | Python | DRAM ↔ VRF/MRF transfers with address auto-increment logic. |
| **MVU** | Python (control) + C (arithmetic) | Python sets up operands; calls `libnpukernels.so::mv_mul()` via ctypes. |
| **MFU** (GELU, add, sub, mul) | Python (control) + C (arithmetic) | Python selects MFU and routes operands; arithmetic calls C library (`gelu`, `vec_add`, `vec_sub`, `vec_mul`). |
| **SLU** (softmax, layernorm) | Python (control) + C (arithmetic) | Python manages SerDes reduction loop; arithmetic calls C library (`softmax`, `layernorm`). |
| **BERT firmware** (`firmware/bert/bert_layer.c`) | C (RV64IM) | Compiled with `riscv64-unknown-elf-gcc`, runs on the Python ISS. |
| **Kernel library** (`kernels/*.c` → `libnpukernels.so`) | C (x86) | **Backend compute engine.** Called via ctypes for all numerical operations. |
| **Golden reference** (`tests/gen_golden_bert.py`) | Python | Pure NumPy implementation used by pytest for end-to-end validation. Not called by the emulator. |
| **Tests** | Python | pytest parametrized across dim/hidden/seq_len/config. |

> **`libnpukernels.so` is mandatory.** If it does not exist, the emulator
> raises `FileNotFoundError` at startup and cannot run. It is not a
> standalone golden reference — it is the core arithmetic backend.

## Timing Model

The project has three complementary execution layers:

| Layer | Authority | Main use |
|-------|-----------|----------|
| `NpuDeviceMini` | Numerical values and instruction semantics | Functional correctness, DRAM accounting, and trace/DAG generation |
| `NpuDeviceTimed` | Firmware-visible FIFO/status timing | Busy/done polling, queue pressure, scoreboarding, and resource contention |
| Verilator RTL timing core | RTL command/control schedule | ROB dependencies, controller overlap, SRAM-bank conflicts, fences, and cycle counters |

`NpuDeviceMini` itself remains a functional model without timing. Each
instruction written to `INST_FIFO` executes synchronously before the next one:

```python
def _push_instruction(self, inst):
    self._status = STATUS_BUSY
    self._execute(...)      # ← entire instruction finishes here
    self._status = STATUS_DONE
```

Its `tick()` method is a no-op, so timing conclusions must come from one of the
timing layers. `NpuDeviceTimed` reuses the functional emulator as its retirement
oracle while modeling firmware-visible progress. The RTL backend replays the
same canonical command trace through a synthesizable scheduling core. The
active RTL latency and initiation-interval parameters are first-pass,
HDL-derived contracts rather than calibrated silicon measurements; see
`docs/hdl-derived-timing-parameters.md` and `docs/rtl-timing-simulator.md`.

---

## Comparison with Related Accelerator Architectures

This NPU shares high-level design principles with several contemporary and
research accelerators, but differs in microarchitecture, scale, and target
deployment. The table below compares key architectural characteristics.

### Microsoft Brainwave (ISCA 2018)

Brainwave is a cloud-scale FPGA-based DNN serving platform. Key differences:

| Aspect | Brainwave | This NPU |
|--------|-----------|----------|
| Implementation | Intel Stratix 10 FPGA overlay | Python emulator |
| Compute core | Systolic array / tensorized MVU | Sequential dot-product engine (1 row/cycle) |
| Matrix size | Arbitrary (tensorized tiling) | Single `N×N` tile (N=LANES) |
| Dispatch host | x86 CPU over PCIe | On-chip RISC-V via MMIO |
| Softmax / LN | Microcoded sequences | Fused `V_FUNC` instructions |
| Scale | Datacenter fabric | Single-chip research |
| Memory hierarchy | Deep FPGA BRAM hierarchy | Flat DRAM + VRF/MRF register files |

### Groq LPU (ISCA 2021 / Hot Chips 2023)

Groq's Language Processing Unit is a deterministic tensor streaming architecture
with a compile-time-scheduled dataflow fabric. Key differences:

| Aspect | Groq LPU | This NPU |
|--------|----------|----------|
| Scheduling | Compile-time static schedule | Runtime firmware dispatch |
| Compute model | Tensor streaming (SIMD across lanes + time) | Vector sequential (per-op) |
| Hazard model | None — all dependencies resolved at compile time | Scoreboard per chain (RAW/WAR/WAW) |
| Memory | Distributed SRAM tiles (~230MB total) with streaming | Centralized DRAM + VRF register files |
| Determinism | Cycle-exact deterministic (no arbitration) | Data-dependent via runtime hazard detection |
| Compilation | Full static scheduling (tensor → assembly → time) | C firmware with inline MMIO instructions |
| Functional units | 320 SXM modules (each with MAC + mem) | MVU + MFU(3) + SLU + TMM |

### FSA — Fused Systolic Attention

FSA (arXiv:2507.11331 "SystolicAttention: Fusing FlashAttention within a
Single Systolic Array", [github.com/VCA-EPFL/FSA](https://github.com/VCA-EPFL/FSA))
is an enhanced systolic array architecture that runs the entire FlashAttention
(Q×K^T → softmax → S×V) on a single systolic array without external vector
units. It achieves fine-grained element-wise overlapping of the attention
operations, maximizing array utilization while preserving the original FP
operation order of FlashAttention.

| Aspect | FSA | This NPU |
|--------|-----|----------|
| Compute engine | Single systolic array for all attention phases | Sequential MVU + MFU + SLU |
| Softmax | Element-wise on systolic array | Discrete `V_FUNC(SOFTMAX)` instruction |
| Overlap | Fine-grained, element-wise fused | Per-operation (load → compute → store) |
| Implementation | Synthesizable RTL (16nm, 1.5GHz) | Python functional emulator |
| Programmability | Custom kernel (SystolicAttention) | RISC-V firmware with MMIO instructions |
| Tiling strategy | FlashAttention-style (tiled across SRAM banks) | Tile-row based (weight tiles × positions) |

### Key Architectural Differences Summary

| Feature | Brainwave | Groq LPU | FSA | This NPU |
|---------|-----------|----------|-----|----------|
| Fabric | FPGA overlay | Custom ASIC (7nm) | Custom ASIC (simulated) | Python emulator |
| Dispatch model | CPU-driven | Static schedule (dataflow) | CUDA-like kernel launch | Per-instruction + chain-based (dual mode) |
| Dot product | Systolic array | SIMD MAC array | Systolic + fused softmax | Sequential (1 row/cycle) |
| Normalization | Microcoded | SIMD elementwise | Online fused (in GEMM) | Fused `V_FUNC` HW |
| Hazard detection | None (sequencer order) | None (compile-time) | None (kernel sync) | Scoreboard (RAW/WAR/WAW) |
| On-chip storage | BRAM banks (~10MB) | Distributed SRAM (~230MB) | SRAM hierarchy (banks) | VRF + MRF (~64KB) |
| Target deployment | Cloud inference serving | LLM inference (datacenter) | LLM prefill (research) | Single-chip research |
