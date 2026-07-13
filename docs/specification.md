# NPU Specification

## 1. Overview

The NPU is a firmware-driven SIMD accelerator for transformer inference.
A RISC-V control processor sends 32-bit MMIO instruction words; the NPU
executes them synchronously and signals completion via a status register.

### 1.1 Key Design Principles

- **Firmware-driven**: all computation is expressed as a sequence of 32-bit
  MMIO writes from a RISC-V CPU. The NPU has no instruction cache or sequencer.
- **Synchronous execution**: each instruction runs to completion before the
  next starts. No pipelining, no parallel dispatch.
- **Data-parallel**: compute units operate on vectors of `NATIVE_DIM` elements.
  Internal arithmetic uses FP16 with optional block floating point (BFP).
- **Transformer-optimized**: the ISA directly supports matrix-vector multiply,
  softmax, layer normalization, GELU, residual adds, and attention masking.

### 1.1a Terminology: NATIVE_DIM vs LANES

Two related but distinct parameters define the vector width:

| Term | Scope | Definition |
|------|-------|-----------|
| `NATIVE_DIM` | Firmware (compile-time macro) | Number of elements in a logical vector. Set per firmware build via `-DNATIVE_DIM=N`. All VRF transfers, MV_MUL iterations, and DRAM strides use this value. |
| `LANES` | Hardware (parameter) | Number of parallel FP16 data paths in the physical hardware. Fixed at design time. All compute units (MVU multipliers, MFU pipelines, SLU ALUs) have `LANES` copies. |

**Relationship:** `NATIVE_DIM` must be a multiple of `LANES`. When
`NATIVE_DIM > LANES`, the SLU operates in SerDes mode (Section 4.3) —
it splits the vector into `NATIVE_DIM / LANES` chunks and accumulates
reduction statistics across chunks before the final normalize pass.

Example: with `LANES = 2`, firmware can be compiled at `NATIVE_DIM = 2`
(no SerDes) or `NATIVE_DIM = 4` (SerDes active, vector split into 2
chunks of 2 elements each).

### 1.2 System Block Diagram

```
RISC-V CPU ──MMIO──► Decoder ──┬── MVU (matrix-vector multiply)
  (firmware)         (decode)   ├── MFU (GELU, add, mul)
                                ├── SLU (softmax, layernorm)
                                └── TMM (DRAM ↔ register files)
                                       ├── VMM (vector)
                                       └── MMM (matrix)
```

---

## 2. Instruction Set

Instructions are 32-bit words in two formats:

### 2.1 SI Format (Standard Instruction)

```
Bit:  31      24  23      16  15                       0
      ├──────────┼──────────┼──────────────────────────┤
      │  OpCode  │   Opd0   │          Opd1             │
      │  (8 bit) │  (8 bit) │        (16 bit)           │
      └──────────┴──────────┴──────────────────────────┘
```

OpCode selects the operation; Opd0/Opd1 encode operands (memory target,
sub-opcode, immediate).

### 2.2 LO Format (Long Offset)

```
Bit:  31      24  23                                   0
      ├──────────┼──────────────────────────────────────┤
      │  OpCode  │          Address (24 bit)             │
      │  (8 bit) │                                      │
      └──────────┴──────────────────────────────────────┘
```

Used by DRAM memory operations (opcode ≥ 20). Provides a 24-bit byte address.

### 2.3 Opcode Table

| Dec | Name | Format | Operation |
|-----|------|--------|-----------|
| 0 | `S_WR` | SI | Scalar register write: `reg[opd0] = opd1` |
| 1 | `S_RD` | SI | Scalar register read |
| 2 | `V_RD` | SI | Vector load from VRF: `pipeline = vrf[opd0]` |
| 3 | `M_RD` | SI | Matrix load from row buffer: `mrf = row_buffer` |
| 5 | `V_WR` | SI | Vector store to VRF: `vrf[opd0] = pipeline` |
| 6 | `M_WR` | SI | Matrix write acknowledge |
| 7 | `MV_MUL` | SI | Matrix-vector multiply: `pipeline[r] = Σ mrf[r][c] × v[c]` (no accumulation) |
| 8 | `VV_ADD` | SI | Vector add: `pipeline = vpipe_a + pipeline` |
| 11 | `VV_MUL` | SI | Elementwise multiply: `pipeline = vpipe_a × pipeline` |
| 20 | `V_RD_DRAM` | LO | Vector load from DRAM: `pipeline = dram[addr]` |
| 21 | `V_WR_DRAM` | LO | Vector store to DRAM: `dram[addr] = pipeline` |
| 22 | `V_RD_DRAM_INC` | LO | Vector load with auto-incrementing address |
| 23 | `V_WR_DRAM_INC` | LO | Vector store with auto-incrementing address |
| 24 | `M_RD_DRAM` | LO | Matrix tile load: `mrf = dram[addr..addr+N²]` |
| 25 | `M_WR_DRAM` | LO | Matrix tile store: `dram[addr..addr+N²] = mrf` |
| 27 | `MV_MUL_INC` | SI | Accumulating MV_MUL: loads prev sum from `MVM_ACC_VRF`, adds result |
| 42 | `V_GELU` | SI | GELU activation via LUT |
| 43 | `V_FUNC` | SI | Vector function: opd0=0 → softmax, opd0=1 → layernorm |
| 44 | `SS_ADD` | SI | Scalar-scalar add |
| 45 | `INST_ISSUE` | SI | Chain start: toggles chain ID (not implemented in emulator) |

### 2.3a Format Detection

The decoder distinguishes SI and LO formats by the opcode value.
SI format splits the 24-bit operand field into an 8-bit register/file ID
(opd0) and a 16-bit immediate or sub-opcode (opd1). This is sufficient
because internal register files (VRF, MRF, scalar regs) have IDs < 256
and their indices fit in 16 bits.

OpCodes ≥ 20 access DRAM, which needs a larger address range. These use
LO format: the full 24-bit operand field is a flat byte address, giving
16 MB of addressable DRAM space. The 8-bit opd0 and 16-bit opd1 fields
overlap with the address — the decoder concatenates them:

```python
if opcode >= 20:
    # LO format: operand is the full lower 24 bits
    addr = inst & 0xFFFFFF
else:
    # SI format
    file_id = (inst >> 16) & 0xFF      # opd0: which VRF/MRF/reg file
    index   = inst & 0xFFFF            # opd1: offset within that file
```

### 2.4 VRF Memory Targets

| Name | Value | Size | Purpose |
|------|-------|------|---------|
| `MEM_DRAM` | 0 | 2M+ | External DRAM (flat 24-bit address) |
| `MEM_MULTIPLY_VRF` | 1 | 64 | Temporary multiply storage |
| `MEM_MATRIX_RF` | 4 | 128×128 | Weight matrix register file (MRF) |
| `MEM_MVM_INITIAL_VRF` | 5 | 20480 | MVU input vector RF |
| `MEM_MFU_INITIAL_VRF` | 6 | 4096 | MFU input / VRF cache |
| `MEM_ADDSUB_VRF_0` | 7 | 1024 | AddSub operand A |
| `MEM_ADDSUB_VRF_1` | 8 | 4096 | AddSub operand B |
| `MEM_MVM_ACC_VRF` | 13 | — | MVM accumulator for tiled matmul |
| `MEM_VEC_TO_MAT_ROW` | 18 | — | Vector → matrix row buffer |

### 2.5 Scalar Registers

Scalar registers are internal NPU state written via `S_WR(opd0=addr, opd1=value)`
and read via `S_RD`. They control data transfer dimensions, lane masking,
and precision mode. All registers default to 0 unless noted.

| Name | Addr | Default | Written By | Purpose |
|------|------|---------|------------|---------|
| `TILE_ROWS` | 1 | 1 | Firmware | Number of tile rows for multi-transfer (INC variants iterate `TILE_ROWS × TILE_COLS` vectors). Also sets MRF dimension for `M_RD_DRAM`: loads `TILE_ROWS * NATIVE_DIM` rows. |
| `TILE_COLS` | 2 | 1 | Firmware | Number of tile columns for multi-transfer. Combined with `ITERATIONS` to compute `vec_count = ITERATIONS × TILE_COLS` for INC variants. |
| `ITERATIONS` | 3 | 1 | Firmware | Outer loop count for INC variants. Controls how many batches of `TILE_COLS` vectors are transferred. |
| `READ_VECTOR_MASK` | 15 | 0xFF | Firmware | Per-lane read mask for `V_RD_DRAM`. Bit `i % 8` controls lane `i`. Default `0xFF` enables all lanes. Used for multi-head attention to load only head-specific elements. |
| `WRITE_VECTOR_MASK` | 16 | 0xFF | Firmware | Per-lane write mask for `V_WR`. Bit `i % 8` controls lane `i`. Default `0xFF` enables all lanes. Used for multi-head attention to write context to head-specific element slots. |
| `READ_MATRIX_MASK` | 17 | 0xFF | Firmware | Row select mask for `MV_MUL`. Bit `i` masks rows `i * (NATIVE_DIM/8)` to `(i+1) * (NATIVE_DIM/8) - 1`. Zeroed rows produce zero dot products. |
| `PRECISION_MODE` | 20 | 0 | Firmware | 0 = FP16, 1 = BFP (block floating point). Set to 1 before projection phases, reset to 0 for attention. |

**Firmware usage example** (from `bert_layer.c`):

```c
// Configure for tiled matrix multiply (dim=2, hidden=4, 2×2 tiles):
SEND_SI(OP_S_WR, REG_TILE_ROWS, num_tiles);      // 2 tile rows
SEND_SI(OP_S_WR, REG_TILE_COLS, num_tiles);       // 2 tile columns
SEND_SI(OP_S_WR, REG_ITERATIONS, seq_len);        // iterate per position
SEND_SI(OP_S_WR, REG_READ_MATRIX_MASK, 0xFF);     // enable all MRF rows
SEND_SI(OP_S_WR, REG_PRECISION_MODE, 1);          // enable BFP

// Multi-head attention: mask per head (heads_per_tile=2):
for (int h = 0; h < heads_per_tile; h++) {
    uint8_t head_mask = (h == 0) ? 0x03 : 0x0C;   // elements [0,1] vs [2,3]
    SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, head_mask);
    // ... load Q/K/V slice for this head ...
    SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, head_mask);
    // ... compute and store context for this head ...
}
SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);     // restore full mask
SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, 0xFF);    // restore full mask
```

---

## 3. Command Chain Model

A **command chain** is a sequence of three micro-operations that forms a
effective VLIW instruction:

```
load  operand  → VRF / MRF      (V_RD_DRAM, M_RD_DRAM, V_RD, etc.)
compute       → pipeline        (MV_MUL, VV_ADD, V_FUNC, V_GELU, etc.)
store result  → VRF / DRAM      (V_WR, V_WR_DRAM, etc.)
```

Every NPU operation follows this load-compute-store pattern, which is
expressed as individual 32-bit instructions written sequentially to
`INST_FIFO`. The three instructions in a chain target different functional
units (TMM → MVU/MFU/SLU → TMM), but in the Python emulator they execute
one at a time.

Examples of common chains:

| Load | Compute | Store | Purpose |
|------|---------|-------|---------|
| `V_RD_DRAM input` | `MV_MUL` | `V_WR VRF[7]` | Matrix-vector multiply |
| `V_RD VRF[7]` | `V_GELU` | `V_WR VRF[6]` | GELU activation |
| `V_RD VRF[7]` | `V_FUNC(SOFTMAX)` | `V_WR VRF[7]` | Softmax in-place |
| `V_RD_DRAM a` + `M_RD_DRAM w` | `MV_MUL_INC` + `V_RD_DRAM b` | `VV_ADD` + `V_WR_DRAM out` | Tiled matmul with bias |
| `V_RD VRF[7]` | `VV_ADD` (reads vpipe_a) | `V_WR VRF[7]` | Residual add |

### 3.1 Execution Order

The Python emulator executes each instruction synchronously. Every write
to `INST_FIFO` decodes and runs the instruction to completion before the
next one starts:

```python
def store(self, addr, data):
    if addr == NPU_INST_FIFO:
        self._status = STATUS_BUSY
        self._execute(opcode, opd0, opd1)   # runs to completion
        self._status = STATUS_DONE
```

There is no instruction queue, no scoreboard, no hazard detection, and
no parallel dispatch. Instructions are processed one at a time.

### 3.2 Status

| Register | Offset | Value | Meaning |
|----------|--------|-------|---------|
| `STATUS` | 0x04 | 0 | IDLE — ready for next instruction |
| | | 1 | BUSY — instruction in flight |
| | | 2 | DONE — instruction complete |

Firmware polls `STATUS` until `DONE` before writing the next instruction.

### 3.3 Inter-Chain Overlap and Intra-Chain Tensor Pipelining

The chain model enables two distinct forms of parallelism, both of which
are aspirational (not implemented in the Python emulator):

#### Inter-Chain Overlap (SMC — Simultaneous Multi-Chaining)

Independent chains that have no RAW/WAR/WAW hazards can run concurrently.
Each chain targets a different functional unit (TMM, MVU/MFU/SLU, TMM),
and the `CHAIN_STATUS` register (0x0C) tracks per-unit busy state.

Example: while chain 1 computes `MV_MUL` (MVU busy), chain 2 can load
the next tile into MRF (TMM busy):

```
Time ────────────────────────────────────────────────────>
     ┌─────────────────────────────────────┐
C1   │ M_RD_DRAM(K.T) │ MV_MUL  │ V_WR    │  ← chain 1: Q × K.T
     └───────┬─────────┴─────────┴─────────┘
             │ overlap: TMM loads next tile
             │ while MVU computes
     ┌───────┴──────────────────────────────┐
C2   │ M_RD_DRAM(V) │ MV_MUL  │ V_WR      │  ← chain 2: attn × V
     └──────────────┴─────────┴────────────┘
```

#### Intra-Chain Tensor Pipelining

Within a single chain, the pipeline register threads values between
instructions without explicit source/dest fields.  This enables a
pipelined dataflow where the output of one micro-op is consumed by
the next in the same cycle (in hardware), avoiding intermediate
VRF lookups:

```
Cycle:    1           2           3           4           5
         V_RD_DRAM → MV_MUL → V_FUNC(SOFTMAX) → MV_MUL → V_WR
         (TMM load)  (MVU)     (SLU)            (MVU)    (TMM store)
         │           │         │                │        │
pipe:    Q           scores    attn             context  context
```

Each stage reads the pipeline value written by the previous stage,
forming a **tensor pipeline** where a vector flows through multiple
compute units without leaving the datapath.  In a hardware
implementation, the pipeline register is a single vector-length
register (NATIVE_DIM × FP16) that all units can read/write in the
same cycle.

#### Hazards

| Hazard | Condition | Detection |
|--------|-----------|-----------|
| RAW | Load to MRF → MV_MUL in same chain | Ordering: MRF set by M_RD_DRAM, consumed by MV_MUL |
| RAW | V_RD_DRAM → MV_MUL in same chain | Ordering: pipe set by V_RD_DRAM, consumed by MV_MUL |
| WAR | Two chains writing same VRF | Chain scheduler: check VRF target before issue |
| WAW | Two chains writing same VRF | Chain scheduler: check VRF target before issue |

#### Relation to Existing Examples

The chain examples in `firmware/examples/` demonstrate these concepts:
- `01_single_chain.c`: Basic load-compute-store in one chain
- `02_multi_chain.c`: Two independent chains (MVM + bias add) — candidates
  for inter-chain overlap
- `03_softmax_chain.c`: Q×K.T → softmax → attn×V pipeline through MVU→SLU→MVU
  — the canonical example of intra-chain tensor pipelining

---

## 4. Compute Units

### 4.1 MVU — Matrix-Vector Multiply

Computes dot products between an MRF row and the input vector.
Numerical computation is delegated to `libnpukernels.so::mv_mul()` via ctypes.

- `MV_MUL`: overwrites internal accumulator (accum=0)
- `MV_MUL_INC`: loads previous sum from `MVM_ACC_VRF`, adds, stores back

### 4.2 MFU — Multifunction Unit

Three MFUs with capability-based routing:

| MFU | Operations |
|-----|------------|
| 0 | GELU (1024-entry LUT) |
| 1 | Vector add/sub |
| 2 | Tanh, add/sub, multiply |

The Python emulator routes the instruction to the appropriate C library
function (`gelu`, `vec_add`, `vec_sub`, `vec_mul`) via ctypes.

### 4.3 SLU — Softmax / LayerNorm

A single `V_FUNC` dispatch triggers fused reduction and normalization:

**Softmax:** max reduction → exp(x - max) → sum exp → normalize
**LayerNorm:** sum → mean → variance → inv_sqrt → scale + shift

When `NATIVE_DIM > LANES`, the SLU operates in SerDes mode, iterating over
chunks of `LANES` elements and accumulating statistics before the final
normalization pass.

---

## 5. Memory System

### 5.1 DRAM

Flat 24-bit addressable memory (512K float32 elements). The firmware loads
input tensors and weight matrices before issuing compute instructions.

### 5.2 Vector Register Files

| VRF | ID | Elements | Connected To |
|-----|----|----------|-------------|
| `MVM_INITIAL_VRF` | 5 | 20480 | MVU input |
| `MFU_INITIAL_VRF` | 6 | 4096 | MFU input / VRF cache |
| `ADDSUB_VRF_0` | 7 | 1024 | MFU AddSub operand A |
| `ADDSUB_VRF_1` | 8 | 4096 | MFU AddSub operand B |
| `MVM_ACC_VRF` | 13 | 256 | MVM accumulator (bias pre-load) |

The `*_INC` instruction variants maintain an internal DRAM address register
that auto-increments after each transfer, enabling streaming without explicit
address updates.

### 5.3 Matrix Register File (MRF)

Single `NATIVE_DIM × NATIVE_DIM` tile. Loaded from DRAM via `M_RD_DRAM`.
Supplies one row per `MV_MUL` cycle. Loading a new tile overwrites the
previous one.

---

## 6. TMM — Tensor Memory Manager

Two independent sub-units handle DRAM ↔ register file transfers:

### 6.1 VMM (Vector Memory Manager)

Handles `V_RD_DRAM`, `V_WR_DRAM`, and their INC variants. Single-command FSM
with multi-vector transfer support controlled by `ITERATIONS` × `TILE_COLS`.

### 6.2 MMM (Matrix Memory Manager)

Handles `M_RD_DRAM` and `M_WR_DRAM`. Iterates element-by-element
across matrix tiles.

---

## 7. Precision

- **FP16**: IEEE 754-2008 binary16 at all compute unit boundaries
- **BFP** (optional): block floating point with shared exponent per `LANES` group
- **BFP widths**: 4-bit mantissa (vector), 4-bit mantissa (matrix)

---

## 8. Firmware Interface

### 8.1 MMIO Registers

| Offset | Name | Access | Purpose |
|--------|------|--------|---------|
| 0x00 | `INST_FIFO` | W | Write instruction word |
| 0x04 | `STATUS` | R | 0=IDLE, 1=BUSY, 2=DONE |
| 0x08 | `RESET` | W | Write non-zero → reset |
| 0x0C | `CHAIN_STATUS` | R | Per-unit busy bits |
| 0x20 | `HIDDEN_SIZE` | R/W | Hidden dimension config |
| 0x24 | `SEQ_LEN` | R/W | Sequence length config |

### 8.2 Firmware Send Protocol

```c
// Push one instruction, wait for completion:
npu_send_inst(instruction_word);
while (npu_read_reg(NPU_STATUS) != NPU_STATUS_DONE);
```
