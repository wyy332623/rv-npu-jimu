# NPU Closed-Loop FW-HW Co-Optimization: Design & Skills

> **Audience**: Primary — the optimization agent (AI); Secondary — human experts reviewing, updating, and trusting the system  
> **Status**: Implementation v3 — declarative goal engine  
> **Date**: 2026-06-22

---

## Abstract

This document describes a closed-loop system for automatically optimizing NPU firmware. The system treats each firmware optimization as an **embodied hypothesis**: the candidate firmware expresses a guess about how to better use the NPU's resources (DRAM bandwidth, vector register file capacity, SPU scalar ops, on-chip MRF tile storage). The **3-round validation pipeline** (golden reference → emulator → DAG audit) serves as the evaluation function that accepts or rejects each hypothesis.

### v3 configuration boundary

The executable policy is no longer embedded in the shell loop. A versioned
`jimu-dse/goals/<name>/goal.yaml` declares the target and baseline, hardware
probe, ordered skills, prompt, hard gates, weighted score, convergence, and
artifacts. `jimu-dse/scripts/closed_loop.py` validates this document and owns
the state machine; `npu_closed_loop.sh` is only a compatibility launcher.

The boundary is intentionally declarative: v1 goal files can select built-in
probes and command gates but cannot import or execute Python hooks. A candidate
flows through `PROBE → PROMPT → AGENT → GATE → SCORE`; it becomes the next
iteration's input only after all gates pass and its score clears the configured
promotion delta. The original working firmware is restored on every exit.

Each run persists its resolved configuration and fingerprint, baseline metrics,
per-iteration decisions, best candidate, final JSON summary, and Markdown
report. Resume is allowed only when the configuration fingerprint matches.
The complete schema and operational examples are maintained in
`jimu-dse/docs/how-to-run.md`.

The optional weighted-latency cost model derives NPU DRAM operation counts
from emulator DRAM statistics and on-chip register accesses from EventTracer
def-use resources. Its units remain deliberately abstract and separate from
the cycle-like timing models.

For a more architectural estimate, the optional SCALE-Sim adapter replays each
executed MVU tile operation as a systolic GEMM. Its schema-v2 backend schedules
the traced memory and auxiliary operations on explicit DRAM, VMM, MMM, MVU,
and SPU resources, permitting legal memory/compute overlap while enforcing
data and structural hazards. The legacy additive estimate remains available
for comparison. The adapter, pinned upstream version, model boundary, and
alternatives are documented in `jimu-dse/docs/timing-simulator-selection.md`.

The current probe also supports a native lock-step timing wrapper plus a
workload manifest. The wrapper makes firmware polling observe finite FIFO,
scoreboard, execution-unit, and DRAM timing while preserving
`NpuDeviceMini` as the functional oracle. The manifest names tensor regions
and observable outputs. Executed commands, tensor edges, source lines, and
timing records are joined into `cross-layer-graph.json`; its bounded text view
is supplied to the agent. This generic evidence path is documented in
`docs/unified-firmware-optimization.md` and is independent of the legacy
BERT-specific graph renderers.

Beyond instruction-level optimizations, the document introduces a formal framework for **intentional approximation + compensation** — a class of optimizations where a computation is deliberately performed incorrectly to enable tiling or fusion, followed by a correction pass. This generalizes FlashAttention-style reasoning for the NPU architecture.

The document is structured for **dual reading**: an AI agent reads it as a knowledge base and skill reference for autonomous reasoning; a human expert reads it to understand, trust, and update the system's capabilities.

---

## Table of Contents

1. [The Closed-Loop Architecture](#1-the-closed-loop-architecture)
2. [The NPU Dataflow Model (Agent's Physical Intuition)](#2-the-npu-dataflow-model-agents-physical-intuition)
3. [Optimization Targets & Measurement](#3-optimization-targets--measurement)
4. [Design Exploration Space](#4-design-exploration-space)
5. [Intentional Approximation + Compensation](#5-intentional-approximation--compensation)
6. [Agent Prompting & Skill-Guided Reasoning](#6-agent-prompting--skill-guided-reasoning)
7. [Skill Library Design & Import Pipeline](#7-skill-library-design--import-pipeline)
8. [Validation Protocol & Regression Detection](#8-validation-protocol--regression-detection)
9. [Appendix A: BERT Encoder Layer Tensor Flow Graph](#appendix-a-bert-encoder-layer-tensor-flow-graph)
10. [Appendix B: DRAM Layout for the Skill Library](#appendix-b-dram-layout-for-the-skill-library)
11. [Appendix C: Compensation Formula Reference](#appendix-c-compensation-formula-reference)

---

## 1. The Closed-Loop Architecture

### 1.1 Overview

```
PROBE → ANALYZE → AGENT → VALIDATE → DEPLOY → LOOP
```

| Phase | What happens | Backend |
|-------|-------------|---------|
| **PROBE** | Build firmware, run on emulator at seq=2 and seq=6, measure DRAM traffic + instruction trace + DAG clusters | Emulator (`npu_device_mini.py`) + DAG (`visualize_graph.py`) |
| **ANALYZE** | Compare seq6/seq2 DRAM ratio, detect save-load pairs from DAG, compute arithmetic intensity per cluster | `npu_closed_loop.sh` |
| **AGENT** | AI agent (`pi`) reads DAG + DRAM clusters + skills, generates candidate patch to `bert_layer.c` | pi + skill library (`jimu-dse/docs/skills/`) |
| **VALIDATE** | Rebuild firmware with candidate, re-run emulator, measure DRAM improvement vs run-start baseline | Emulator |
| **DEPLOY** | Save candidate to run directory, generate post-opt DAG graphs for audit, restore baseline for next iteration | `npu_closed_loop.sh` |

### 1.2 The Loop in Detail

```
┌──────────────────────────────────────────────┐
│  Initialize: snapshot current target firmware  │
│  (committed baseline remains an audit ref)     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  PROBE: build firmware + run emulator        │
│  → DRAM stats at seq=2 and seq=6            │
│  → DAG micro-op graph (dag_agent/)           │
│  → DRAM cluster analysis (Load/Store/FLOPs)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  ANALYZE: classify bottleneck                │
│  → seq6/seq2 DRAM ratio                      │
│  → match to skill trigger pattern            │
│  → build agent context                       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  AGENT: generate candidate firmware patch     │
│  → read DAG: identify save-load pairs        │
│    (dag-analyze.md skill)                    │
│  → apply VRF cache transformation            │
│    (vrf-cache.md skill)                      │
│  → output modified bert_layer.c              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  VALIDATE: rebuild + re-probe                │
│  → DRAM reduction vs run-start baseline      │
│  → save diff against baseline file           │
│  → generate post-opt DAG (dag_iterN/)        │
└──────────────────────┬───────────────────────┘
                       │
               ┌───────┴───────┐
               │               │
           IMPROVED         CONVERGED
               │               │
               ▼               ▼
┌──────────────────┐  ┌──────────────────────┐
│  Deploy: save    │  │  Print summary,      │
│  candidate to    │  │  exit loop           │
│  results/run-*/  │  │                      │
│  Continue iter   │  │  Cleanup: restore    │
│  (incremental)   │  │  baseline file       │
└──────────────────┘  └──────────────────────┘
```

### 1.3 Within a Run vs Between Runs

| | Within a run | Between runs |
|--|-------------|--------------|
| **Behavior** | Iteration N+1 starts from iteration N's result (incremental) | A fresh run starts from the target file supplied at invocation |
| **Flag needed** | None — always incremental | `--resume <dir>` to continue from a previous run's best candidate |
| **Baseline** | Measured from the run-start target | The committed file is retained as an audit/reference baseline |

### 1.4 Key Design Points

1. **No git dependency**: Baseline management uses `cp`, not `git checkout`. Works on exported code.
2. **Safe current-target baseline**: scoring starts from the current target; `jimu-dse/baseline/bert_layer.c` remains a committed historical reference.
3. **DAG-guided**: The agent reads `dag_agent/micro_op_dag.txt` to identify save-load pairs before applying optimizations.
4. **Timestamped run directories**: Each run creates `jimu-dse/results/run-<timestamp>/` with all artifacts.
5. **Only `bert_layer.c` is modified**: The agent never touches emulator, ISS, or test code.

---

## 2. The NPU Dataflow Model (Agent's Physical Intuition)

This is the agent's mental model of how data flows through the NPU. Every instruction modifies this state; the agent must track it to reason about optimization.

### 2.1 State at Any Point

```
State after N instructions executed:

DRAM[addr]:     [x0 .. x_{H*SL-1}]     — Input X, Q/K/V/W/b matrices, output buffer
                [tiled Q proj]         — Q=Wq×X in VRF_ACC at 0x200
                [tiled K proj]         — K=Wk×X in VRF_ACC at 0x210
                [tiled V proj]         — V=Wv×X in VRF_ACC at 0x220
                
MRF (on-chip):  [W_tile]               — Currently loaded weight tile (NATIVE_DIM²)
                
VRF (on-chip):
  IVRF (mem 5): [NATIVE_DIM floats]    — Current input vector chunk
  AS0  (mem 7): [NATIVE_DIM floats]    — Tile row 0 accumulator
  AS1  (mem 8): [NATIVE_DIM floats]    — Tile row 1 accumulator (if NUM_TILES > 1)
  MUL  (mem 1): [NATIVE_DIM floats]    — Temporary multiply result
  ACC  (mem13): [NATIVE_DIM floats]    — MVM accumulator (bias pre-loaded)
  MFU  (mem 6): [4096 floats]          — VRF cache for intermediates (K, V, Q, Z, SO, etc.)

SRF (SPU):      [64 floats]            — Scalar register file
  SRF[0]: current tile max (for tiled softmax)
  SRF[1]: current tile sum

Pipe:           [P floats]             — Current pipeline value (result of last compute op)
vpipe_a:        [P floats]             — Saved operand A for VV ops

Registers:
  read_vector_mask:  0xFF              — Per-lane read mask
  write_vector_mask: 0xFF              — Per-lane write mask
  precision_mode:    1                 — BFP enabled
  tile_rows:         2                 — NUM_TILES
  tile_cols:         2
  iterations:        seq_len
```

### 2.2 Instruction Semantics (Dataflow View)

Each instruction is defined by its **effect on the state above**:

```
OP_V_RD_DRAM(addr):
  State before: DRAM[addr..addr+P-1] = data
  State after:  pipe[0..P-1] = masked(DRAM[addr..addr+P-1], read_vector_mask)
                vpipe_a = old_pipe (save current pipe as operand A)
  DRAM cost:    P elements read
  Cycles:       ~3 (VMM latency)

OP_MV_MUL:
  State before: pipe = vector v, MRF = matrix W (NATIVE_DIM×NATIVE_DIM)
  State after:  pipe[i] = Σⱼ W[i][j] × v[j]   (dot product per MRF row)
  DRAM cost:    0 (MRF is on-chip)
  Cycles:       ~3 (MVU latency)

OP_VV_MUL:
  State before: pipe = v, vpipe_a = u
  State after:  pipe[i] = v[i] × u[i]
  DRAM cost:    0
  Cycles:       1 (MFU combinational)

OP_V_FUNC(SUB_SOFTMAX):
  State before: pipe = score vector
  State after:  pipe[i] = exp(score[i]) / Σⱼ exp(score[j])
  DRAM cost:    0
  Cycles:       ~6 (SLU pipeline)

OP_V_WR(mem, addr):
  State before: pipe = data
  State after:  VRF[mem][addr..addr+P-1] = masked(pipe, write_vector_mask)
  DRAM cost:    0

OP_SPU_MAX_REDUCE(addr):
  State before: pipe = vector v
  State after:  SRF[addr] = max(v[0..P-1])
  DRAM cost:    0
  Cycles:       1 (combinational tree, register result on next cycle)

OP_SPU_BROADCAST(addr):
  State before: SRF[addr] = scalar s
  State after:  pipe[i] = s (all P elements)
  DRAM cost:    0
  Cycles:       1 (broadcast)
```

### 2.3 Resource Constraints the Agent Must Track

| Resource | Size | Type | Notes |
|----------|------|------|-------|
| DRAM | 524288 floats | Off-chip | Main memory. Every read/write costs energy + time. |
| MRF | 1 tile (ND²) | On-chip SRAM | Holds exactly one matrix tile. Loading a new tile overwrites the old one. |
| VRF (IVRF) | 20480 floats | On-chip SRAM | MVM initial vector RF. Largest VRF. |
| VRF (AS0) | 1024 floats | On-chip SRAM | AddSub VRF 0. Used for tile accumulator. |
| VRF (AS1) | 4096 floats | On-chip SRAM | AddSub VRF 1. Second tile accumulator. |
| VRF (MFU) | 4096 floats | On-chip SRAM | MFU initial VRF. Used as VRF cache for intermediates. |
| VRF (MUL) | 64 floats | On-chip SRAM | Temporary multiply result storage. |
| VRF (ACC) | 256 floats | On-chip SRAM | MVM accumulator (bias pre-load). |
| SRF | 64 floats | On-chip regs | SPU scalar register file. Tiny but fast. |
| Pipe | P floats | Pipeline reg | Single vector. Every compute op writes here. |
| vpipe_a | P floats | Pipeline reg | Saved operand A. Set by any V_RD/V_RD_DRAM. |

**Critical constraint**: Only one tile can be in MRF at a time. If you need two different weight tiles (e.g., Q and K projections interleaved), you must reload.

---

## 3. Optimization Targets & Measurement

### 3.1 Hierarchical Goals

| Priority | Goal | Metric | Acceptance | When |
|----------|------|--------|------------|------|
| **P0** | Numerical correctness | `max_diff` vs golden | `< 0.05` (1 layer FP16) | Every candidate |
| **P0** | No regressions | Existing test suite | All pass | Every candidate |
| **P1** | DRAM traffic reduction | `total_bytes_moved` | `> 10%` reduction | Every candidate |
| **P2** | Instruction count reduction | `len(trace)` | `> 5%` reduction | Secondary |
| **P3** | Cycle count | HDL sim cycles | Any improvement | When available |

### 3.2 DRAM Cost Model

Each instruction's DRAM cost in FP16 elements (2 bytes each):

| Opcode | Elements per Issue | When |
|--------|-------------------|------|
| `V_RD_DRAM` | P (= NATIVE_DIM) | Every issue |
| `V_WR_DRAM` | P | Every issue |
| `V_RD_DRAM_INC` | P × vec_count | If `vec_count > 1` |
| `V_WR_DRAM_INC` | P × vec_count | If `vec_count > 1` |
| `M_RD_DRAM` | P² (= tile) | Every issue |
| `M_WR_DRAM` | P² | Every issue |
| All others | 0 | On-chip only |

Total = P × (V_RD_count + V_WR_count) + P² × (M_RD_count + M_WR_count)

### 3.3 Typical Baseline DRAM (dim=2, hidden=4)

| Metric | seq=2 | seq=6 | What it reveals |
|--------|-------|-------|-----------------|
| V_RD_DRAM ops | ~180 | ~456 | Scales with seq_len (per-position loads) |
| V_WR_DRAM ops | ~60 | ~156 | Scales with seq_len (per-position saves) |
| M_RD_DRAM ops | 144 | 144 | Constant (weight matrices, independent of seq_len) |
| Total bytes | ~2,144 | ~6,240 | 2.9× ratio — VRF overflow for seq=6 |

Optimization reduces the per-position vector traffic by caching intermediates
in `MFU_INITIAL_VRF` (mem 6). After full optimization:

| Metric | seq=2 | seq=6 |
|--------|-------|-------|
| V_RD_DRAM ops | ~60 | ~168 |
| V_WR_DRAM ops | ~12 | ~12 |
| Total bytes | ~1,248 | ~3,744 |

The remaining vector traffic is irreducible: LN gamma/beta loads, unit vectors
for V.T re-transpose, and weight matrix loads.

---

## 4. Design Exploration Space

### 4.1 Level 1: Instruction Scheduling

| Dimension | Detection Pattern | Transformation | DRAM Saving | Skill |
|-----------|------------------|----------------|-------------|-------|
| **INC folding** | `V_WR_DRAM(addr) → ... → V_RD_DRAM(addr)` | Replace pair with INC variants | ~1 save/load per pair | `isa/inc_folding` |
| **Loop interchange** | Tile loop `for tr: for tc:` with poor MRF reuse | Swap to `for tc: for tr:` | Depends on access pattern | `tiling/loop_interchange` |
| **Op reordering** | Dependency analysis shows stall opportunity | Move M_RD_DRAM earlier | Hides latency, not DRAM | `tiling/op_scheduling` |
| **Dead code elimination** | Instructions whose output is never read | Remove entire sequence | Variable | `isa/dead_code` |

### 4.2 Level 2: VRF Cache (Primary Optimization)

The VRF cache transformation replaces DRAM save-load roundtrips with on-chip
copies to `MFU_INITIAL_VRF` (mem 6, 4096 elements). This is the main
optimization the pipeline targets.

| Intermediate | DRAM Eliminated | Cache Location | Skill |
|-------------|----------------|----------------|-------|
| **K** per position | 2 V_WR + 2 V_RD per position | VRF[6][pos × stride] | `vrf-cache` |
| **V** per position | 2 V_WR + 2 V_RD per position | VRF[6][K_size + pos × stride] | `vrf-cache` |
| **Q** per position | 2 V_RD per position | VRF[6][K_size + V_size + ...] | `vrf-cache` |
| **Z** (attention context) | 2 V_WR + 2 V_RD per position | VRF[6] | `vrf-cache` |
| **SO** (self-output) | 2 V_WR + 2 V_RD per position | VRF[6] | `vrf-cache` |
| **LN scratch** | 2 V_WR + 2 V_RD per LN | VRF[6] | `vrf-cache` |
| **GELU output** | 2 V_WR + 2 V_RD per position | VRF[6] | `vrf-cache` |
| **X input** | 3× X reload per position | VRF[6] via VRF_ADDSUB_1 | `vrf-cache` |

### 4.3 Level 3: Microarchitecture Configuration

| Parameter | Register | Exploration Range | What Changes | Skill |
|-----------|----------|-------------------|--------------|-------|
| Precision mode | REG 20 | 0 (FP16) / 1 (BFP) | DRAM vs accuracy | `hw/precision` |
| Tile rows | REG 1 | 1..NUM_TILES | MRF tile count | `tiling/tile_size` |
| Tile cols | REG 2 | 1..NUM_TILES | MRF tile count | `tiling/tile_size` |
| Iterations | REG 3 | 1..seq_len | Batch size | `tiling/batch_size` |

### 4.4 Level 4: Intentional Approximation + Compensation

See [Section 5](#5-intentional-approximation--compensation).

---

## 5. Intentional Approximation + Compensation

### 5.1 The General Pattern

```
Let O(x) be a gold-standard operator that cannot be tiled.
Let x = [x₀, x₁, ..., x_{T-1}] be tiled input.

Phase 1 — Approximate (per tile, intentionally wrong):
  y_t = tile_O(x_t)        # Compute as if this tile were the whole
  s_t = g(y_t)             # Per-tile statistics (e.g., max, sum)

Phase 2 — Merge (global aggregation):
  global_s = merge(s₀, ..., s_{T-1})   # Combine statistics across tiles
  Uses: SPU.SS_ADD, SPU.MAX_REDUCE     # (accumulate, compare-and-swap)

Phase 3 — Correct (per tile, using global stats):
  y_corrected_t = h(y_t, global_s)     # Apply correction factor
  Uses: SPU.broadcast, VV_MUL, VV_ADD # (scalar→vector, elementwise)

Phase 4 — Verify:
  max_diff(y_corrected, O(x)) < tolerance  # Emulator comparison
```

### 5.2 Concrete: Tiled Softmax (FlashAttention Pattern for NPU)

**Standard softmax**: `P(i) = exp(S[i]) / Σⱼ exp(S[j])`  
Requires global max (for numerical stability) and global sum. Cannot tile naively.

**Tiled softmax with correction**:

```
Per tile t (processing elements i ∈ tile_t):
  1. tile_max[t] = max(S[i] for i ∈ tile_t)
  2. exp_sum[t] = Σ exp(S[i] - tile_max[t])
  3. Save exp_vals[t][i] = exp(S[i] - tile_max[t]) to DRAM

Global merge (after all tiles processed):
  4. global_max = max(tile_max[0], ..., tile_max[T-1])
  5. For each tile t: correction[t] = exp(tile_max[t] - global_max)
  6. global_sum = Σ correction[t] × exp_sum[t]

Per tile correction (reload exp_vals from DRAM):
  7. P(i) = exp_vals[t][i] × correction[t] / global_sum
      = exp(S[i] - tile_max[t]) × exp(tile_max[t] - global_max) / global_sum
      = exp(S[i] - global_max) / global_sum                           ✓
```

**NPU requirements**: SPU (`MAX_REDUCE` for tile_max, `ADD_REDUCE` for exp_sum, `SS_ADD` for global merge, `broadcast` for correction factor), V_EXP (or precomputed exp LUT in DRAM).

### 5.3 The Compensation Pattern Catalog

| Algorithm | Intentional Error | Correction | NPU Prerequisites | Status |
|-----------|------------------|------------|-------------------|--------|
| **Tiled LayerNorm** | Per-tile mean/var on partial vector | Recompute using global mean/var | SLU SERDES | ✅ Exists |
| **Tiled Softmax** | Per-tile softmax on partial vector | Multiply by correction factor using global max/sum | SPU reduce+broadcast | ✅ SPU done; V_EXP needed |
| **Tiled Attention** | Per-tile Q×K attention | Accumulate per-tile context with renormalization | SPU + VRF accumulators | 🔜 After softmax |
| **BFP tile approximation** | Reduced precision per tile | Accumulate in FP32, requantize at boundary | BFP mode toggle + SRF accumulator | ✅ Feasible |
| **Approximate GELU on tile boundary** | Boundary errors at tile edges | Pre-computed correction LUT | LUT in SLU | ✅ Feasible |

### 5.4 How the Agent Discovers Compensation

Given:
- The mathematical definition of an operator (`softmax(S) = exp(S) / sum(exp(S))`)
- The NPU's resource constraints (tile size, SRF depth, SPU operations)
- The primitives available (SPU reduce, SPU broadcast, elementwise arithmetic)

The agent can reason:

> "Softmax requires global statistics. I can only hold one tile in MRF. If I tile the score vector, each tile's softmax is wrong. But I can compute per-tile statistics, merge them across tiles, and apply a per-element correction. Let me derive the correction formula..."

This is **not template matching**. The agent derives the formula from:
1. **The operator's mathematical definition** (from the skill library)
2. **The primitive semantics** (SPU reduce = Σ or max over vector; broadcast = scalar→vector)
3. **The composition rules** (elementwise ops compose: `P = exp(S) × correction_sum / global_sum`)

The derivation is explicit in the agent's output and is **mathematically verified** by the emulator.

---

## 6. Agent Prompting & Skill-Guided Reasoning

### 6.1 The Prompt Structure (Four Layers)

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Current State (from analyze)                              │
│                                                                      │
│  Firmware source: bert_layer.c (current version, line-numbered)     │
│  Instruction trace: last 200 MMIO ops (annotated with DRAM cost)    │
│  Bottleneck: "V_WR_DRAM + V_RD_DRAM pairs consume 40% of DRAM"     │
│  Resource state at bottleneck point:                                │
│    MRF = W_tile[0][0],  VRF_AS0 = acc_row_0,  VRF_AS1 = acc_row_1  │
│    Pipe = last compute result,  SRF = [0, 0, ...]                   │
│                                                                      │
│  Prior attempts: iter1=FAIL(max_diff=0.5), iter2=PASS(DRAM -8%)     │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: NPU Dataflow Model (the "physics engine")                 │
│                                                                      │
│  (The full dataflow model from Section 2 of this document)          │
│  Key: every instruction transforms the state. Track it.             │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Skill Library (the "toolbox")                             │
│                                                                      │
│  Available skills (matching the bottleneck):                         │
│    dag-analyze — read DAG to identify save-load pairs               │
│    vrf-cache   — replace DRAM save/load pairs with VRF cache        │
│    self-verify — run pytest to check correctness                    │
│                                                                      │
│  Each skill includes: trigger, preconditions, transformation,        │
│  validation hook, cost model, and prior success rate.               │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: Task Specification                                        │
│                                                                      │
│  Goal: Generate candidate firmware patch that:                      │
│    (1) Preserves correctness (P0: max_diff < 0.05)                 │
│    (2) Reduces DRAM traffic or instruction count (P1/P2)            │
│    (3) Explains the reasoning explicitly                            │
│                                                                      │
│  Output format: diff-style patch OR full file                        │
│  State your compensation derivation if using intentional approximation│
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Skill Composition

Skills are composable. The agent chains them:

```
Skill A: dag-analyze
  Reads:  jimu-dse/results/dag_agent/micro_op_dag.txt
  Action: Identify DRAM_STORE → DRAM_LOAD pairs by following edges
  
Skill B: vrf-cache
  Depends: A (identified save-load pairs)
  Action: Replace DRAM_WR + DRAM_RD with VREG_MOVE + V_RD from MFU_VRF
```

The agent explores the **skill dependency graph** to find the highest-impact combination.

### 6.3 Reasoning Transparency

The agent must show its reasoning explicitly:

```
Step 1: DAG analysis
  Found edge: node 33 (DRAM_STORE to 0x300) → node 150 (DRAM_LOAD from 0x300)
  This is K[0] — saved to DRAM then read back for attention
  
Step 2: Skill matching
  Pattern matches vrf-cache skill (trigger: DRAM save-load pair)
  Precondition check: no intervening write to same address → OK
  
Step 3: Apply transformation
  Replace V_WR_DRAM(0x300) + ... + V_RD_DRAM(0x300) with
  VREG_MOVE(ADDSUB_VRF_0 → VRF[6], offset) + ... + V_RD(VRF[6], offset)
  
Step 4: Cost model
  Before: 2 × P = 16 elements per position
  After: 0 DRAM elements per position
  For seq=6: 96 elements saved = 384 bytes
```

---

## 7. Skill Library Design & Import Pipeline

### 7.1 Skill Format

Skills are Markdown files in `jimu-dse/docs/skills/`. Each skill includes:

```yaml
name: vrf-cache
version: 1.0.0
category: [isa, fusion]
description: >
  Replace DRAM save-load pairs with on-chip VRF cache operations.
  Used for K, V, Q, Z, SO, LN, GELU, and X intermediates.

trigger:
  pattern: [DRAM_STORE, *, DRAM_LOAD]  # * = any intervening ops
  constraint: |
    addr(store) == addr(load)
    AND no_write_between(store, load, addr)
    AND load_consumes_once(store, load)

preconditions:
  - "The intervening ops do not modify the saved data"
  - "MFU_INITIAL_VRF (mem 6) has sufficient capacity"

transformation:
  before: |
    V_WR_DRAM(addr)
    <intervening ops>
    V_RD_DRAM(addr)
  after: |
    VREG_MOVE(ADDSUB_VRF → VRF[6], offset)
    <same intervening ops>
    V_RD(VRF[6], offset)

cost_model:
  dram_elements_before: 2 × P
  dram_elements_after: 0
  saving: "2P elements per save-load pair"

dependencies: [dag-analyze]
conflicts: []
```

### 7.2 Skill Files

| Skill | File | Description |
|-------|------|-------------|
| dag-analyze | `jimu-dse/docs/skills/isa/dag-analyze.md` | Read micro-op DAG to identify save-load pairs |
| vrf-cache | `jimu-dse/docs/skills/isa/vrf-cache.md` | Replace DRAM roundtrips with on-chip VRF cache |
| self-verify | `jimu-dse/docs/skills/isa/self-verify.md` | Self-verify after firmware modification |

### 7.3 How to Add a New Skill

```
1. Write the skill as a Markdown file in jimu-dse/docs/skills/<category>/
2. Include: name, trigger pattern, preconditions, transformation, cost model
3. Reference any base skills in `based_on:` metadata
4. The agent discovers the skill via the file listing in its prompt
```

---

## 8. Validation Protocol & Regression Detection

### 8.1 The Validation Pipeline

| Round | Backend | Speed | What It Validates | Use |
|-------|---------|-------|-------------------|-----|
| 0 | numpy golden | Instant | Algorithmic correctness | Every candidate |
| 1 | Emulator | ~ms | Instruction semantics, DRAM layout, trace | Every candidate |
| Audit | DAG graphs | ~s | Post-optimization DAG for human/agent review | Every candidate |

HDL rounds (Amaranth cycle-accurate sim) are available on the `master`
branch but not in this FW-only optimization pipeline.

### 8.2 Acceptance Criteria per Optimization Type

| Optimization Type | Tolerance (max_diff) | Notes |
|-------------------|---------------------|-------|
| Instruction reordering | < 1e-6 | No new data paths |
| INC folding | < 1e-6 | Verify INC addressing |
| VRF cache | < 0.05 | Accumulation order changes |
| Operator fusion | < 0.05 | Accumulation order changes |
| Tiled computation | < 0.1 | FP16 rounding across tiles |
| Intentional approx + compensation | < 0.2 | Most aggressive |

### 8.3 Regression Detection

Regression policy is goal-specific. Numerical correctness, build success,
probe completeness, and modified-file scope are represented as hard gates in
`acceptance.gates`. Performance regressions remain visible in normalized metric
details and reduce the weighted score; a goal may turn them into hard command
gates when required.

### 8.4 Convergence

The loop stops when its weighted score reaches `loop.target_score`, when
`loop.max_no_improvement` consecutive candidates fail to advance the best
score by `loop.min_score_delta`, or when `loop.max_iterations` is reached.
All values are stored in the run's resolved configuration.

---

## Appendix A: BERT Encoder Layer Tensor Flow Graph

This is the agent's mental model of tensor flow through one BERT position.

```
                          X (DRAM[0..hidden_size-1])
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    Q = Wq×X + bq   K = Wk×X + bk   V = Wv×X + bv
    (save Q→0x200)  (save K→0x210)  (save V→0x220)
          │               │               │
          ▼               ▼               │
    load Q (mask=0x0F)  load K (mask=0x0F)│
          │               │               │
          └─────── VV_MUL ─┘               │
                      │                    │
                  score_vec                │
                      │                    │
              V_FUNC(SOFTMAX)              │
                      │                    │
                   prob_vec                │
                      │                    │
                      └─── VV_MUL ─────────┘
                               │
                          context_vec
                               │
                    ┌──────────┘
                    ▼
             self_output = Wso × context + bso
                    │
                    ▼
             res1 = self_output + X
                    │
                    ▼
             LN1: layernorm(res1)
                    │
                    ▼
             FFN_intermediate = GELU(Wi × LN1_out + bi)
                    │
                    ▼
             FFN_output = Wo × FFN_intermediate + bo
                    │
                    ▼
             res2 = FFN_output + X
                    │
                    ▼
             LN2: layernorm(res2) = final output
```

### Critical Optimization Points (labeled in the flow)

| Label | Optimization | Skill |
|-------|-------------|-------|
| A: X loaded 3× for Q/K/V | Cache X in VRF across projections | `vrf-cache` |
| B: K, V save-load pairs | Cache K/V in VRF instead of DRAM | `vrf-cache` |
| C: score → softmax → context | Stream V through softmax output | `vrf-cache` |
| D: context → self_output | Eliminate redundant save/load | `vrf-cache` |
| E: residual → LN | Fold bias into LN gamma/beta | `vrf-cache` |
| F: FFN intermediate → GELU | Inline GELU into FFN MVM | `vrf-cache` |

---

## Appendix B: DRAM Layout for the Skill Library

```
DRAM layout for BERT encoder layer (hidden_size=16, NATIVE_DIM=8, num_tiles=2):

Offset  | Content
────────┼──────────────────────────────────────────
0x0000  | X[0..15]  (input vector, seq_len × hidden_size)
0x0010  | X[16..31] (next position, if seq_len > 1)
...     |
0x0014  | Wq (16×16 = 256 elements, tiled as 2×2 tiles of 8×8)
0x0114  | bq (16 elements)
0x0124  | Wk (256 elements, tiled)
0x0224  | bk (16 elements)
0x0234  | Wv (256 elements, tiled)
0x0334  | bv (16 elements)
0x0344  | Wso (256 elements, tiled)
0x0444  | bso (16 elements)
0x0454  | Wi (intermediate FFN, 256 elements, tiled)
0x0554  | bi (16 elements)
0x0564  | Wo (output FFN, 256 elements, tiled)
0x0664  | bo (16 elements)
0x0674  | LN1_gamma (16 elements)
0x0684  | LN1_beta (16 elements)
0x0694  | LN2_gamma (16 elements)
0x06A4  | LN2_beta (16 elements)
...     |
0x0200  | Save area: Q projection result
0x0210  | Save area: K projection result
0x0220  | Save area: V projection result
0x0300  | Save area: self_output + residual
0x0400  | Save area: final output
```

**Tiling convention**: Each W matrix (hidden_size×hidden_size) is stored as
`num_tiles × num_tiles` submatrices of `NATIVE_DIM×NATIVE_DIM`. Tile `[tr][tc]`
is at DRAM offset `base + (tr × num_tiles + tc) × NATIVE_DIM²`.

---

## Appendix C: Compensation Formula Reference

### C.1 Tiled LayerNorm

```
Given: x = [x_0, ..., x_{T-1}]  (T tiles)

Per tile t:
  sum[t] = Σ x_i
  sumsq[t] = Σ x_i²
  n[t] = len(x_t)

Global:
  N = Σ n[t]
  mean = Σ sum[t] / N
  var = Σ sumsq[t] / N - mean²
  inv_std = 1 / sqrt(var + ε)

Per element:
  y_i = gamma × (x_i - mean) × inv_std + beta
```

### C.2 Tiled Softmax

```
Given: S = [S_0, ..., S_{T-1}]  (T tiles of score)

Per tile t:
  tile_max[t] = max(S_i)
  exp_sum[t] = Σ exp(S_i - tile_max[t])
  Save: exp_vals[t][i] = exp(S_i - tile_max[t])

Global:
  global_max = max(tile_max[0], ..., tile_max[T-1])
  correction[t] = exp(tile_max[t] - global_max)
  global_sum = Σ correction[t] × exp_sum[t]

Per element (reload exp_vals):
  P(i) = exp_vals[t][i] × correction[t] / global_sum
       = exp(S_i - global_max) / Σⱼ exp(S_j - global_max)
       = exp(S_i) / Σⱼ exp(S_j)  ✓
```

### C.3 General Pattern

```
For any reduction operator O that needs global statistics to normalize:

1. Identify the statistics needed: g = {g_0, g_1, ..., g_{k-1}}
   Example: softmax needs g = {max, sum}

2. Identify the per-tile computation of g:
   tile_g[t] = per_tile_g(x_t)

3. Identify the merge function M:
   global_g = M(tile_g[0], ..., tile_g[T-1])
   Example: max merge = max over tiles; sum merge = sum over tiles

4. Identify the correction function H:
   corrected_tile = H(O_tile(x_t), tile_g[t], global_g)
   Example: softmax correction = exp(S - tile_max) × exp(tile_max - global_max) / global_sum

5. Verify: H(O_tile(x), g_tile, M(g_tile, ..., g_T)) == O(x)
   This must hold mathematically. If it doesn't, the compensation is wrong.
```

---

## References

1. rv-npu Architecture — `docs/architecture.md`
2. rv-npu Firmware Guide — `docs/firmware-guide.md`
3. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
4. JIMU Tech Report — `docs/jimu-tech-report/` in pyspike-fpga
