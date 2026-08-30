# NPU Closed-Loop Firmware Optimization — How to Run

> **Purpose**: Automated DAG-guided firmware optimization for the rv-npu.
> The loop probes DRAM traffic, generates micro-op DAG graphs, invokes an
> AI agent to identify optimization opportunities (VRF cache, save-load
> pair elimination), validates correctness, and repeats until convergence.
>
> **No git dependency**: All baseline management uses file copies.
> Works on exported code, shallow clones, or any filesystem without .git.

---

## Quick Start (from exported folder)

These steps take you from a fresh export to a completed optimization run.
Expected time: ~3 minutes (first run includes kernel + firmware builds).

### 1. Install Dependencies

```bash
# System packages
sudo apt install -y build-essential cmake python3 python3-pip python3-venv \
    gcc-riscv64-unknown-elf

# Python packages
python -m venv venv_jimu
source venv_jimu/bin/activate
pip install numpy pyelftools pytest
```

### 2. Build Artifacts

```bash
# Build C kernel library (libnpukernels.so)
make kernels

# Build RISC-V firmware for the probe
CC=riscv64-unknown-elf-gcc \
  NATIVE_DIM=2 SEQ_LEN=2 \
  _HIDDEN_SIZE=4 _PROJ_BASE=12 _MAT_SIZE=16 _STRIDE=20 _NUM_TILES=2 \
  _LN1_GAMMA=132 _LN1_BETA=140 _LN2_GAMMA=148 _LN2_BETA=156 \
  _SCRATCH=1280 NUM_HEAD=2 \
  make -C firmware BUILD_DIR=build_dim2 all
```

### 3. Run the Closed Loop

**Pick a goal and run:**

```bash
# G1: DRAM optimization at dim=2 (VRF cache)
make opencode
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization --agent opencode

# G2: Compute efficiency at dim=4 (single-tile projections)
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode

# G3: Combined (dim=4 + VRF cache)
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode
```

**With pi (default agent):**
```bash
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh
```

**With a custom model:**
```bash
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh \
    --goal combined --agent opencode --model opencode/deepseek-v4-flash-free
```

Expected output (pi):
```
[START] Starting from baseline — /* NPU — BERT Encoder Layer Firmware (All Features)
--- Iteration 1 ---
[PROBE] seq=2...  [BUILD] seq=2 OK
[PROBE] seq=6...  [BUILD] seq=6 OK
[PROBE] Generating micro-op DAG for agent analysis...
  seq=2: 2144B  seq=6: 6240B
  Baseline for this run set to 6240B
[AGENT] Invoking pi (timeout: 600s)...
...
[VALIDATE] DRAM: 5856B (saved 384B vs run-start 6240B)
...
===== Done =====
Baseline:   6240B
Best:       5856B
Improvement: 384B (6.2%)
To resume:  ./jimu-dse/scripts/npu_closed_loop.sh --resume .../run-...
```

### 4. Inspect Results

```bash
# View the optimized candidate
ls jimu-dse/results/run-*/candidate_best.c

# View the DAG graphs for audit
ls jimu-dse/results/run-*/dag_iter1/

# View diff against baseline
cat jimu-dse/results/run-*/diff_1.patch
```

### 5. Run with Different Goals and Workloads

The pipeline applies optimizations independent of the workload. You can target either the default BERT model or the Adderboard by passing `--workload adder`.

```bash
# G1: DRAM optimization at dim=2 (VRF cache) on default BERT workload
bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization

# G2: Compute efficiency at dim=4 (single-tile projections) on BERT
bash jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode

# G3: Combined (dim=4 + VRF cache) on BERT
bash jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode

# M0/G1: Run G1 VRF cache optimization on Adderboard
bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization --workload adder --agent opencode
```

---

## Optimization Goals

The pipeline supports three standard optimization goals, each defined in `jimu-dse/goals/<name>/goal.sh`. These goals can be applied to different workloads by passing the `--workload` flag.

| Goal | Name | Dim (BERT) | Dim (Adder) | Skill | Primary Metric |
|------|------|------------|-------------|-------|----------------|
| **G1** | `dram-optimization` | 2 | 4 | `vrf-cache` | `total_bytes` (DRAM) |
| **G2** | `compute-optimization` | 4 | 4 | `dim-optimize` | `test_pass` |
| **G3** | `combined` | 4 | 4 | `dim-optimize` + `vrf-cache` | `test_pass` |

### G1: DRAM Optimization

Reduces DRAM bytes at a fixed dimension by applying VRF cache to eliminate
save-load roundtrips. For BERT, this eliminates roundtrips for K, V, Q, Z, SO, LN, GELU intermediates. For Adder, it focuses on context and score caches.

### G2: Compute Efficiency

Restructures firmware from multi-tile to single-tile projections by
increasing NATIVE_DIM to match hidden_size (BERT specifically). Reduces MV_MUL operations
per projection from 4 to 1, and weight tile loads (M_RD_DRAM) from
144 to 36 ops at seq=6.

### G3: Combined

Applies both transformations. The `dim-optimize` skill is
applied first (restructure for single-tile), then `vrf-cache`
eliminates remaining DRAM save-load roundtrips.


---

## Prerequisites

### System Dependencies (apt)

```bash
sudo apt install -y \
    build-essential cmake \
    python3 python3-pip python3-venv \
    gcc-riscv64-unknown-elf \
    graphviz           # optional, for DOT → SVG rendering of DAG graphs
```

### Python Dependencies (pip)

```bash
# Required — core pipeline
pip install numpy pyelftools pytest

# Optional — full 4-round validation with Amaranth HDL simulation
pip install amaranth

# Optional — for rendering DAG .dot files to SVG
# sudo apt install graphviz   (see above)
```

### RISC-V Cross-Compiler

The firmware targets RV64IM and is compiled with `riscv64-unknown-elf-gcc`:

```bash
# Verify:
riscv64-unknown-elf-gcc --version
# Expected: gcc 10.x or later, target: riscv64-unknown-elf
```

On Debian/Ubuntu: `sudo apt install gcc-riscv64-unknown-elf`

### AI Agents

The loop supports two AI agents. Only one needs to be installed.

#### Option 1: pi (default)

```bash
npm install -g @earendil-works/pi-coding-agent
pi --version
# Expected: 0.79.x or later
```

#### Option 2: OpenCode (alternative)

```bash
# Install OpenCode CLI
npm install -g @openai-code/cli

# Configure agent skills and permissions (run once):
make opencode

# This creates:
#   .opencode/skills/dag-analyze/SKILL.md
#   .opencode/skills/vrf-cache/SKILL.md
#   opencode.json (permissions for file write, read, pytest)
```

The skills are generated from `jimu-dse/docs/skills/isa/*.md` — single source
of truth. Run `make opencode` again if skills are updated.

If neither agent is installed, the loop copies the unmodified firmware
as the candidate and runs as a measurement-only pipeline.

---

## Quick Start

### 1. Build C Kernel Library

```bash
make kernels
```

This builds `_build/kernels/libnpukernels.so` used by the emulator.

### 2. Run a Fresh Optimization Loop

```bash
bash jimu-dse/scripts/npu_closed_loop.sh
```

This runs 5 iterations (configurable via `JIMU_MAX_ITER`):

| Phase | What happens |
|-------|-------------|
| **START** | Copies `jimu-dse/baseline/bert_layer.c` → `firmware/bert/bert_layer.c` |
| **PROBE seq=2** | Builds firmware, runs emulator, measures DRAM traffic |
| **PROBE seq=6** | Same for seq_len=6, generates DAG graphs for agent |
| **ANALYZE** | Reports DRAM ratio seq6/seq2 and cluster breakdown |
| **CONVERGE** | Stops early if improvement < 15% vs run-start baseline |
| **AGENT** | Invokes pi with DAG + DRAM + skills, pi patches bert_layer.c |
| **VALIDATE** | Rebuilds firmware with candidate, re-measures DRAM |
| **DAG** | Generates post-optimization DAG graphs for audit |
| **DEPLOY** | Saves candidate, repeats from iteration 2 with optimized code |

### 3. Resume from a Previous Run

```bash
bash jimu-dse/scripts/npu_closed_loop.sh --resume jimu-dse/results/run-20260622-142411/
```

The run output prints a ready-to-use resume command at the end.

### 4. Run the Validation Test Suite

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

---

## Run Output Structure

Each run creates a timestamped directory:

```
jimu-dse/results/run-<YYYYMMDD>-<HHMMSS>-<PID>/
├── candidate_best.c       ← Best optimized firmware (copy for --resume)
├── candidate_1.c          ← Iteration 1 candidate
├── candidate_2.c          ← Iteration 2 candidate
├── dag_agent/             ← DAG graphs for the agent prompt
│   ├── micro_op_dag.dot
│   └── micro_op_dag.txt
├── dag_iter1/             ← DAG graphs for iteration 1 (audit trail)
│   ├── micro_op_dag.dot
│   ├── micro_op_dag.txt
│   ├── instr_dag.dot
│   ├── op_graph.dot
│   ├── sym_graph.dot
│   └── sym_graph_instantiated.dot
├── dag_iter2/             ← DAG graphs for iteration 2
├── prompt_1.txt           ← Agent prompt for iteration 1
├── prompt_2.txt           ← Agent prompt for iteration 2
├── diff_1.patch           ← Diff against baseline (iteration 1)
├── diff_2.patch           ← Diff against baseline (iteration 2)
├── p2_probe.json          ← DRAM probe result for seq=2
├── p6_probe.json          ← DRAM probe result for seq=6
└── val_1.json             ← Validation result for iteration 1
```

To view a DAG as SVG (requires graphviz):

```bash
dot -Tsvg jimu-dse/results/run-*/dag_iter1/micro_op_dag.dot -o /tmp/dag.svg
```

---

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `JIMU_MAX_ITER` | 5 | Max optimization iterations per run |
| `JIMU_THRESHOLD` | 0.15 | Convergence threshold (fraction of baseline). Stop when improvement < 15% |
| `JIMU_AGENT_TIMEOUT` | 600 | Timeout in seconds for each agent invocation (pi or opencode) |
| `OPENCODE_MODEL` | `opencode/big-pickle` | OpenCode model in provider/model format |
| `CC` | `riscv64-unknown-elf-gcc` | RISC-V cross-compiler |

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--goal <name>` | `dram-optimization` | Optimization goal: `dram-optimization`, `compute-optimization`, `combined` |
| `--agent pi` | pi | Agent to use: `pi` or `opencode` |
| `--model <provider/model>` | `$OPENCODE_MODEL` | OpenCode model (overrides env var) |
| `--resume <dir>` | — | Resume from a previous run's directory |

Examples:

```bash
# G1: DRAM optimization at dim=2
JIMU_MAX_ITER=10 JIMU_THRESHOLD=0.05 bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization

# G2: Compute efficiency at dim=4 with OpenCode
make opencode
JIMU_MAX_ITER=5 bash jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode

# G3: Combined with custom model and resume
bash jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode --model opencode/deepseek-v4-flash-free
```

---

## Optimization Target

The pipeline measures two configurations to detect DRAM traffic patterns:

| Test | seq_len | What it reveals |
|------|---------|-----------------|
| dim2-seq2 | 2 | Baseline with minimal intermediate saves |
| dim2-seq6 | 6 | 3× DRAM scaling reveals VRF overflow / save-load pairs |

Typical unoptimized baseline DRAM:

| Metric | seq=2 | seq=6 |
|--------|-------|-------|
| V_RD_DRAM ops | ~180 | ~456 |
| V_WR_DRAM ops | ~60 | ~156 |
| M_RD_DRAM ops | 144 (constant) | 144 (constant) |
| Total bytes | ~2,144 | ~6,240 |

After VRF cache optimization, typical results:

| Iteration | seq=6 DRAM | Savings vs baseline |
|-----------|-----------|-------------------|
| 0 (baseline) | 6,240B | — |
| 1 (K/V cache) | 5,856B | 6.2% |
| 2 (Q/SO/Z cache) | 4,704B | 24.6% |
| 3 (LN scratch) | 3,936B | 36.9% |
| 4 (X cache) | 3,744B | 40.0% |

---

## Baseline File

The unoptimized reference firmware files are stored in `jimu-dse/baseline/`.
For example, `jimu-dse/baseline/bert_layer.c` and `jimu-dse/baseline/adder_140p.c`.
These are committed copies of the original firmware before any optimizations. To update the baseline:

```bash
# After a successful optimization, promote the best candidate:
cp jimu-dse/results/run-<timestamp>/candidate_best.c jimu-dse/baseline/<target_file>.c
```

---

## Manual Debugging

### Per-Operator Diagnostics

```bash
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s --no-header 2>&1 | grep "max_diff"
```

Shows only final output comparison (Q, K, V intermediates are
optimization-free and not validated).

### Generate All DAG Graphs

```bash
python3 jimu-dse/scripts/generate_all_graphs.py
```

Output in `_out/graphs/<config-name>/`.

---

## Skill Library

| Skill | File | Description |
|-------|------|-------------|
| dag-analyze | `jimu-dse/docs/skills/isa/dag-analyze.md` | Read DAG to identify save-load pairs |
| vrf-cache | `jimu-dse/docs/skills/isa/vrf-cache.md` | Replace DRAM roundtrips with on-chip VRF cache |
| self-verify | `jimu-dse/docs/skills/isa/self-verify.md` | Self-verify after firmware modification |

---

## File Layout

```
.
├── firmware/bert/bert_layer.c    ← Target firmware (modified by agent depending on workload)
├── adderboard/firmware/          ← Adder target firmware
├── jimu-dse/
│   ├── baseline/                 ← Unoptimized reference (committed, no git needed)
│   ├── workloads/                ← Workload manifests (e.g. bert.sh, adder.sh)
│   ├── goals/                    ← Optimization goal configurations
│   │   ├── dram-optimization/    ← G1: VRF cache
│   │   ├── compute-optimization/ ← G2: Single-tile projection
│   │   └── combined/             ← G3: Both G1 + G2
│   ├── scripts/
│   │   ├── npu_closed_loop.sh    ← Main pipeline driver
│   │   └── visualize_graph.py    ← DAG graph generator
│   ├── docs/
│   │   ├── how-to-run.md         ← This file
│   │   ├── npu-closed-loop-design.md ← Full architecture document
│   │   └── skills/isa/           ← Optimization skills for the agent
│   │       ├── dag-analyze.md    ← Read DAG to find save-load pairs
│   │       ├── vrf-cache.md      ← VRF cache (G1 skill)
│   │       ├── dim-optimize.md   ← Dim efficiency (G2/G3 skill)
│   │       └── self-verify.md    ← Post-optimization verification
│   └── results/                  ← Run outputs (gitignored)
├── emulator/                     ← NPU behavioral model (do NOT modify)
├── iss/                          ← RISC-V ISS (MiniRV64)
├── tests/integration/            ← BERT E2E validation tests
└── _build/                       ← C kernel library (make kernels)
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `riscv64-unknown-elf-gcc: command not found` | Cross-compiler not installed | `apt install gcc-riscv64-unknown-elf` |
| `ModuleNotFoundError: No module named 'iss'` | Wrong working directory | Run from repo root |
| `_build/kernels/libnpukernels.so not found` | Kernel library not built | `make kernels` |
| `firmware/build_dim2/bert.elf not found` | Firmware build failed | Check `CC` env var, run `make -C firmware` manually |
| `pi: command not found` | Agent not installed | `npm install -g @earendil-works/pi-coding-agent` |
| pi outputs summary instead of code | Prompt missing write instruction | The loop prompt includes write instructions; check prompt_X.txt |
| `max_diff > 0.05` on output | Optimization introduced numerical error | Revert candidate, check diff_X.patch for incorrect changes |
| `dot: command not found` | Graphviz not installed | `apt install graphviz` (optional, for SVG rendering) |
