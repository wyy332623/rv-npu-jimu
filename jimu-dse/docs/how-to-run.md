# NPU Closed-Loop Firmware Optimization — How to Run

> Start at the [documentation index](README.md). The [current project status](project-status.zh.md) records verified coverage and the boundary between proven L1/L2 transformations and blocked L3 work.

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
[START] canonical-baseline: jimu-dse/baseline/bert_layer.c
--- Iteration 1 ---
[PROBE] seq=2...  [BUILD] seq=2 OK
[PROBE] seq=6...  [BUILD] seq=6 OK
[PROBE] Generating micro-op DAG for agent analysis...
  seq=2: 2144B  seq=6: 7200B
  Baseline for this run set to 7200B
[AGENT] Invoking pi (timeout: 1800s)...
...
[VALIDATE] DRAM: 5856B (saved 1344B vs run-start 7200B)
...
===== Done =====
Baseline:   7200B
Best:       5856B
Improvement: 1344B (18.7%)
Continue: ./jimu-dse/scripts/npu_closed_loop.sh --start-from .../run-.../candidate_best.c
```

### 4. Inspect Results

```bash
# View the optimized candidate
ls jimu-dse/results/run-*/candidate_best.c

# View the DAG graphs for audit
ls jimu-dse/results/run-*/dag_iter1/

# View diff against this run's optimization_baseline.c snapshot
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
save-load roundtrips. `vrf-cache` 2.3 applies one independently accepted level
per iteration:

1. L1: intermediate save/load elimination;
2. L2: loop-invariant constant caching;
3. L3: optional weight-stationary scheduling.

Every candidate must provide a capacity/lifetime allocation proof. The G1
metric gate requires seq2 bytes not to increase, seq6 bytes to strictly
decrease, and seq6 instruction count to stay within the configured regression
limit (10% by default):

```bash
JIMU_INSTR_REGRESSION_LIMIT=0.10 \
  bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization \
  --validation-dim all \
  --agent opencode
```

The instruction-count policy can be disabled explicitly while still recording
the before/after counts:

```bash
# Environment switch
JIMU_INSTR_GATE=off bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --validation-dim all --agent opencode

# Equivalent command-line switch
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --validation-dim all --agent opencode \
  --instruction-gate off

# Backward-compatible shorthand
JIMU_INSTR_REGRESSION_LIMIT=off bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --validation-dim all --agent opencode
```

The DAG evidence gate is enabled by default. It requires the Agent to declare
one eligible PR4 L1 macro, or one primitive fallback when no macro is provable.
It verifies that the exact declared Tensor/address scope and measured seq2/seq6
traffic move in the claimed direction. Diagnostic runs can disable rejection
while retaining `dag_diff_N.json` and `dag_diff_N.md`:

```bash
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --agent opencode \
  --dag-evidence-gate off

# Equivalent environment switch
JIMU_DAG_EVIDENCE_GATE=off bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --agent opencode
```

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

# Configure and version agent skills:
make opencode

# This creates:
#   .opencode/skills/common-constraints/SKILL.md
#   .opencode/skills/dag-analyze/SKILL.md
#   .opencode/skills/vrf-cache/SKILL.md
#   .opencode/skills/dim-optimize/SKILL.md
#   jimu-dse/docs/skills/skills.lock.json
```

The skills are generated from `jimu-dse/docs/skills/isa/*.md` — single source
of truth. Run `make opencode` again if skills are updated.

The closed loop also runs this synchronization automatically before each run.
`skillctl.py` archives each semantic version, supports rollback, and rejects a
changed skill whose version was not bumped:

```bash
python3 jimu-dse/scripts/skillctl.py list
python3 jimu-dse/scripts/skillctl.py rollback vrf-cache 1.0.0
python3 jimu-dse/scripts/skillctl.py verify
```

Every run stores `skills_manifest.json` and embeds each effective skill's name,
version, and SHA256 in `run_manifest.json`. OpenCode receives all effective
skills explicitly; PI receives the generated `skills_bundle.md`.

To inspect exactly what an agent would receive without probing or modifying
firmware:

```bash
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal combined \
  --agent opencode \
  --validation-dim all \
  --prepare-only
```

The generated skill order is always `common-constraints`, `dag-analyze`, the
goal-specific skills, then `self-verify`.

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
| **PROBE seq=6** | Measures seq_len=6, generates concrete seq2/seq6 DAG evidence for agent |
| **ANALYZE** | Reports DRAM ratio seq6/seq2 and cluster breakdown |
| **CONVERGE** | Stops early if improvement < 15% vs run-start baseline |
| **AGENT** | Invokes pi with DAG + DRAM + skills, pi patches bert_layer.c |
| **VALIDATE** | Rebuilds firmware with candidate, re-measures DRAM |
| **DAG** | Generates post-optimization DAG graphs for audit |
| **DEPLOY** | Saves candidate, repeats from iteration 2 with optimized code |

### 3. Continue from a Previous Result

```bash
# Continue from a run's candidate_best.c
bash jimu-dse/scripts/npu_closed_loop.sh \
  --start-from jimu-dse/results/run-20260730-141105-154854

# Or continue from one exact accepted iteration
bash jimu-dse/scripts/npu_closed_loop.sh \
  --start-from jimu-dse/results/run-20260730-141105-154854/candidate_4.c
```

`--resume <run-dir>` remains a compatibility alias for selecting that
directory's `candidate_best.c`. The run output prints a ready-to-use
`--start-from` command at the end.

### 4. Run the Validation Test Suite

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

---

## Run Output Structure

Each run creates a timestamped directory:

```
jimu-dse/results/run-<YYYYMMDD>-<HHMMSS>-<PID>/
├── optimization_baseline.c ← Immutable snapshot selected for this run
├── candidate_best.c       ← Best optimized firmware (use with --start-from)
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
├── diff_1.patch           ← Diff against optimization_baseline.c (iteration 1)
├── diff_2.patch           ← Diff against optimization_baseline.c (iteration 2)
├── p2_probe.json          ← DRAM probe result for seq=2
├── p6_probe.json          ← DRAM probe result for seq=6
└── val_1.json             ← Validation result for iteration 1
```

To view a DAG as SVG (requires graphviz):

```bash
dot -Tsvg jimu-dse/results/run-*/dag_iter1/micro_op_dag.dot -o /tmp/dag.svg
```

Each `dag_agent/`, `dag_before_iterN/`, and `dag_iterN/` uses seq6 as the
authoritative root DAG and stores the short trace under `seq2/`. DAG-PR5/PR6 add:

- `multiseq_metadata.json`: binds both concrete configurations and ELF hashes;
- `multiseq_summary.md`: readable L1/L2/L3 opportunity ranking;
- `loop_invariants.json`: exact cross-sequence reuse counts and upper bounds;
- `candidate_evidence.jsonl`: compact records and representative nodes passed
  to the agent. Full JSONL DAGs remain on disk for the independent gate.
- `proof/dim*-h*-head*-seq*/`: every additional correctness configuration;
- `allocation_summary.md`: compact matrix, proof digests and reference regions
  explicitly attached to the agent;
- `allocation_proof.json`: complete L1/L2 capacity, alignment and lifetime
  proof across dim2-h4, dim4-h4 and dim4-h8 at seq2/seq6.

Only macros whose allocation has `cross_config_proven=true` and
`validation_matrix_complete=true` may be selected. L1 must finish before L2.
L3 remains blocked until schedule, MRF residency, partial sums and FP16 order
also have an independent proof.

---

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `JIMU_MAX_ITER` | 5 | Max optimization iterations per run |
| `JIMU_THRESHOLD` | 0.15 | Convergence threshold (fraction of baseline). Stop when improvement < 15% |
| `JIMU_AGENT_TIMEOUT` | 1800 | Timeout in seconds for each agent invocation (pi or opencode) |
| `OPENCODE_MODEL` | `opencode/big-pickle` | OpenCode model in provider/model format |
| `CC` | `riscv64-unknown-elf-gcc` | RISC-V cross-compiler |

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--goal <name>` | `dram-optimization` | Optimization goal: `dram-optimization`, `compute-optimization`, `combined` |
| `--agent pi` | pi | Agent to use: `pi` or `opencode` |
| `--model <provider/model>` | `$OPENCODE_MODEL` | OpenCode model (overrides env var) |
| `--start-from <path>` | canonical baseline | Start from a `.c` file or run directory |
| `--resume <dir>` | — | Compatibility alias for a run directory's `candidate_best.c` |

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

Typical unoptimized baseline DRAM after measuring the real seq-specific ELF:

| Metric | seq=2 | seq=6 |
|--------|-------|-------|
| V_RD_DRAM elements | 240 | 912 |
| V_WR_DRAM elements | 104 | 312 |
| M_RD_DRAM ops | 144 (constant) | 144 (constant) |
| Total bytes | 2,144 | 7,200 |

After VRF cache optimization, typical results:

| Iteration | seq=6 DRAM | Savings vs baseline |
|-----------|-----------|-------------------|
| 0 (canonical baseline) | 7,200B | — |
| K/V cache | 5,856B | 18.7% |
| + Q cache | 5,664B | 21.3% |
| + Z/SO/LN1/GELU cache | 4,896B | 32.0% |

---

## Baseline File

The known-correct unoptimized firmware references are stored in
`jimu-dse/baseline/`, for example `bert_layer.c` and `adder_140p.c`.
They are canonical references and must not be replaced by optimized results.

Fresh runs start from the canonical reference. Continued runs use
`--start-from`; the selected source is frozen as
`optimization_baseline.c`, and all metrics and diffs in that run are relative
to that snapshot. At exit, the working firmware is restored to the canonical
unoptimized source.

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
| dag-analyze | `jimu-dse/docs/skills/isa/dag-analyze.md` | Compare seq2/seq6 DAGs and select proved L1 candidates |
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

---

## Workspace Cleanup

The cleanup command is dry-run by default and preserves virtual environments and all run history:

```bash
# Preview verified build/cache/backup targets.
bash jimu-dse/scripts/clean_workspace.sh

# Remove reproducible outputs.
bash jimu-dse/scripts/clean_workspace.sh --apply

# Optionally remove one exact temporary run.
bash jimu-dse/scripts/clean_workspace.sh --apply \
  --run-dir jimu-dse/results/run-<timestamp>
```

Run `make kernels` again before tests after cleanup. Promote durable conclusions to `jimu-dse/docs/reports/`; do not treat a run-local `candidate_best.c` as the canonical baseline.
