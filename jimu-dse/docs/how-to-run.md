# Configurable Closed-Loop Firmware Optimization

The driver implements `PROBE → PROMPT → AGENT → GATE → SCORE → PROMOTE → LOOP`.
Goals, skills, acceptance, scoring, and convergence are declared in
`jimu-dse/goals/<name>/goal.yaml`.

## Install and inspect

```bash
pip install -r requirements.txt
make kernels
python3 jimu-dse/scripts/closed_loop.py list-goals
python3 jimu-dse/scripts/closed_loop.py validate-config --goal dram-optimization
python3 jimu-dse/scripts/closed_loop.py render-prompt --goal dram-optimization
```

`render-prompt` is a dry run: it validates configuration and renders skills
without invoking an agent or changing firmware.

## Run and resume

```bash
# Compatibility entry point
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --agent opencode

# Direct CLI
python3 jimu-dse/scripts/closed_loop.py run \
  --goal combined --agent opencode --model provider/model

python3 jimu-dse/scripts/closed_loop.py inspect-run \
  jimu-dse/results/run-YYYYMMDD-HHMMSS-PID

python3 jimu-dse/scripts/closed_loop.py run --goal combined \
  --resume jimu-dse/results/run-YYYYMMDD-HHMMSS-PID
```

Long runs print timestamped progress messages to stderr. The messages identify
the resolved run directory, baseline and per-iteration probes, Agent start and
20-minute heartbeat, changed-file count, each acceptance gate, score,
promotion, checkpoints, and final stop reason. Subprocess excerpts remain
bounded in the JSON artifacts instead of flooding the terminal; the complete
Agent streams are retained as `agent-N.stdout.jsonl` and
`agent-N.stderr.log`. Use `--quiet` to suppress progress while retaining the
final report.

The shell wrapper and Makefile prefer `.venv/bin/python` (or the Windows
`.venv/Scripts/python.exe`) when it exists, then fall back to `python3`. This
same PATH is inherited by command gates, so `python3 -m pytest` does not
silently select a dependency-free system interpreter. The driver also searches
`~/.npm-global/bin` and `~/.local/bin` when locating Agent CLIs in a
non-interactive shell.

Explicit loop controls are available without editing YAML:

```bash
# Run exactly 10 iterations unless an infrastructure/agent-start failure occurs.
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal cycle-latency-optimization --agent opencode \
  --max-iterations 10 --full-iterations

# Additionally disable the per-iteration Agent timeout and advisory work budget.
# Build, probe, and correctness-gate safety timeouts remain enabled.
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal cycle-latency-optimization --agent opencode \
  --max-iterations 10 --full-iterations --agent-timeout 0
```

`--full-iterations` ignores `target_score` and the consecutive-no-promotion
limit. It does not override unrecoverable failures, missing Agent executables,
provider rate limits, or failed baseline infrastructure.

Resume loads `candidate_best.c` and requires the selected goal's fully resolved
configuration fingerprint to match. This prevents accidentally continuing a run
with different hardware, gates, skills, or scoring.

## Configuration precedence

Highest to lowest:

1. Explicit CLI values (`--agent`, `--model`, `--max-iterations`,
   `--agent-timeout`).
2. Compatibility environment variables (`JIMU_MAX_ITER`,
   `JIMU_AGENT_TIMEOUT`, `OPENCODE_MODEL`).
3. Values in `goal.yaml`.

Every run stores the final value set in `resolved-config.yaml`.

## `goal.yaml` reference

All roots and sections reject unknown fields. Every repository path is resolved
against the repository root and may not escape it.

| Section | Fields |
|---|---|
| root | `schema_version`, `name`, `description` |
| `target` | `firmware`, `baseline`, `allowed_files`, exact `hardware` values, `sequence_lengths` |
| `agent` | `backend` (`pi`/`opencode`), `model`, `timeout_seconds` (`0` disables the Agent hard timeout), `context_files` |
| `prompt` | `template`, `goal`, `constraints`, `self_verify` |
| `skills` | ordered `{name, path}` entries |
| `probe` | built-in `metrics`, `cycle_limit`, `dag.enabled`, optional scoring sequence, cost model, and cycle-model profile |
| `acceptance.gates` | named `allowed_files`, `build`, `probe`, or `command` gates |
| `acceptance.score` | `{metric, direction, weight, target?}` entries; weights sum to `1.0` |
| `loop` | `max_iterations`, `min_score_delta`, `max_no_improvement`, optional `target_score` |
| `artifacts` | switches for candidates, diffs, prompts, probes, and graphs |

Built-in metrics are `total_bytes`, `dram_elements`,
`functional_container_bytes`, `rtl_payload_bytes`, `instr_count`, `mv_mul_count`,
`mat_rd_ops`, `test_pass`, `memory_access_count`, `memory_read_count`,
`memory_write_count`, `register_access_count`, `register_read_count`,
`register_write_count`, and `estimated_time`. A score metric must also appear
in `probe.metrics`; directions are `minimize` or `maximize`.

Optional SCALE-Sim metrics are `scalesim_layer_count`,
`scalesim_compute_cycles`, `scalesim_stall_cycles`, `trace_memory_cycles`,
`auxiliary_cycles`, and `predicted_npu_cycles`. The parallel backend also
provides `parallel_predicted_npu_cycles`, `overlap_saved_cycles`,
`memory_compute_overlap_cycles`, `max_concurrent_ops`, resource utilizations,
and `schedule_chain_count`. Requesting one requires:

```yaml
cycle_model:
  profile: jimu-dse/timing/scalesim-parallel-dim4.yaml
```

The schema-v3 Verilator backend provides `rtl_predicted_npu_cycles`, controller
and DRAM utilization, memory/compute overlap, concurrency, and RTL counters for
front-end, dependency, unit, DRAM, bank, and fence stalls:

```yaml
cycle_model:
  profile: jimu-dse/timing/jimu-rtl-dim4.yaml
```

For RTL schedules, `rtl_predicted_npu_cycles` and
`rtl_completion_makespan_cycles` end at the last operation completion;
`rtl_idle_cycles` includes the in-order retirement tail. The compatibility
metric `overlap_saved_cycles` means net savings versus serial command duration.
Use `gross_overlap_cycles`, `scheduler_idle_hole_cycles`, and
`memory_compute_overlap_cycles` to distinguish overlapping work, inactive
schedule holes, and actual DRAM/compute intersection. RTL stall counters are
non-additive pressure indicators, not independent cycle penalties.

`total_bytes` is the legacy float32 emulator-container byte count. For a
representation-neutral comparison use `dram_elements`; for modeled bus bytes
use `rtl_payload_bytes`.

Command gates support:

```yaml
- name: numerical-correctness
  type: command
  command: python3 -m pytest tests/integration/test_bert_e2e.py -k "dim{dim} and h{hidden}" -q
  timeout_seconds: 600
  success_codes: [0]
```

Command templates expose `{dim}`, `{hidden}`, and `{firmware}`.

## Prompt templates

The following fields are allowed:

| Variable | Content |
|---|---|
| `{goal_name}`, `{goal_description}` | Goal identity and optimization intent |
| `{iteration}`, `{target_file}`, `{hardware}` | Current execution context |
| `{metrics}`, `{clusters}` | Latest probe results and DAG cluster text |
| `{cost_model}` | Configured memory/register weights and resource scope |
| `{skills}` | Full ordered skill instructions |
| `{constraints}`, `{self_verify}` | Goal constraints and verification guidance |
| `{gate_commands}` | Configured gate commands/types |

Unknown variables fail configuration validation before a run starts.

## Acceptance and convergence

A candidate is never promoted unless every hard gate passes. Eligible
candidates are scored relative to the run-start baseline:

- minimize: `(baseline - candidate) / abs(baseline)`
- maximize: `(candidate - baseline) / abs(baseline)`
- total score: sum of normalized improvements multiplied by their weights

The candidate becomes the new best only when its score exceeds the best score
by at least `min_score_delta`. Otherwise the working firmware is restored to the
current best. The loop stops at `target_score`, after
`max_no_improvement` unsuccessful rounds, or at `max_iterations`.
The repository's original firmware is restored when the run exits.

## Weighted memory/register cost

The `weighted-latency-optimization` goal minimizes a dimensionless estimate:

```text
estimated_time =
    memory_access_count × memory_weight
  + register_access_count × register_weight
```

It is a reproducible comparison metric, not cycle-accurate hardware time.
Its default configuration is:

```yaml
probe:
  scoring_sequence_length: 6
  metrics:
    - memory_access_count
    - register_access_count
    - estimated_time
  cost_model:
    memory_weight: 10
    register_weight: 1
    register_resources: [VRF, MRF, SRF, REG]
```

Memory reads and writes are the executed NPU vector/matrix DRAM operations;
each instruction counts once regardless of transfer width. Register reads are
selected-resource entries in EventTracer `uses`, and writes are entries in
`defs`. Pipeline temporaries, RISC-V GPRs, and CPU memory operations are
excluded. The JSON and Markdown reports retain weights and the complete
read/write breakdown.

Weights must be non-negative. `scoring_sequence_length` must be one of
`target.sequence_lengths`; goals without this field continue to score the last
configured sequence. Requesting `estimated_time` requires a `cost_model`.

```bash
python3 jimu-dse/scripts/closed_loop.py validate-config \
  --goal weighted-latency-optimization
python3 jimu-dse/scripts/closed_loop.py render-prompt \
  --goal weighted-latency-optimization
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal weighted-latency-optimization --agent opencode
```

## SCALE-Sim cycle model

The cycle goal can also enable the native lock-step timing device and a
workload manifest:

```yaml
probe:
  workload_manifest: jimu-dse/workloads/bert-dim4-seq6.yaml
  timed_device:
    profile: jimu-dse/timing/npu-timed-v1.yaml
```

This adds BUSY/DONE/CHAIN_STATUS polling, finite FIFO capacity, scoreboarding,
unit pipelines, DRAM contention, source provenance, and tensor identities to
the same probe. Goals for non-BERT firmware may declare `target.build.command`,
`target.build.elf`, `target.build.cwd`, and optional environment variables;
template fields include `firmware`, `elf`, `dim`, `hidden`, `num_head`,
`seq_len`, and `repo`.

Install the optional, pinned timing backend:

```bash
make timing-deps
```

The `cycle-latency-optimization` goal translates every executed `MV_MUL` into
a small GEMM for SCALE-Sim v2.0.2. It retains the legacy additive estimate:

```text
predicted_npu_cycles =
    scalesim_compute_cycles
  + trace_memory_cycles
  + auxiliary_cycles
```

SCALE-Sim stall cycles remain visible as a diagnostic but are excluded by
default because the firmware trace already includes explicit memory
instructions.

The schema-v2 profile then schedules the same dynamic trace on a single DRAM
bus and one VMM, MMM, MVU, and SPU. The scheduler enforces RAW, WAR, WAW,
overlapping-address and structural-resource hazards. It permits independent
DRAM transfers and computation to overlap:

```text
parallel_predicted_npu_cycles = end cycle of the scheduled critical path
overlap_saved_cycles = predicted_npu_cycles - parallel_predicted_npu_cycles
```

Instructions enter a two-entry queue in trace order at one instruction per
cycle. Explicit `INST_ISSUE` operations end a chain and form a completion
barrier. Because the current BERT firmware has no `INST_ISSUE`, its complete
trace is treated as one compatible implicit ordered stream. Cross-chain SMC is
not modeled. The versioned profile is
`jimu-dse/timing/scalesim-parallel-dim4.yaml` and is outside the agent's allowed
files.

```bash
python3 jimu-dse/scripts/closed_loop.py validate-config \
  --goal cycle-latency-optimization
python3 jimu-dse/scripts/closed_loop.py render-prompt \
  --goal cycle-latency-optimization
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal cycle-latency-optimization --agent opencode
```

The final report compares SCALE-Sim layers, legacy additive cycles, scheduled
cycles, overlap, concurrency, and per-resource utilization. Every timed probe
also writes `timing-schedule.json`; each record contains start/end cycles,
resources, dependency reasons, queue wait, chain ID, DRAM range, and
critical-path status. The file also contains `optimization_diagnostics`, a
bounded summary of the longest critical events, largest waits, wait reasons,
and resource-utilization ranking. That summary and the exact schedule path are
included in every scored Agent prompt; the complete event array remains an
audit artifact instead of consuming prompt space. Dynamic raw instruction
indexes remain available. When the ELF contains DWARF line data, command
events and cross-layer evidence additionally contain the issuing C source
file, line, function, and PC.
The metric block is likewise grouped into the scored value, its delta from the
run baseline, actionable overlap/utilization diagnostics, and the legacy model
breakdown. Layer counts, diagnostic stall counts, and chain counts remain in
the JSON artifacts unless they directly affect a score.

The prompt and `cycle-latency` Skill define the exact operation-to-resource
mapping. Different units may overlap only when the events have no RAW, WAR,
WAW, configuration-fence, DRAM-range, queue-window, or chain-boundary conflict.
All DRAM transfers still serialize on the shared `dram_bus`. This remains a
resource-level analytical model rather than native RTL cycle accuracy. See
`timing-simulator-selection.md` for the model boundary and calibration path.

### Verilator RTL scoring goal

`rtl-cycle-optimization` runs the same functional/timed probe, then replays its
complete dynamic command trace through the synthesizable RTL timing core.  It
scores `rtl_predicted_npu_cycles` and writes the RTL schedule to the normal
`timing-schedule.json` contract, including exact dependency resources and bank
stall attribution.

```bash
python3 jimu-dse/scripts/closed_loop.py validate-config \
  --goal rtl-cycle-optimization
python3 jimu-dse/scripts/closed_loop.py render-prompt \
  --goal rtl-cycle-optimization
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal rtl-cycle-optimization --agent opencode
```

Use this goal as the promotion tier for firmware scheduling/data-flow changes.
The existing SCALE-Sim goal remains useful for faster analytical exploration.
Verilator must be installed locally or in the profile's configured WSL distro.
The profile and RTL are outside the agent's allowed files.

## Creating a goal

1. Copy an existing directory under `jimu-dse/goals/`.
2. Change `name`, description, target hardware, prompt, ordered skills, gates,
   score weights, and loop thresholds.
3. Ensure every referenced path exists and all weights sum to `1.0`.
4. Run `validate-config`, then inspect `render-prompt`.
5. Start with `JIMU_MAX_ITER=1` and review the generated report before a longer run.

No Python or shell driver change is required for a new declarative goal.

## Run artifacts

Each `jimu-dse/results/run-*` directory contains:

- `resolved-config.yaml` and `baseline-probe.json`
- `prompt-N.txt`, `candidate-N.c`, `diff-N.patch`, and graph directories
- `baseline/timing-schedule.json`, `pre-iteration-N/timing-schedule.json`, and
  `iteration-N/timing-schedule.json` for schema-v2 cycle runs
- `iteration-N.json` with agent status, gates, raw metrics, score, and promotion
- `agent-N.stdout.jsonl` and `agent-N.stderr.log` with the complete Agent output
- `candidate_best.c`
- `run-summary.json` and human-readable `report.md`

Agent unavailable, timeout, no change, build failure, probe failure, gate
failure, and lack of score improvement are represented as distinct statuses.
When an Agent makes no change, the acceptance gates are reported as `SKIPPED`,
not as failures. On WSL, the driver exports the worktree's resolved `GIT_DIR`
and `GIT_WORK_TREE` so an Agent can inspect a worktree whose gitfile was created
by Windows Git.

`make clean` removes only rebuildable artifacts and deliberately preserves
closed-loop results. Use `make clean-results` only when no run is active and
you explicitly intend to remove every directory under `jimu-dse/results`.
Agents are forbidden from running either repository-root cleanup command or
modifying an active run directory. If an external process still removes the
directory, the executor stops with `run_artifacts_lost` and recreates a minimal
recovery package containing the resolved configuration, iteration records, and
the last validated `candidate_best.c`.

## Troubleshooting

- `No module named yaml`: run `pip install -r requirements.txt`.
- Missing cross compiler: install `riscv64-unknown-elf-gcc` and set `CC` if needed.
- Invalid goal: run `validate-config`; it reports the exact field or path.
- Candidate rejected: inspect the relevant `iteration-N.json` gate output.
- Resume rejected: use the same resolved goal and overrides as the original run.
- `run_artifacts_lost`: inspect `artifact-recovery.json`; restore or archive the
  preserved best candidate before starting another run.
