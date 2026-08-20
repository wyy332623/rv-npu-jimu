# RTL Timing Simulator

## Purpose and scored boundary

The RTL backend exists to evaluate firmware/data-flow transformations that a
serial or additive instruction model cannot distinguish.  It models command
timing and control at RTL clock granularity while the existing C/Python device
remains the numerical oracle.

The primary metrics are:

- last-operation completion makespan (`rtl_predicted_npu_cycles`, also exposed
  explicitly as `rtl_completion_makespan_cycles`);
- fully-idle time (`rtl_idle_cycles`) and the in-order retirement tail between
  completion and idle (`rtl_retirement_tail_cycles`);
- DRAM/compute overlap cycles and overlap ratio;
- per-controller utilization and maximum concurrent operations;
- front-end, dependency, initiation-interval, DRAM, SRAM-bank, and fence stalls;
- event-level enqueue/start/finish cycles, predecessors, source, and tensors.

`overlap_saved_cycles` is retained as a compatibility alias for
`net_parallelism_savings_cycles = serial_command_cycles - completion_makespan`.
It is not the number of physically overlapping cycles.  The decomposition is:

```text
gross_overlap_cycles = sum(command durations) - union(active intervals)
scheduler_idle_hole_cycles = completion_makespan - union(active intervals)
net_parallelism_savings_cycles = gross_overlap_cycles - scheduler_idle_hole_cycles
```

Use `memory_compute_overlap_cycles` for the actual intersection of DRAM and
compute activity.

Functional correctness is still gated by observable DRAM regions from the
workload manifest.  Changing the RTL profile cannot make a numerically wrong
firmware candidate pass.

## Open-source architecture study

Three mature open designs were considered:

| Design | Useful evidence | Decision |
|---|---|---|
| [Apache VTA](https://github.com/apache/tvm-vta) | Four FIFO-connected modules and decoupled load/compute/store pipelines; Apache-2.0 | Adopt the task-level pipeline structure and explicit dependency idea. |
| [Berkeley Gemmini](https://github.com/ucb-bar/gemmini) | Decoupled load/store/execute controllers, ROB hazard management, DMA, and banked scratchpad; BSD-3-Clause | Adopt the finite ROB, cross-controller scheduling, and bank/port contention model. |
| [NVDLA](https://github.com/nvdla/hw) | Production-scale Verilog/C-model/testbench, independent DMA engines, multi-stage pipelines, and ping-pong register groups | Use as evidence for DMA/compute concurrency and double buffering, but do not integrate it: its layer descriptors and convolution pipeline do not match the Jimu firmware ISA. |

The implementation is original SystemVerilog aligned with Jimu commands; no
source from these projects is copied.  The architectural choices follow the
official descriptions of [VTA's decoupled access-execute modules](https://tvm.apache.org/2018/07/12/vta-release-announcement.html),
[Gemmini's controllers/ROB/scratchpad](https://github.com/ucb-bar/gemmini#major-components),
and [NVDLA's DMA pipelines and ping-pong stages](https://nvdla.org/hw/v1/hwarch.html).

## Implemented microarchitecture

```text
firmware ELF -> functional emulator/event trace -> semantic token encoder
                                                    |
                                                    v
                     +---------------- finite ROB / scoreboard ----------------+
                     | oldest-ready issue, same-controller order, chain fence |
                     +--------+---------+---------+---------+------------------+
                              |         |         |         |
                           LOAD       STORE      MVU      VECTOR      CONTROL
                              \         /         \         /
                               shared DRAM       banked local SRAM
                                      \             /
                                       RTL counters
```

`rtl/jimu_npu_timing_core.sv` is synthesizable and is compiled by Verilator.
It contains:

- a configurable finite ROB and one-command-per-cycle oldest-ready dispatcher;
- five independent controllers: load, store, MVU, vector, and control;
- exact older-command RAW/WAR/WAW checks over encoded semantic resources;
- in-order issue within each controller and out-of-order issue across controllers;
- a single shared DRAM bus, with load/store serialized for their transfer duration;
- configurable pipelined unit initiation intervals;
- banked local SRAM with one read and one write stream per bank;
- full chain fences for `INST_ISSUE` in the conservative profile;
- cycle counters and optional VCD waveform generation.

The RTL stall counters are pressure diagnostics for the oldest blocked ROB
entry.  A younger independent command can dispatch during the same cycle, and
different counter deltas may be consequences of the same schedule change.
They therefore must not be summed as if they were independent makespan losses.

The trace encoder converts `pipe` and `vpipe_a` into SSA-like elastic tokens.
This preserves true producer-consumer edges without introducing a false global
WAW edge on every pipeline value.  VRF, MRF, SRF, configuration registers, and
granular DRAM ranges remain physical resources.  Scoreboard bits are not reused
within one ROB window; any forced collision is reported in
`resource_encoding.conservative_hash_collisions`.

## Repository integration

The RTL backend is connected to the existing project at four levels:

| Level | Entry point | Responsibility |
|---|---|---|
| RTL | `rtl/jimu_npu_timing_core.sv` | ROB, controller, dependency, DRAM, SRAM-bank, fence, and cycle-counter behavior |
| Replay adapter | `emulator/npu_rtl_sim.py` and `sim/jimu_rtl_harness.cpp` | Encode trace resources, build/run Verilator, and emit the schedule contract |
| Firmware analysis | `scripts/analyze_firmware.py --rtl-profile ...` | Run functional equivalence, trace extraction, RTL replay, and cross-layer graph generation |
| Agent loop | `jimu-dse/goals/rtl-cycle-optimization/goal.yaml` | Restrict firmware edits, apply correctness gates, and score `rtl_predicted_npu_cycles` |

The shorter project-level descriptions live in `README.md`,
`jimu-dse/docs/how-to-run.md`, and
`jimu-dse/docs/timing-simulator-selection.md`.  This document is the canonical
operational and architecture reference for the RTL backend.

## Environment

Run the RTL flow from Linux or WSL.  The repository Makefile uses POSIX tools
such as `make`, `ln`, `rm`, and `find`; invoking it directly from PowerShell is
not supported.  A typical WSL setup is:

```bash
cd /mnt/d/PersonalPrograms/jimu-3/rv-npu-jimu-3-codex-20260817
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt -r requirements-timing.txt
```

Verilator, CMake, a native C compiler, and `riscv64-unknown-elf-gcc` must be on
`PATH`.  The supplied RTL profile can also invoke Verilator through its
configured WSL distribution when the Python entry point is started on Windows.

Shell entry points are forced to LF by the repository `.gitattributes`.  If
Bash reports `set: pipefail: invalid option name`, inspect the checkout with:

```bash
file jimu-dse/scripts/npu_closed_loop.sh
```

The result must not say `with CRLF line terminators`.

## Running firmware analysis

Build firmware and kernels, then run unified analysis:

```bash
make kernels firmware
python3 scripts/analyze_firmware.py \
  --manifest jimu-dse/workloads/bert-dim4-seq6.yaml \
  --rtl-profile jimu-dse/timing/jimu-rtl-dim4.yaml \
  -o _out/bert-rtl --no-render
```

Add `--rtl-wave` to emit `_out/bert-rtl/rtl-wave.vcd`.  For an existing trace:

```bash
python3 scripts/simulate_rtl.py \
  --events _out/firmware-analysis/trace-events.json \
  --manifest jimu-dse/workloads/bert-dim4-seq6.yaml \
  --profile jimu-dse/timing/jimu-rtl-dim4.yaml \
  -o _out/firmware-analysis/rtl-timing-schedule.json
```

Useful generated artifacts are:

- `rtl-timing-schedule.json`: full RTL schedule, counters, blockers, and profile;
- `timing-schedule.json`: generic alias consumed by data-flow mining tools;
- `rtl-commands.txt`: exact packets presented to RTL;
- `rtl-harness-schedule.csv`: raw RTL/harness observations;
- `cross-layer-graph.json`: tensors, commands, exact dependencies, timing, source;
- `run-summary.json`: functional equivalence plus aggregate RTL metrics;
- `rtl-wave.vcd`: optional signal-level waveform.

## Manual verification

The focused RTL tests are executable examples, not only regression gates:

| Test file | Behavior demonstrated |
|---|---|
| `tests/unit/test_npu_rtl_sim.py` | SSA token encoding, independent DRAM/compute overlap, true dependency blocking, SRAM read-port conflicts, and graph dependency transfer |
| `tests/unit/test_npu_rtl_optimization_space.py` | MRF ping-pong prefetch, elimination of DRAM materialization, and VRF bank rotation |

Run the synthesizable RTL lint and all focused examples with:

```bash
make rtl-lint
make rtl-test
```

To isolate one class of behavior:

```bash
python3 -m pytest tests/unit/test_npu_rtl_sim.py -v
python3 -m pytest tests/unit/test_npu_rtl_optimization_space.py -v
```

Run the command above to establish the completion and fully-idle values for the
exact firmware/profile revision under test; these values intentionally are not
treated as permanent constants after firmware, RTL, or profile changes.

## RTL-scored agent loop

Validate the goal and inspect the exact prompt before allowing an agent to edit
firmware:

```bash
python3 jimu-dse/scripts/closed_loop.py validate-config \
  --goal rtl-cycle-optimization
python3 jimu-dse/scripts/closed_loop.py render-prompt \
  --goal rtl-cycle-optimization
```

Start with one iteration:

```bash
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal rtl-cycle-optimization \
  --agent opencode
```

The goal permits changes only to `firmware/bert/bert_layer.c`.  The RTL,
profile, simulator adapter, tests, artifacts, and scoring implementation are
outside the agent's allowed files.  Each run is stored below
`jimu-dse/results/run-*`; review `baseline-probe.json`, `probe-1.json`,
`diff-1.patch`, `iteration-1.json`, `report.md`, and the per-probe
`timing-schedule.json` before increasing `JIMU_MAX_ITER`. Complete Agent output
is preserved in `agent-N.stdout.jsonl` and `agent-N.stderr.log`; the JSON report
contains bounded excerpts. A `no_change` iteration deliberately skips gates and
shows them as `SKIPPED`.

The shell wrapper is only a compatibility entry point.  The equivalent direct
driver command is:

```bash
python3 jimu-dse/scripts/closed_loop.py run \
  --goal rtl-cycle-optimization \
  --agent opencode \
  --max-iterations 1
```

## Optimization space exposed to agents

The model intentionally makes the following transformations measurable:

| Transformation | Expected schedule evidence |
|---|---|
| Hoist/prefetch a DRAM load | Load moves into an independent MVU/vector interval; overlap rises. |
| Keep weights or intermediates resident | Fewer DRAM events and boundary bytes; makespan and bus utilization fall. |
| Double-buffer or rename scratch storage | WAR/WAW edges and dependency stalls disappear, subject to bank pressure. |
| Rotate/partition VRF banks | Bank stalls fall without changing data dependencies. |
| Software-pipeline tiles/positions | ROB concurrency and overlap rise; controller holes shrink. |
| Fuse epilogues or reductions | Pipeline tokens and materializations disappear; vector/DRAM work falls. |
| Hoist invariant setup | Repeated control events and fence-adjacent waits disappear. |
| Batch INC operations or widen chains | Front-end/fence overhead falls, unless queue or bank pressure becomes limiting. |
| Narrow/elide a proven redundant fence | Barrier stalls fall and cross-group overlap becomes visible. |

Every hypothesis must still prove observable equivalence, storage capacity,
alias safety, bank feasibility, and numerical legality.  A missing graph edge is
not evidence that reordering is safe.

## Diagnostic and byte-count boundaries

`critical_path_top_events` is a post-hoc causal predecessor chain ending at the
latest completion, not a formal zero-slack critical-path calculation.
`critical_path_top_blockers` is filtered strictly to events on that chain;
`top_blockers` remains the global pressure ranking.

The functional emulator stores values in NumPy `float32` containers, while the
default RTL timing profile models an FP16 payload. Closed-loop probes therefore
report three separate quantities:

- `dram_elements`: transfer element count, independent of representation;
- `functional_container_bytes`: element count times four;
- `rtl_payload_bytes`: element count times the timing profile's
  `memory_element_bytes` (two in the default profile).

Legacy `total_bytes` remains an alias of `functional_container_bytes` so old
goals and run histories remain comparable; it must not be described as RTL bus
traffic. Matrix DRAM events use
`(REG_TILE_ROWS * native_dim)^2` elements, matching the functional device, and
record `tile_rows` in each event.

## Current limitations and calibration

This is a command/control RTL co-simulator, not yet a bit-accurate FP16 datapath
or full SoC RTL simulation.  In particular:

- the RISC-V ISS and numerical kernels are not inside Verilator;
- trace replay is offline, so RTL FIFO backpressure does not yet change CPU poll
  instruction counts (the lock-step Python device remains available for that);
- local SRAM bank masks are derived from Jimu resource/address metadata, and a
  command conservatively holds its ports for its modeled duration;
- default latency/bandwidth values are architectural hypotheses, not silicon
  measurements; compare candidates only under the same versioned profile;
- a full `INST_ISSUE` fence is conservative and may understate legal cross-chain
  overlap.

The next fidelity steps are an RTL MMIO/FIFO wrapper connected to the ISS,
AXI-like request/response timing, explicit ping-pong MRF addressing, calibrated
MVU/vector pipelines, and optional DPI calls to the functional kernels.  These
can be added without changing the schedule/cross-layer artifact contract.
