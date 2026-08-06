---
name: cycle-latency
version: 2.1.0
category: performance-model
description: Minimize SCALE-Sim-backed NPU cycle estimates on the critical firmware trace.
compatibility:
  schema_version: 1
---

# Parallel SCALE-Sim Cycle Latency Optimization

## Model Boundary

The cycle score is a hybrid performance model:

- SCALE-Sim v2.0.2 models each executed `MV_MUL` as a `1×DIM · DIM×DIM`
  weight-stationary GEMM on a DIM×DIM systolic array.
- The actual EventTracer sequence supplies DRAM transfers, data hazards, chain
  boundaries, and all non-MVU instructions.
- Explicit trace memory cycles use the configured transfer bandwidth and
  setup latency.
- Auxiliary operations use the versioned timing profile.
- A bounded scoreboard scheduler assigns traced operations to a single shared
  DRAM bus plus VMM, MMM, MVU, and SPU resources. Independent memory and
  compute operations may overlap; RAW, WAR, WAW, structural conflicts, and
  overlapping DRAM ranges prevent illegal overlap.

This is more architectural than an access-count proxy, but it is not the
cycle-accurate RTL for this custom NPU. SCALE-Sim stall cycles are diagnostic
by default because the firmware trace already contains explicit memory
operations; including both would double count memory cost.

## Resource and overlap rules

Use these exact mappings when deciding whether a source reordering can expose
parallel work:

| Operation class | Occupied resources | May overlap when independent |
| --- | --- | --- |
| `V_RD_DRAM`, `V_WR_DRAM` | `dram_bus + vmm` | MVU, MMM, or SPU work |
| `M_RD_DRAM`, `M_WR_DRAM` | `dram_bus + mmm` | MVU, VMM, or SPU work |
| `MV_MUL` | `mvu` | DRAM, VMM, MMM, or SPU work |
| `M_RD`, `M_WR` | `mmm` | VMM, MVU, vector DRAM, or SPU work |
| Scalar/configuration operations | `spu` | Independent non-SPU work |
| Other vector and activation operations | `vmm` | MVU, MMM, matrix DRAM, or SPU work |

Different resource names do not by themselves make a pair safe. The scheduler
also rejects overlap across RAW, WAR, WAW, overlapping DRAM address ranges,
configuration fences, queue-window limits, and explicit chain boundaries. All
DRAM transfers serialize on the single `dram_bus`. `S_WR` is a conservative
configuration fence. Use the critical-event predecessor reasons in the scored
schedule instead of guessing that two nearby source statements are independent.

## Optimization Strategy

1. Reduce the number of executed `MV_MUL` tile operations without changing the
   mathematical result.
2. Eliminate explicit vector and matrix DRAM round trips on the critical path.
3. Retain and reuse weights or intermediates in VRF/MRF when capacity and
   lifetimes permit.
4. Move independent DRAM transfers earlier so their latency overlaps useful
   MVU or vector work, without overwriting live pipe, MRF, or VRF values.
5. Avoid clustering unrelated transfers on the single DRAM bus; reduce the
   scheduled critical path and improve MVU utilization rather than maximizing
   concurrency blindly.
6. Fuse adjacent auxiliary operations and remove unnecessary register moves,
   waits, and chain boundaries.
7. Prefer changes that reduce `parallel_predicted_npu_cycles`. The legacy
   additive `predicted_npu_cycles` remains diagnostic and may stay unchanged
   when a legal scheduling optimization succeeds.

## Constraints

- Modify only the firmware files allowed by the goal.
- Never modify the timing profile, simulator adapter, emulator, tests, or
  reported metrics to lower the score.
- Preserve numerical correctness for every configured workload.
- Do not optimize Python host execution time; it is unrelated to NPU cycles.

## Verification

After every change, run all hard gates and compare:

- SCALE-Sim MVU compute cycles;
- trace-derived memory cycles;
- auxiliary instruction cycles;
- legacy additive predicted cycles;
- parallel predicted cycles, memory/compute overlap, and resource utilization;
- critical-path events in `timing-schedule.json`;
- numerical correctness.

Only a gate-passing candidate with a lower parallel cycle estimate may advance.
The prompt contains a bounded critical-path and blocking summary. Use the
reported `timing_schedule_file` for the complete event timeline; dynamic raw
instruction indexes are available, but direct C source-line mapping is not.
