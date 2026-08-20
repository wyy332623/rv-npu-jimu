---
name: rtl-dataflow
description: Optimize Jimu firmware using the Verilator RTL schedule, explicit dependency resources, bank conflicts, and memory/compute overlap.
version: 1.0.0
category: optimization
---

# RTL Data-flow Optimization

## Scored boundary

Minimize `rtl_predicted_npu_cycles` under the fixed
`jimu-rtl-dim4.yaml` profile.  The functional emulator and numerical gates are
authoritative for values; the RTL is authoritative for command/control timing.

Do not edit the RTL, profile, simulator adapter, metrics, tests, or artifacts.
Only edit the firmware file allowed by the goal.

## Hardware model

- A finite ROB accepts commands in program order and dispatches the oldest
  eligible command each cycle.
- Load, store, MVU, vector, and control controllers execute independently.
- Commands within one controller issue in order; different controllers may
  overlap or pass a blocked command.
- All DRAM transfers serialize on one bus, but the bus works in parallel with
  independent MVU/vector work.
- The scoreboard enforces explicit RAW/WAR/WAW on semantic resources.
- `pipe` and `vpipe_a` are SSA elastic tokens. VRF, MRF, SRF, configuration
  registers, and granular DRAM ranges are physical resources.
- Local SRAM has banked read/write ports. Same-bank streams can stall even when
  there is no value dependency.
- `INST_ISSUE` is a conservative full fence in this profile.

## Evidence workflow

1. Read the bounded causal-chain/resource summary in the prompt. The chain is
   post-hoc from the latest completion, not a formal zero-slack critical path.
2. Open `timing-schedule.json` only for the candidate region. Use
   `dependency_predecessors`, `dependency_reasons`, `dependency_resources`,
   `rtl_stall_cycles_by_reason`, timing, tensor names, and raw instruction index.
   Treat stall counters as non-additive pressure indicators: a younger command
   may dispatch during a cycle charged to the oldest blocked command.
3. Confirm source intent in `firmware/bert/bert_layer.c`. Dynamic command IDs are
   not C line numbers; current DWARF often points at `npu_send_inst`.
4. State one transformation and the exact scored component it should change.
5. Prove value semantics, alias/control order, observable equivalence, capacity,
   banks, ports, queue window, and live ranges.
6. Estimate a conservative bound capped by causal/near-causal contribution;
   do not sum stall-counter deltas to predict makespan.
7. Implement only the highest-ranked hypothesis and return control for official
   gates and scoring.

## High-value transformations

- hoist repeated frozen loads and retain weights/parameters on chip;
- keep non-observable scratch values in pipeline/VRF and fuse consumers;
- rename or rotate VRF addresses to remove false WAR/WAW and bank conflicts;
- software-pipeline tiles/positions so next loads overlap current compute;
- hoist invariant `S_WR` setup and batch repeated command groups;
- remove a fence only with affirmative control/dependency proof;
- reduce MVU/vector work through legal fusion or reuse.

Predict bottleneck migration.  Reducing DRAM work can make the vector controller
or a local SRAM bank dominant.  Increasing overlap can increase live storage and
create spills or front-end pressure.

## Acceptance

An improvement is valid only when all configured builds and numerical tests
pass, `rtl_predicted_npu_cycles` decreases, and the schedule delta matches the
intended cause.  Reject unexplained score movement, hash collisions, new spills,
or local wins that extend another critical chain.
