# BERT dim4/seq6 RTL Baseline

This report preserves the first RTL evidence baseline for commit `0810dd8`,
workload `jimu-dse/workloads/bert-dim4-seq6.yaml`, and profile
`jimu-dse/timing/jimu-rtl-dim4.yaml`.  Reproduce it with the command in
`docs/rtl-timing-simulator.md`.

## Validation and performance boundary

| Gate/metric | Result |
|---|---:|
| Observable functional equivalence | pass, max diff `0.0` |
| Expanded dynamic commands | 1,907 |
| RTL makespan | 3,673 cycles |
| Sum of command durations | 4,552 cycles |
| Overlap saving versus serial sum | 879 cycles |
| DRAM/compute overlap | 850 cycles (57.59% of memory-active time) |
| Maximum concurrent operations | 3 |
| Load / store utilization | 32.83% / 7.35% |
| MVU / vector utilization | 9.15% / 56.66% |
| Oldest-command dependency stall counter | 2,248 cycles |
| DRAM / SRAM-bank stall counter | 102 / 76 cycles |
| Semantic scoreboard collisions | 0 (peak live resources: 81/128) |

The summed per-command queue waits are intentionally not a wall-clock counter:
several queued commands may wait during the same RTL cycle.  Promotion decisions
must use RTL makespan and the non-overcounted RTL counters.

## Evidence-backed optimization hypotheses

| Priority | Hypothesis | Evidence and conservative bound | Feasibility / expected migration |
|---:|---|---|---|
| 1 | Retain or hoist frozen parameters | Each of 16 frozen weight/bias/LN tensors is loaded 6 times. Removing loads after the first has a serial-work bound of 330 cycles, capped by actual critical contribution. | Requires a VRF/MRF capacity and bank proof. A single MRF cannot retain all matrices; staged VRF residency or persistent kernels are more plausible. Vector pressure is likely to become dominant. |
| 2 | Fuse non-observable scratch materializations | `scratch_ln`, `scratch_z`, `scratch_ln1`, and `scratch_gelu` account for 84 DRAM events and 252 command cycles (48/12/12/12 events). | Their manifest roles are intermediate and non-observable, but live ranges, FP16 ordering, and VRF capacity must be checked. DRAM pressure falls; vector/reduction work remains. |
| 3 | Rename/double-buffer high-pressure storage | Explicit edges include MRF: 118 WAR + 71 WAW; VRF5: 131 WAR + 132 WAW; VRF7: 131 WAR + 185 WAW. | VRF address rotation is firmware-visible. True MRF ping-pong needs an ISA/hardware extension or staged VRF-to-MRF path. The regression suite proves that legal ping-pong exposes prefetch overlap. |
| 4 | Rotate local-memory banks | RTL directly measures 76 bank-stall cycles; banks 5 and 7 occur on near-critical work. | Change addresses only after proving no alias/observable change and that total live storage fits. Maximum makespan bound is 76 cycles. |
| 5 | Hoist/batch repeated setup subgraphs | The analyzer finds 120 instances of `S_WR, V_RD_DRAM, V_WR, S_WR, V_RD_DRAM, V_WR`. | Separate invariant configuration from per-position addresses. Queue/fence pressure may replace control overhead. |

Model-regression tests also prove three required response properties: ping-pong
MRF storage improves prefetch overlap, removing a DRAM materialization lowers
makespan, and VRF bank rotation removes a modeled port stall.  Those are
microarchitecture-causality tests, not claimed BERT speedups.

## Evidence limitations

- The graph is dynamic for one shape and path; test at another sequence length
  before accepting a shape-general transformation.
- The RTL schedule supplies affirmative dependency edges.  Pipeline registers
  use SSA elastic tokens; VRF/MRF/SRF/DRAM/configuration remain physical.
- DWARF currently maps MMIO stores to `npu_driver.c:npu_send_inst`, so dynamic
  command IDs must be mapped back through `raw_instruction_idx` and call-site
  analysis before editing `firmware/bert/bert_layer.c`.
- Latencies and bandwidth are versioned architectural assumptions, not measured
  silicon values.  Compare candidates only under the same profile.
- A full chain fence is conservative; removing it requires control-order proof.

<dfg_optimization_hypotheses>
[
  {
    "kind": "live_range",
    "evidence": ["16 frozen tensors each loaded 6 times", "330 removable serial command cycles"],
    "change": "retain staged parameter tiles in VRF/MRF and hoist invariant loads",
    "expected_metric": {"name": "rtl_predicted_npu_cycles", "upper_bound_delta": -330},
    "feasibility": "prove VRF/MRF capacity, bank ports, last use, and no spill before implementation",
    "confidence": "medium",
    "measured_result": {"status": "not_tested", "delta": 0}
  },
  {
    "kind": "fusion",
    "evidence": ["84 scratch DRAM events", "252 serial command cycles", "scratch tensors non-observable"],
    "change": "forward/fuse scratch_ln, scratch_z, scratch_ln1, and scratch_gelu producers into consumers",
    "expected_metric": {"name": "rtl_predicted_npu_cycles", "upper_bound_delta": -252},
    "feasibility": "prove FP16 ordering and keep the concurrent live set within VRF/bank limits",
    "confidence": "medium",
    "measured_result": {"status": "not_tested", "delta": 0}
  },
  {
    "kind": "false_dependency",
    "evidence": ["MRF WAR=118 WAW=71", "VRF5 WAR=131 WAW=132", "VRF7 WAR=131 WAW=185"],
    "change": "rename VRF addresses and evaluate ping-pong MRF storage for adjacent tiles",
    "expected_metric": {"name": "rtl_dependency_stall_cycles", "upper_bound_delta": 0},
    "feasibility": "distinguish value dependencies from name reuse; MRF ping-pong is not expressible in the current ISA",
    "confidence": "medium",
    "measured_result": {"status": "not_tested", "delta": 0}
  },
  {
    "kind": "overlap",
    "evidence": ["850 memory/compute overlap cycles", "DRAM active 1476 cycles", "MVU utilization 9.15%"],
    "change": "software-pipeline independent loads ahead of MVU/vector consumers within the ROB window",
    "expected_metric": {"name": "memory_compute_overlap_cycles", "upper_bound_delta": 626},
    "feasibility": "destination must remain unmodified; prove queue, bank, alias, and fence legality",
    "confidence": "medium",
    "measured_result": {"status": "not_tested", "delta": 0}
  },
  {
    "kind": "pattern",
    "evidence": ["120 repeated setup/load/store subgraphs", "vector utilization 56.66%"],
    "change": "hoist invariant S_WR operations and batch or specialize repeated command groups",
    "expected_metric": {"name": "rtl_predicted_npu_cycles", "upper_bound_delta": 0},
    "feasibility": "separate invariant state from per-position address/configuration and preserve chain ordering",
    "confidence": "medium",
    "measured_result": {"status": "not_tested", "delta": 0}
  }
]
</dfg_optimization_hypotheses>
