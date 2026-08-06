---
name: dataflow-optimize
version: 1.1.0
category: dataflow-analysis
description: Derive firmware optimization hypotheses from scored dataflow graphs, quantify total-cycle impact, and validate one evidence-backed change at a time.
---

# Dataflow-Driven Firmware Optimization

## Establish the boundary

1. Confirm the scored hardware dimensions, sequence length, timing model, target
   metric, allowed files, and correctness gates.
2. Reject graph evidence whose recorded configuration differs from the scored
   workload.
3. Treat the graph as a dynamic trace, not a complete control-dependency or
   cycle-accurate pipeline model.
4. Distinguish the scored timing backend from legacy diagnostic metrics. Under
   the parallel scheduler, a legal reordering may improve the score by exposing
   memory/compute overlap even when the event count and additive cycle estimate
   remain unchanged.

## Build hypotheses from evidence

Read evidence in this order:

1. Read the prompt's critical-path events, blocking reasons, and utilization
   ranking to identify the active scheduled bottleneck.
2. Use the symbolic and operator graphs to find expensive repeated phases and
   producer-consumer boundaries.
3. Use the micro-op and DRAM-cluster graphs to identify the exact transfers,
   register moves, and repeated operations responsible for the cost.
4. Map the selected nodes back to the firmware source before editing. Raw trace
   indexes are not direct C source-line numbers.

Check every applicable opportunity class:

- a DRAM store followed by a reload of the same live value;
- repeated configuration writes, register moves, waits, or other auxiliary work;
- weight, MRF, or VRF values that can remain live across positions, heads, or
  loop iterations;
- layout conversions, transposes, or tile reconstruction between adjacent
  operators;
- adjacent operators that can be fused while reducing the number of modeled
  events;
- a change that lowers one component but moves the bottleneck into compute,
  memory, or auxiliary cycles.

Do not infer dead code solely from a disconnected configuration node. Control
dependencies may be absent from the graph.

## Select and implement

For each hypothesis, record:

- graph node, address, operator, or cluster evidence;
- proposed source change;
- affected metric and estimated total-cycle delta;
- value lifetime and register-capacity requirements;
- multi-tile, multi-head, and numerical-correctness risks.

Rank hypotheses by expected total-cycle benefit divided by risk. Implement one
primary hypothesis at a time. Preserve the pre-change candidate so a failed or
non-improving change can be discarded.

## Verify

Run every configured scope, build, probe, and numerical gate. Compare predicted
and measured compute, memory, auxiliary, and total cycles. Promote only a
gate-passing candidate whose total score improves by the configured minimum.
Reject local wins whose added cost makes the total metric worse.

End the response with a machine-readable record:

```text
<dataflow_hypotheses>
[
  {
    "evidence": "graph node/address/operator/cluster",
    "change": "implemented or rejected change",
    "expected_metric": {"name": "parallel_predicted_npu_cycles", "delta": -1},
    "risk": "capacity/lifetime/correctness risk",
    "measured_result": {"status": "accepted|rejected|not_tested", "delta": 0}
  }
]
</dataflow_hypotheses>
```
