---
name: cycle-latency
version: 1.0.0
category: performance-model
description: Minimize SCALE-Sim-backed NPU cycle estimates on the critical firmware trace.
compatibility:
  schema_version: 1
---

# SCALE-Sim Cycle Latency Optimization

## Model Boundary

The cycle score is a hybrid performance model:

- SCALE-Sim v2.0.2 models each executed `MV_MUL` as a `1×DIM · DIM×DIM`
  weight-stationary GEMM on a DIM×DIM systolic array.
- The actual EventTracer sequence supplies DRAM transfers and all non-MVU
  instructions.
- Explicit trace memory cycles use the configured transfer bandwidth and
  setup latency.
- Auxiliary operations use the versioned timing profile.

This is more architectural than an access-count proxy, but it is not the
cycle-accurate RTL for this custom NPU. SCALE-Sim stall cycles are diagnostic
by default because the firmware trace already contains explicit memory
operations; including both would double count memory cost.

## Optimization Strategy

1. Reduce the number of executed `MV_MUL` tile operations without changing the
   mathematical result.
2. Eliminate explicit vector and matrix DRAM round trips on the critical path.
3. Retain and reuse weights or intermediates in VRF/MRF when capacity and
   lifetimes permit.
4. Fuse adjacent auxiliary operations and remove unnecessary register moves,
   waits, and chain boundaries.
5. Prefer changes that reduce `predicted_npu_cycles`; individual components may
   trade off if the total falls.

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
- total predicted NPU cycles;
- numerical correctness.

Only a gate-passing candidate with a lower total cycle estimate may advance.
