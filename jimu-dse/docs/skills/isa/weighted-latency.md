---
name: weighted-latency
version: 1.0.0
category: cost-model
description: Minimize weighted NPU DRAM and on-chip register access cost.
compatibility:
  schema_version: 1
---

# Weighted Latency Optimization

## Applicability

Use this skill when the goal defines:

```text
estimated_time =
    memory_access_count × memory_weight
  + register_access_count × register_weight
```

This value is a reproducible cost estimate, not cycle-accurate hardware time.
Read the weights and current read/write breakdown from the generated prompt.

## Counting Model

- A vector or matrix NPU DRAM read/write instruction counts as one memory
  access, regardless of the number of transferred elements.
- Each VRF, MRF, SRF, or internal scalar REG entry in an event's `uses` counts
  as one register read.
- Each such entry in `defs` counts as one register write.
- Pipeline temporaries, RISC-V general-purpose registers, and CPU memory
  operations are excluded.

## Transformation Strategy

1. Identify repeated DRAM loads and save/reload pairs in the DAG.
2. Keep reusable values in VRF, MRF, or SRF when their added register accesses
   cost less than the avoided DRAM operations.
3. Reuse already-loaded weights and intermediates across adjacent operations
   where lifetime and capacity permit.
4. Remove redundant register moves and reloads after introducing caching.
5. Recompute the whole weighted estimate; do not optimize either component in
   isolation.

With the default 10:1 weights, eliminating one DRAM operation may justify up to
nine additional register accesses and still reduce estimated cost.

## Constraints

- Only modify files allowed by the goal configuration.
- Preserve numerical correctness and the firmware/NPU interface.
- Do not change the emulator, probe, weights, or tests to manufacture a lower
  score.
- Do not assume that fewer source lines or fewer instructions imply lower
  weighted cost.

## Verification

Run every configured hard gate. Compare baseline and candidate values for:

- memory reads and writes;
- register reads and writes;
- `estimated_time`;
- numerical correctness.

Accept a transformation only when all hard gates pass and estimated time
decreases by the configured promotion delta.

## Expected Impact

The typical win is replacing expensive DRAM round trips with a controlled
number of on-chip register accesses, followed by eliminating redundant
register traffic introduced by the cache.
