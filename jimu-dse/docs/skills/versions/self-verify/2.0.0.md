---
name: self-verify
version: 2.0.0
description: Enforce scoped firmware correctness gates without weakening or filtering validation
license: MIT
---

# Self-Verify Skill

This skill is mandatory and runs after the goal-specific transformation skill.
It does not replace the closed loop's independent acceptance gate.

## Authority

- The exact command under `Independent Acceptance Gate` in the run prompt is
  authoritative.
- Run that command without appending `grep`, `head`, `tail`, `|| true`, a
  different `-k` expression, or any other output filter.
- Success requires pytest exit status 0, the expected number of passed tests,
  and zero skipped tests.
- Never modify tests, tolerances, golden data, skip/xfail markers, the emulator,
  ISS, hardware model, or validation command.
- A diagnostic subset can locate a defect, but it cannot accept a candidate.

## BERT Acceptance Matrix

| Validation scope | Required configurations | Expected passed |
|---|---|---:|
| `dim2` | dim2/hidden4, seq2 and seq6 | 2 |
| `dim4` | dim4/hidden8 seq2/seq6 and dim4/hidden4 seq2/seq6 | 4 |
| `all` | every dim2 and dim4 configuration above | 6 |

`all` is the production acceptance scope. `dim2` and `dim4` are supported for
targeted debugging; a partially validated candidate must not be described as
cross-dimension verified.

## Required Workflow

1. Compile the modified target firmware.
2. Run the exact independent acceptance command from the prompt.
3. Check the command's exit status, not filtered console text.
4. Confirm the passed-test count matches the selected scope.
5. Confirm pytest reports no skipped checks.
6. Only after correctness passes, compare the optimization metric.

If correctness fails, do not relax the tolerance. Use instrumentation or a
targeted test only to identify the first divergent stage, then rerun the full
selected gate.

## Diagnostic Order

Localize the first divergence in this order:

1. Q/K/V projections;
2. attention scores and softmax;
3. context/self-output;
4. first residual and LayerNorm;
5. FFN intermediate and GELU;
6. FFN output, second residual, and final LayerNorm.

Some optimized candidates intentionally keep tensors in VRF and do not
materialize every intermediate in DRAM. Do not claim an intermediate matched
unless the active instrumentor actually captured it.

## Required Agent Report

Before finishing, report:

```text
validation_scope:
acceptance_command:
pytest_returncode:
passed/expected:
skipped:
metric_before:
metric_after:
result: PASS or FAIL
```

Only the independent closed-loop gate may mark the candidate accepted.
