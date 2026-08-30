# Structured DAG and cross-configuration proof

The structured DAG is a read-only analysis layer over the event trace. It folds instructions into micro-operations, builds def-use dependencies, recovers tensors and phases, and emits candidates and VRF allocation proofs that an independent gate can verify. It does not change firmware or emulator semantics.

Current schemas are `jimu-npu-micro-op-dag 1.2.0`, `jimu-npu-multiseq-dag 2.0.0`, and `jimu-npu-cross-config-allocation-proof 1.0.0` (BERT specialized) or `1.1.0` (generic conservative).

Schema 1.2 adds `metadata.workload`, adapter-owned Tensor roles/cache policy,
and optional authoritative execution ranges for hybrid workloads. See
[`dag-workload-adapters.md`](dag-workload-adapters.md) for the extension API
and fail-closed rules.

## Evidence flow

```text
event trace
  -> per-configuration micro-ops, edges, tensors, phases and lifetimes
  -> concrete seq2/seq6 families and L1 macros
  -> DIM/hidden/seq validation matrix and L2 allocation proof
  -> frozen before/after DAG plus measured-metric acceptance gate
```

Agent prose is not evidence. Raw `defs`, `uses`, addresses, event indices, ELF hashes, and machine-readable gate results are authoritative.

## Per-configuration artifacts

| File | Purpose |
|---|---|
| `run_metadata.json` | Schema, firmware configuration, ELF SHA256 and annotation method |
| `micro_ops.jsonl` | Stable operation IDs, kinds, phases, defs/uses and tensor annotations |
| `edges.jsonl` | Producer-consumer dependencies |
| `tensors.json` | DRAM tensor slices, producers and consumers |
| `phases.json` | Phase hierarchy, traffic, FLOPs and arithmetic intensity |
| `patterns.json` | Repeated semantic phase patterns |
| `lifetimes.json` | Observed resource definition-use lifetimes |
| `candidates.json` | Primitive exact-address DRAM round trips and proof status |
| `candidate_summary.md` | Compact primitive ranking |
| `summary.md` | Per-configuration index |

Legacy TXT/DOT/SVG files remain useful for inspection, but graph layout is never dependency evidence.

## Multi-sequence and proof artifacts

The root of `dag_agent/` or immutable `dag_before_iterN/` also contains:

| File | Purpose |
|---|---|
| `multiseq_metadata.json` | Sequence inputs, required proof configurations, ELF hashes and matrix completeness |
| `loop_invariants.json` | Concrete seq2/seq6 families and implementation readiness |
| `candidate_evidence.jsonl` | Compact primitive/macro evidence index |
| `multiseq_summary.md` | Compact cross-sequence summary |
| `macro_candidates.json` | L1/L2/L3 macros, eligibility, level and exact scope |
| `macro_candidate_summary.md` | Macro priority and blocking reasons |
| `allocation_proof.json` | Exact per-configuration bank/row plans, lifetimes, conflicts and proof digest |
| `allocation_summary.md` | Compact proof entry point; open the full JSON only after choosing a macro |

For BERT with `--validation-dim all`, the required matrix is dim2/h4 and dim4/h4,h8, each at seq2 and seq6. A missing required ID makes the matrix incomplete and blocks cross-configuration eligibility.

## Generate

```bash
python3 jimu-dse/scripts/visualize_graph.py \
  --phase micro --dim 2 --hidden 4 --seq-len 6 --num-head 2 \
  --no-render -o /tmp/jimu-dag-seq6

JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --agent opencode --validation-dim all
```

Build directories include dim, hidden and sequence length so configurations cannot silently reuse the wrong ELF. The closed loop generates and merges the full required matrix automatically.

## Optimization levels and allocation

- **L1:** remove intermediate-tensor DRAM store-load round trips.
- **L2:** retain loop invariants or sequence inputs, only with a complete cross-configuration allocation proof.
- **L3:** weight stationarity and loop interchange; currently blocked.

The L2 allocator uses the first-to-last exact-address read interval, proves the value read-only in the trace, and performs deterministic `NATIVE_DIM`-aligned first-fit allocation in `MFU_INITIAL_VRF`. It checks every allocation against all observed VRF lifetimes.

L3 still lacks loop-interchange, MRF-clobber, per-position partial-sum, and FP16 operation-order proofs. Its projected saving is an upper bound, not implementation authorization.

## Declaration and independent gate

Firmware declares exactly one scope:

```c
// JIMU_DAG_MACRO: macro-dram-l1-attention
```

Only when no macro is eligible may it use a primitive fallback:

```c
// JIMU_DAG_CANDIDATE: candidate-dram-0007
```

The gate requires eligibility, complete allocation and validation-matrix proof, enforces L1 before L2, rejects L3, checks every exact resource in the declared scope, and rejects reductions outside that scope. DRAM counting examines node resources in `uses` and `defs`, including accesses folded into compute nodes. It also requires non-regressing seq2 traffic, strictly improved seq6 traffic, correctness, and the configured instruction gate.

Each iteration freezes `dag_before_iterN/`; metric-only probes cannot replace it. Results are written to `dag_diff_N.json` and `dag_diff_N.md`.

See the [current status](project-status.zh.md) for verified coverage and remaining limits.
