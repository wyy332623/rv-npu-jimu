# HDL-derived timing parameters and memory hierarchy approximation

## Status and provenance

The active Jimu RTL timing profiles use first-pass controller contracts derived
from `hdl_functional_model.zip` (SHA-256
`85762AD875EA855F5896C756E87148AB30569889C5F83B79BA2538D983D46BB3`).
The archive contains Python abstractions inferred from Amaranth HDL, not the HDL
itself or calibrated RTL measurements. These parameters are therefore marked
`hdl-derived-first-pass-uncalibrated` and remain replaceable profile data.

No source code from the archive is copied into the project. The integration
uses only the documented command envelopes and resource timing formulas.

## Directly adopted controller contracts

For the supplied dim4/4-lane configuration:

| Operation | Latency | II | Basis |
|---|---:|---:|---|
| `MV_MUL` | 8 | 10 | MVU row-fire schedule, adder-tree latency, DONE/DONE2 return |
| MFU add/sub/multiply | 3 | 3 | one vector group plus controller entry/exit |
| GELU/EXP | 3 | 3 | combinational primitive inside the MFU command envelope |
| Softmax/LayerNorm | 6 | 1 | supplied single-group SLU pipeline contract |
| `M_RD@18` | 17 | 17 | VecToMat drain: `1 + native_dim^2` serialized MRF writes |
| SPU scalar operation | 1 | 1 | supplied scalar contract |
| SPU reduction (`V_WR@14..16`) | 2 | 2 | supplied reduction contract |
| SPU broadcast (`V_RD@17`) | 1 | 1 | supplied broadcast contract |

`OP@opd0` is an operand-qualified timing key. It lets a reduction or broadcast
have a different contract from an ordinary VRF read/write without creating a
new ISA opcode.

The MVU and VecToMat values are computed from `native_dim` rather than stored as
dim4-only constants. For a single-group MVU:

```text
dot_latency = ceil(log2(min(lanes, native_dim))) + 1
latency = native_dim + dot_latency + 1
II = latency + 2
```

For configurations requiring multiple lane groups, the model includes the
supplied WAIT_MVU and ACC_GROUP/DRAIN gap between row/group fires.

## DRAM versus on-chip storage

The supplied HDL model assumes combinational external DRAM reads, so it cannot
provide realistic DRAM latency. The project adds a separate, deliberately
conservative hierarchy approximation:

| Parameter | External DRAM | On-chip VRF/SRF path |
|---|---:|---:|
| request/setup | 12 cycles | 1 cycle |
| data width | 8 bytes/cycle | `lanes * 2` bytes/cycle |
| minimum transfer | 16 bytes | none |
| element size | 2 bytes (FP16) | 2 bytes (FP16) |

The formulas are:

```text
dram_cycles = direction_setup
            + ceil(max(payload_bytes, minimum_transfer_bytes)
                   / dram_bytes_per_cycle)

on_chip_cycles = direction_setup
               + ceil(payload_bytes / on_chip_bytes_per_cycle)
```

Consequently, a dim4 vector transfer is modeled as 2 cycles from an on-chip
bank and 14 cycles from DRAM. A dim4 4x4 matrix DRAM transfer is 16 cycles.
The exact values are not silicon claims: they create the expected order-of-
magnitude separation while keeping all assumptions explicit and tunable.

DRAM commands continue to serialize on the RTL shared bus. On-chip accesses use
the existing bank masks and may overlap independent work when dependencies and
bank ports permit it. VecToMat holds its MRF write bank for its derived drain
duration.

## Integration points

- `emulator/npu_rtl_sim.py` converts each trace event into a per-command
  latency, opcode/operand-specific II, timing-source label, and memory tier.
- `rtl/jimu_npu_timing_core.sv` consumes those command contracts and determines
  ROB, dependency, controller, DRAM, bank, and fence scheduling.
- `emulator/npu_device_timed.py` uses the same hierarchy and instruction
  contracts for lock-step BUSY/DONE behavior.
- `jimu-dse/timing/jimu-rtl-dim{2,4}.yaml` and
  `jimu-dse/timing/npu-timed-v1.yaml` hold the replaceable parameters.

Every scheduled event records `timing_model.latency_source`, `memory_tier`,
`latency_cycles`, and `initiation_interval` so a closed-loop result can be
audited without reconstructing profile precedence.

## Calibration boundary

Before treating these values as hardware-validated, obtain the referenced
Amaranth revision and run component microbenchmarks for command accept, result
valid, next accept, simultaneous pipe completion, SRAM bank conflicts, and
external memory request/response timing. MFU add/sub and the single-group SLU
path are specifically called out as inconsistent or alignment-sensitive by the
supplied model.
