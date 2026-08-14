# Open-source NPU Timing Simulator Selection

## Decision

The closed loop uses two complementary timing layers. The native
`TimedNpuDevice` is a lock-step MMIO device for the complete custom instruction
stream. It models decoder and FIFO pressure, scoreboarding, issue width,
per-unit pipelines, and a shared DRAM bus while delegating retirement to the
functional emulator. This makes firmware BUSY/DONE/FULL and CHAIN_STATUS
polling part of execution.

The second layer integrates
[SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) v2.0.2 at upstream
commit `c2b408e4b5fd8951f69b7a455e9ad5a97eef0e5c` (MIT license).
The installed Python package reports version 2.0.1 because that is the package
version declared by the v2.0.2 release.

SCALE-Sim was selected because it:

- models systolic-array GEMM and convolution workloads, including attention
  GEMMs;
- reports compute and stall cycles, utilization, bandwidth, and memory access
  details;
- has analytical compute cycles validated against RTL;
- exposes a Python API and accepts simple GEMM topology CSV files;
- has a permissive MIT license.

The project uses the stable v2 release instead of v3 because v2 has a smaller,
well-understood configuration surface for scripted GEMM integration. Upgrading
to v3 can be evaluated later for bank conflicts, Ramulator, sparsity, layouts,
and Accelergy integration.

## Alternatives considered

| Tool | Strength | Reason not selected as the first backend |
|---|---|---|
| [Timeloop/Accelergy](https://timeloop.csail.mit.edu/) | Arbitrary memory hierarchies, mappings, cycles, energy and area | Better suited to architecture/mapping exploration than replaying this firmware ISA |
| [MAESTRO](https://research.nvidia.com/publication/2020-04_maestro-data-centric-approach-understand-reuse-performance-and-hardware-cost) | Fast analytical DNN mapping cost model | Requires translating the firmware into data-centric mapping directives and does not model its instruction trace directly |
| [NVDLA virtual platform/RTL](https://nvdla.org/vp.html) | Register-accurate software platform and NVDLA RTL verification | Tied to the NVDLA architecture and software stack rather than this custom NPU ISA |
| [Verilator](https://github.com/verilator/verilator) | Cycle-accurate execution of the project's own RTL | The repository currently has no complete RTL backend to compile; this remains the preferred final validation tier |

## Integration boundary

Every raw MMIO word is decoded once into the canonical `NpuCommand` model.
Both EventTracer and the native timed device consume that model, preventing
the timing and graph paths from assigning different semantics to an opcode.
The default native profile is `jimu-dse/timing/npu-timed-v1.yaml`.

The adapter translates each executed `MV_MUL` EventTracer event into:

```text
M=1, N=NATIVE_DIM, K=NATIVE_DIM
```

SCALE-Sim provides systolic MVU compute and diagnostic memory-stall cycles.
The rest of the custom NPU instruction stream is modeled from the actual trace:

```text
predicted_npu_cycles =
    scalesim_compute_cycles
  + trace_memory_cycles
  + auxiliary_cycles
```

For the schema-v2 `scalesim-parallel` backend, this legacy sum is retained for
historical comparison. A deterministic scoreboard scheduler additionally maps
each event onto one shared DRAM bus and one VMM, MMM, MVU, or SPU, enforcing
RAW/WAR/WAW, overlapping DRAM ranges, configuration fences, and structural
hazards:

```text
parallel_predicted_npu_cycles = scheduled trace makespan
```

The scheduler admits one instruction per cycle through a two-entry queue, so a
later independent operation can bypass one blocked operation and overlap on a
different unit. `INST_ISSUE` closes a chain and prevents overlap with the next
chain. A trace with no marker is handled as one implicit ordered stream.

By default, SCALE-Sim's stall cycles are reported but not added to the total.
The firmware trace already contains explicit matrix/vector DRAM operations, so
adding both would double-count memory costs.

Neither profile-based layer claims cycle accuracy for the entire NPU. They improve on
access-count weighting and serial summation by using an external RTL-validated
systolic model for the dominant GEMMs and an auditable resource schedule for
the custom instructions, while keeping every assumption explicit and
versioned.

## Calibration and future work

The timing profile in `jimu-dse/timing/scalesim-dim4.yaml` is protected from
the optimization agent. Its memory bandwidth, setup costs, and auxiliary
instruction latencies should be calibrated with:

1. per-instruction RTL or FPGA microbenchmarks;
2. end-to-end BERT cycle measurements;
3. held-out firmware traces used to measure prediction error.

Calibration should include isolated DRAM, MVU, VMM, and MMM tests followed by
paired DMA–MVU and DMA–VMM overlap tests. These measurements tune the profile;
they do not change the scheduling or artifact interfaces.

When a complete Verilator/RTL backend is available, candidates should use the
lock-step device plus SCALE-Sim for fast per-iteration feedback and the RTL
cycle count as the final promotion/calibration gate.
