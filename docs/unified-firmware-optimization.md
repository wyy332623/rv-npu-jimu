# Unified Firmware Optimisation Evidence

## Purpose

The optimisation loop must work from an executed firmware program rather than
from a hard-coded BERT graph. It therefore keeps four concerns separate:

1. `NpuDeviceMini` is the functional oracle and owns architectural effects.
2. `TimedNpuDevice` runs in lock-step with `MiniRV64` and exposes observable
   BUSY, DONE, FULL, and per-unit busy states.
3. `EventTracer` decodes the commands that actually execute, including expanded
   INC operations, and records def-use, source, memory, and tensor metadata.
4. `CrossLayerGraph` joins tensors, commands, dependencies, source locations,
   and timing records into deterministic JSON for an optimisation agent.

The graph renderer is intentionally not the contract. JSON is stable and
queryable; text, DOT, and SVG are review views generated from the same graph.

## Execution path

```text
firmware source --build--> ELF --MiniRV64--> MMIO command stream
                                      |             |
                                      |             +--> canonical NpuCommand
                                      |                       |
                                      |                       +--> EventTracer
                                      |                       +--> TimedNpuDevice
                                      |
workload manifest --------------------+--> tensor names and observables

events + timed schedule + source map --> CrossLayerGraph --> agent evidence
functional run vs timed run ----------> observable equivalence gate
```

`TimedNpuDevice` delays retirement but delegates every architectural side
effect to the functional device. Its configurable model contains decoder
latency, front-end and per-unit FIFO capacity, issue width, RAW/WAR/WAW
scoreboarding, unit count/latency/initiation interval, and one shared DRAM bus.
The default profile is `jimu-dse/timing/npu-timed-v1.yaml`.

The model creates optimisation space that an immediate-completion emulator
cannot expose: firmware polling time, FIFO pressure, dependency stalls,
structural contention, DRAM serialization, and legal unit overlap. It is a
calibratable architectural timing model, not a claim of RTL cycle accuracy.

## Workload manifest

An ELF can run without a manifest, but an autonomous optimiser needs a
correctness and semantic contract. A schema-v1 manifest declares:

- the ELF and timing profile;
- optional workload initialisation code;
- named tensor/buffer address ranges, shapes, roles, and data types;
- frozen parameter regions;
- externally observable output regions and tolerances;
- cycle and drain limits plus workload metadata.

See `docs/workload-manifest.example.yaml`. Overlapping declared DRAM regions
are rejected. Accesses outside declared ranges remain visible as anonymous
buffers, so incomplete annotation never hides traffic.

## Agent evidence

The cross-layer graph contains:

- tensor-to-command read and command-to-tensor write edges;
- command RAW, WAR, WAW, and chain-barrier edges;
- PC, source file, line, and function when DWARF is available;
- unit assignment, enqueue/start/finish/retire cycles, and stall readiness;
- evidence-backed candidates such as repeated frozen loads, non-observable
  intermediate store/reload pairs, and actionable scheduled waits.

These are hypotheses, not automatic transformations. A candidate is promoted
only after the configured build, scope, numerical, and score gates pass.

## Usage

Compile firmware with debug line information (`-g`) and analyse it with:

```bash
python3 scripts/analyze_firmware.py \
  --manifest path/to/workload.yaml \
  -o _out/firmware-analysis
```

The output directory contains `run-summary.json`, `trace-events.json`,
`timing-timeline.json`, and cross-layer graph JSON/text/DOT (plus SVG when
Graphviz is installed). When observables are declared, the command exits with
status 2 if functional and timed execution differ.

Closed-loop goals may additionally define:

```yaml
target:
  build:
    command: [make, -C, firmware, TARGET=custom, BUILD_DIR=build_custom]
    elf: firmware/build_custom/custom.elf
    cwd: .
    environment: {NATIVE_DIM: "{dim}"}
probe:
  workload_manifest: path/to/workload.yaml
  timed_device: {profile: jimu-dse/timing/npu-timed-v1.yaml}
```

This keeps firmware-specific build policy declarative. BERT goals retain
their existing default builder for backward compatibility. The BERT cycle
goal demonstrates the unified path with
`jimu-dse/workloads/bert-dim4-seq6.yaml`.

## Validation and calibration boundary

Use three tiers:

1. functional-vs-timed observable equivalence on every candidate;
2. timed-device and SCALE-Sim evidence for fast optimisation feedback;
3. Amaranth/Verilator/FPGA measurements for final calibration and promotion
   when the corresponding hardware backend is available.

Calibration changes versioned latency, bandwidth, FIFO, and unit parameters;
it must not change command decoding, graph identity, or observable contracts.
