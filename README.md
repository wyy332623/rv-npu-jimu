# NPU — Open-Source Edition

An open-source FPGA neural processing unit (NPU) emulator.

## Architecture

```
RISC-V firmware (.elf) → MiniRV64 ISS → NPU device (Python + C)
```

The NPU is a Python-based functional emulator driven by a RISC-V firmware
instruction stream. The Python code handles instruction decode, register
files, and control flow. **All numerical computation** (MV_MUL, softmax,
layernorm, GELU, vector add/sub/mul) is delegated to a C kernel library
(`libnpukernels.so`) via ctypes.

A separate NumPy golden reference (`tests/gen_golden_bert.py`) is used by
tests to validate emulator output, but is not part of the emulator itself.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build C kernel library (matrix multiply, GELU, softmax, etc.)
make kernels

# 3. Build RISC-V firmware (bert.elf, BERT encoder layer)
make firmware

# 4. Run all tests
python3 -m pytest tests/ -v
```

Expected output: **all tests pass**.

## Configurable Firmware Optimization

The closed loop is configured with versioned YAML goals:

```bash
python3 jimu-dse/scripts/closed_loop.py list-goals
python3 jimu-dse/scripts/closed_loop.py validate-config --goal dram-optimization
python3 jimu-dse/scripts/closed_loop.py render-prompt --goal dram-optimization
bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization --agent opencode

# SCALE-Sim-backed parallel resource schedule
make timing-deps
python3 jimu-dse/scripts/closed_loop.py validate-config \
  --goal cycle-latency-optimization
```

See `jimu-dse/docs/how-to-run.md` for the goal schema, scoring, skills,
resume behavior, and run artifacts.

For arbitrary firmware supported by the ISS/NPU emulator, a workload manifest
can attach tensor semantics and observable output ranges to the executed
command stream. The lock-step timed device then exposes FIFO, scoreboard,
execution-unit, and DRAM contention while reusing the functional emulator as
the correctness oracle:

```bash
python3 scripts/analyze_firmware.py \
  --manifest path/to/workload.yaml \
  -o _out/firmware-analysis
```

See `docs/unified-firmware-optimization.md` for the manifest, custom build,
cross-layer graph, agent evidence, and calibration contracts.

### Verilator RTL Timing Backend

For hardware-visible load/store/compute concurrency, ROB dependencies, SRAM
bank conflicts, chain fences, cycle counters, and optional VCD waveforms:

```bash
python3 scripts/analyze_firmware.py \
  --manifest jimu-dse/workloads/bert-dim4-seq6.yaml \
  --rtl-profile jimu-dse/timing/jimu-rtl-dim4.yaml \
  -o _out/bert-rtl --no-render
```

The same backend is available to the agent loop as a scored goal:

```bash
python3 jimu-dse/scripts/closed_loop.py validate-config \
  --goal rtl-cycle-optimization
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal rtl-cycle-optimization --agent opencode
```

The RTL is a command/control timing model; functional and numerical equivalence
continues to come from the existing emulator.  See
`docs/rtl-timing-simulator.md` for the architecture, open-source design study,
optimization space, artifacts, and fidelity limits.

## Running Tests

```bash
# C kernel tests only (fast, 15 tests)
python3 -m pytest tests/unit/ -m c_kernel -v

# Integration: BERT E2E (parameterized)
python3 -m pytest tests/integration/ -v

# Diagnostic: dispatch audit
python3 -m pytest tests/diagnostic/ -k static -v
```

## Project Structure

```
kernels/         C compute library (libnpukernels.so)
firmware/        RISC-V firmware (C, bare-metal ELF → bert.elf)
emulator/
  npu_device_mini.py  NPU device — functional Python emulator
  npu_device_timed.py Lock-step FIFO/scoreboard/resource timing wrapper
  npu_command.py      Canonical decoded command and memory-access model
  workload.py         Tensor/observable manifest and ELF source mapping
  npu_cross_layer_graph.py  Tensor → command → timing evidence graph
  trace_recorder.py   MMIO instruction trace recorder
  npu_rtl_sim.py      Trace encoder + Verilator RTL schedule adapter
rtl/
  jimu_npu_timing_core.sv  Synthesizable ROB/DMA/compute timing core
sim/
  jimu_rtl_harness.cpp     Verilator trace replay, counters, and VCD harness
iss/
  mini_rv64.py        MiniRV64 ISS (RV64IM, pure Python)
tests/           pytest suite + golden reference generator
  gen_golden_bert.py   numpy golden reference for BERT encoder layer
jimu-dse/        Closed-loop firmware optimization pipeline
```

## Design Docs

| Document | Covers |
|----------|--------|
| `docs/architecture.md` | NPU architecture, ISA, register files |
| `docs/specification.md` | Full ISA reference, opcode table, execution model |
| `docs/firmware-guide.md` | RISC-V firmware, driver API |
| `docs/test-guide.md` | Test pyramid, fixtures, CI |
| `docs/build-guide.md` | Tool installation, build steps |
| `docs/unified-firmware-optimization.md` | Generic firmware timing, tensor semantics, graphs, and agent evidence |
| `docs/rtl-timing-simulator.md` | RTL architecture, design sources, performance counters, and optimization space |
| `docs/rtl-bert-baseline.md` | Reproducible dim4/seq6 RTL baseline and ranked data-flow hypotheses |

## License

Apache 2.0.
