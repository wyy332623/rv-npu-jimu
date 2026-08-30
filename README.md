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
pip install pytest numpy pyelftools

# 2. Build C kernel library (matrix multiply, GELU, softmax, etc.)
make kernels

# 3. Build RISC-V firmware (bert.elf, BERT encoder layer)
make firmware

# 4. Run all tests
python3 -m pytest tests/ -v
```

Expected output: **all tests pass**.

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
  trace_recorder.py   MMIO instruction trace recorder
iss/
  mini_rv64.py        MiniRV64 ISS (RV64IM, pure Python)
tests/           pytest suite + golden reference generator
  gen_golden_bert.py   numpy golden reference for BERT encoder layer
jimu-dse/        Closed-loop firmware optimization pipeline
```

JIMU-DSE includes versioned agent skills, concrete seq2/seq6 structured DAGs, cross-configuration VRF allocation proofs, and independent correctness/metric/DAG gates. Start with the [JIMU-DSE documentation index](jimu-dse/docs/README.md) and the [current status](jimu-dse/docs/project-status.zh.md).

## Design Docs

| Document | Covers |
|----------|--------|
| `docs/architecture.md` | NPU architecture, ISA, register files |
| `docs/specification.md` | Full ISA reference, opcode table, execution model |
| `docs/firmware-guide.md` | RISC-V firmware, driver API |
| `docs/test-guide.md` | Test pyramid, fixtures, CI |
| `docs/build-guide.md` | Tool installation, build steps |
| `jimu-dse/docs/README.md` | JIMU-DSE documentation entry point |
| `jimu-dse/docs/how-to-run.md` | Closed-loop operation and validation |

## License

Apache 2.0.
