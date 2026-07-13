# Build Guide

## Prerequisites

### Required Tools

| Tool | Min. Version | Install |
|------|-------------|---------|
| Python | ≥ 3.10 | `apt install python3 python3-pip python3-venv` |
| CMake | ≥ 3.20 | `apt install cmake` |
| GCC (native) | ≥ 10 | `apt install gcc` — compiles `libnpukernels.so` for the host |
| RISC-V cross GCC | ≥ 10 | `apt install gcc-riscv64-unknown-elf` — compiles firmware for RV64IM bare-metal |
| numpy | ≥ 2.0 | `pip install numpy` |
| pytest | ≥ 7.0 | `pip install pytest` |
| pyelftools | ≥ 0.29 | `pip install pyelftools` |

### Optional Tools

| Tool | Version | Install | Purpose |
|------|---------|---------|---------|

| Graphviz | — | `apt install graphviz` | Render DAG .dot files to SVG |
| pi | ≥ 0.79 | `npm install -g @earendil-works/pi-coding-agent` | AI agent for automatic optimization |

### ISS

MiniRV64 is a pure-Python RV64IM ISS included in the repo at `iss/mini_rv64.py`.
No external dependencies.

## Build Steps

### 1. C Kernel Library

```bash
make kernels
```

Compiles `kernels/*.c` into `_build/kernels/libnpukernels.so`, a shared
library used by the emulator for fast matrix multiply, GELU, softmax, etc.

### 2. RISC-V Firmware

```bash
make firmware
```

Compiles `firmware/bert/bert_layer.c` into a RISC-V ELF binary. The
firmware is a bare-metal program that runs on MiniRV64 and drives the
NPU via MMIO writes. Output: `firmware/build/bert.elf`.

### 3. Run Tests

```bash
# Integration test (BERT E2E)
python3 -m pytest tests/integration/ -v

# All integration + unit tests
python3 -m pytest tests/ -v
```

The test suite auto-detects optional dependencies:

- **No ISS** — tests skip gracefully with explanatory message

## Directory Layout

```
_build/
└── kernels/libnpukernels.so      ← C kernel library (compiled)
firmware/
└── build_dim{2,4}/bert.elf       ← RISC-V firmware (compiled per config)
```

The firmware ELF is rebuilt automatically with the correct DRAM layout
macros each time the closed-loop pipeline probes a configuration.
