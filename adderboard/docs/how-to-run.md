# Running the NPU Model Emulations

Three configurations are available:

| Model | Data path | Emulator class |
|---|---|---|
| **130p FP32** | cosmin's hand-crafted | `NpuFP32` (FP32, no truncation) |
| **140p FP32** | trained Qwen3 140-param | `NpuFP32` (FP32, no truncation) |
| **140p FP16** | trained Qwen3 140-param | `NpuDeviceMini` (FP16 truncation) |

All emulators run on x86 — no RTL or FPGA needed.

---

## Quick start

```bash
cd /home/ubuntu/work/git-rv-npu

# Build the firmware (needed for ISS tests)
make -C firmware TARGET=adder BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24
make -C firmware TARGET=adder_140p BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24

# Run all tests (~10 minutes, 46 tests)
python3 -m pytest adderboard/tests/ -v
```

---

## Test structure

Tests are consolidated into 3 files, parametrized across models:

```
adderboard/tests/
├── conftest.py        # Shared fixtures, model registry, test cases
├── test_phase1.py     # Python-driven NPU instructions (FP32, both models)
├── test_phase2.py     # ISS + firmware (both models) + replay cross-check
├── test_fp16.py       # FP16 emulator datapath (140p only)
├── _test_130p_phase1.py  # (internal) 130p build_forward
├── _test_130p_phase2.py  # (internal) 130p ISS helpers
├── _test_140p_phase1.py  # (internal) 140p build_forward
├── _test_140p_phase2.py  # (internal) 140p ISS helpers
└── _test_130p_cross.py   # (internal) replay capture helper
```

`_`-prefixed files are internal modules (not runnable test files).

---

## Shared test cases

All models are tested against the same 6 numerical cases:

| Case | Description |
|---|---|
| `0-0-0` | Trivial |
| `5-5-10` | Small |
| `555-445-1000` | Mid-range |
| `19492-23919-43411` | Random |
| `9999999999-1-10000000000` | 10-digit overflow (carry from 9s) |
| `1111111111-8888888889-10000000000` | 10-digit overflow (balanced) |

---

## Test inventory (46 tests)

### `test_phase1.py` — Python-driven NPU instructions, FP32 (17 tests)

```bash
python3 -m pytest adderboard/tests/test_phase1.py -v
```

| Test | Count | Description |
|---|---|---|
| `test_first_step[model]` | 2 | Single-step logits match golden |
| `test_autoregressive[model-case]` | 12 | Autoregressive inference (2 models × 6 cases) |
| `test_bulk_random[model]` | 2 | 50 random autoregressive pairs |
| `test_intermediates_140p` | 1 | Intermediate values (ctx, attn_res) for 140p |

### `test_phase2.py` — ISS + firmware (22 tests)

```bash
python3 -m pytest adderboard/tests/test_phase2.py -v
```

| Test | Count | Description |
|---|---|---|
| `test_iss_build[model]` | 2 | Firmware compiles |
| `test_first_step[model]` | 2 | ISS single-step logits match golden |
| `test_autoregressive[model-case]` | 12 | ISS autoregressive (2 models × 6 cases) |
| `test_phase2_replay_vs_phase1[model]` | 2 | Phase 2 instruction replay matches Phase 1 |

### `test_fp16.py` — FP16 emulator (7 tests)

```bash
python3 -m pytest adderboard/tests/test_fp16.py -v
```

| Test | Count | Description |
|---|---|---|
| `test_first_step` | 1 | FP16 single-step logits (max diff < 20) |
| `test_autoregressive[case]` | 6 | FP16 autoregressive (6 cases) |

---

## Per-model breakdown

| Model | Phase 1 | Phase 2 | FP16 | Total |
|---|---|---|---|---|
| 130p | 9 | 11 | — | 20 |
| 140p | 10 | 11 | 7 | 28 |
| **All** | | | | **46** |

---

## Source files

- `adderboard/golden/golden_130p.py` — 130p golden reference (pure numpy)
- `adderboard/golden/golden_140p.py` — 140p golden reference (Qwen3 architecture)
- `adderboard/layout/layout_130p.py` — 130p DRAM weight layout
- `adderboard/layout/layout_140p.py` — 140p DRAM weight layout
- `adderboard/models/140p/s44_targeted_final_fp16.pt` — trained 140p weights
- `adderboard/firmware/adder_130p.c` — 130p RISC-V firmware (single-phase)
- `adderboard/firmware/adder_140p.c` — 140p RISC-V firmware (two-phase)
- `emulator/npu_fp32.py` — FP32 test helper (shared)
- `emulator/npu_device_mini.py` — NPU emulator with FP16 truncation (shared)

## Architecture overview

```
                    ┌──────────────┐
                    │  Test script │  (Python, pre-fills DRAM)
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌─────────────┐ ┌───────────┐ ┌───────────────┐
    │ Phase 1     │ │ Phase 2   │ │ Phase 2+ISS   │
    │ (Python)    │ │ (replay)  │ │ (firmware)    │
    └──────┬──────┘ └─────┬─────┘ └──────┬────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────────────────────────────────────────┐
    │            NPU Emulator                       │
    │  NpuFP32 (FP32) or NpuDeviceMini (FP16)       │
    └──────────────────────────────────────────────┘
```
