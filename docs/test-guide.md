# Test Guide

## Test Suite

```
tests/
├── gen_golden_bert.py            ← numpy golden reference for BERT encoder layer
├── conftest.py                   ← pytest config (--instrument flag)
└── integration/
    └── test_bert_e2e.py          ← BERT E2E: golden → emulator
```

## BERT End-to-End Test

The primary test `test_bert_e2e_multi_tile` validates the firmware across
multiple configurations:

| Param | Values |
|-------|--------|
| dim (NATIVE_DIM) | 2, 4 |
| hidden_size | 4, 8 |
| seq_len | 2, 6 |
| num_head | 2 |

### Validation Rounds

| Round | Backend | What it checks |
|-------|---------|----------------|
| **R0** | numpy golden | Algorithmic correctness via `gen_golden_bert.py` |
| **R1** | Emulator | Instruction semantics, DRAM layout, opcode coverage, final output comparison |


Only the **final output** is checked for numerical correctness.
Intermediates (Q, K, V, residual, LN) are optimization-free — the firmware
may store them in VRF cache, on-chip SRAM, or DRAM as needed.

## Running Tests

```bash
# All integration tests
python3 -m pytest tests/integration/ -v

# Single configuration
python3 -m pytest tests/integration/test_bert_e2e.py -k "seq6" -v

# With per-operator diagnostics (prints final output comparison only)
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s
```

## DRAM Stats

Each test run prints DRAM traffic:

```
  DRAM traffic (float32 elements):
    V_RD_DRAM: 312 ops × 8 el = 2496 el
    V_WR_DRAM: 12 ops × 8 el = 96 el
    M_RD_DRAM: 144 ops × 64 el = 9216 el
    Total: 11808 elements (47232 bytes)
```

## Pre-Commit Hook

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

Run before committing to ensure no regressions.
