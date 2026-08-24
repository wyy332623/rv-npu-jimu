# Test Guide

## Test Suite

```
tests/
├── gen_golden_bert.py                  ← NumPy golden reference for BERT
├── conftest.py                         ← pytest config (`--instrument`)
├── integration/
│   └── test_bert_e2e.py                ← golden → firmware/ISS → emulator; optional HDL replay
└── unit/
    ├── test_bert_layout.py             ← legacy and packed DRAM layouts
    ├── test_closed_loop.py             ← goal schema, agent loop, promotion, and recovery
    ├── test_event_trace_metadata.py    ← trace tensor/resource metadata
    ├── test_kv_cache_validation.py     ← K/V cache ordering and value gates
    ├── test_npu_rtl_sim.py             ← RTL trace encoding and contention
    ├── test_npu_rtl_optimization_space.py
    ├── test_parallel_scheduler.py
    ├── test_scalesim_adapter.py
    ├── test_visualize_rtl_parallelism.py
    └── test_workload_and_timed_device.py
```

## BERT End-to-End Test

The primary test `test_bert_e2e_multi_tile` validates the firmware across
multiple configurations:

| Param | Values |
|-------|--------|
| dim (NATIVE_DIM) | 2, 4, 16 |
| hidden_size | 4, 8, 16 |
| seq_len | 2, 6, 16 |
| num_head | 1, 2 |

These values form eight versioned configurations rather than a Cartesian
product. They include legacy multi-tile and single-tile layouts, a second
deterministic seed for dim2/seq6, and the packed dim16/seq16 baseline.

### Validation Rounds

| Round | Backend | What it checks |
|-------|---------|----------------|
| **R0** | NumPy golden | FP16-aware algorithmic reference via `gen_golden_bert.py` |
| **R1** | Firmware + ISS + emulator | Instruction semantics, opcode coverage, every output position, and K/V storage |
| **R2** | Amaranth sequential replay | Compatibility hook for optional HDL replay compared with the emulator |
| **R3** | Amaranth batch replay | Compatibility hook for optional batched HDL replay |

Every sequence position's final output is checked. K/V projections are also
validated at their actual storage boundary, either the baseline DRAM layout or
the optimized MFU VRF cache. Other intermediates remain free to move or fuse as
long as the observable output and explicit K/V correctness gates pass. R2 and
R3 are skipped unless both Amaranth and the external `hdl.npu_top` model are
available; the self-contained repository normally runs R0 and R1.

## Running Tests

```bash
# All unit tests
python3 -m pytest tests/unit/ -v

# All integration tests
python3 -m pytest tests/integration/ -v

# Single configuration
python3 -m pytest tests/integration/test_bert_e2e.py -k "seq6" -v

# With per-operator diagnostics
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s

# RTL trace-replay and optimization-space subset
make rtl-test

# Complete suite
python3 -m pytest tests/ -v
```

## DRAM Stats

Each integration configuration prints operation counts, transferred elements,
the FP16 RTL payload, and the float32 storage used by the functional emulator:

```
  DRAM traffic:
    V_RD_DRAM: <ops> ops, <elements> elements
    V_WR_DRAM: <ops> ops, <elements> elements
    M_RD_DRAM: <ops> ops, <elements> elements
    M_WR_DRAM: <ops> ops, <elements> elements
    Total: <elements> elements (FP16 RTL payload <bytes> bytes;
            float32 emulator storage <bytes> bytes)
```

## Pre-Commit Hook

```bash
python3 -m pytest tests/ -v
```

Run before committing to ensure no regressions.
