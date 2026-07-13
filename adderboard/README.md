# AdderBoard Models on NPU

Self-contained adderboard project: model definitions, training, golden
references, DRAM layouts, RISC-V firmware, and tests for the NPU emulator
stack.

## Models

| Model | Params | Architecture | NPU Path |
|---|---|---|---|
| cosminscn_130p | 130 | 1L GPT, d=4, 2h, ReLU, sin PE | FP32 |
| dimopep_140p | 140 | 1L Qwen3, d=4, 1h, SwiGLU, RMSNorm, RoPE | FP32 + FP16 |

## Directory Structure

```
adderboard/
├── docs/                # Project documentation
│   ├── compatibility.md #   Which models work, architecture, constraints
│   ├── porting-log.md   #   Step-by-step porting record, design decisions, bugs
│   └── how-to-run.md    #   Test commands for all 3 configurations
├── models/              # Trained weights
│   └── 140p/            #   dimopep_140p trained weights
├── golden/              # Golden reference implementations (pure numpy)
├── layout/              # DRAM weight layouts
├── firmware/            # RISC-V firmware for both models
├── tests/               # Test files
│   ├── test_130p_phase1.py   # 130p FP32 — Python-driven NPU instructions
│   ├── test_130p_phase2.py   # 130p FP32 — ISS + firmware
│   ├── test_130p_cross.py    # 130p cross-check (Phase 1 vs Phase 2)
│   ├── test_140p_phase1.py   # 140p FP32 — Python-driven NPU instructions
│   ├── test_140p_phase2.py   # 140p FP32 — ISS + two-phase firmware
│   └── test_140p_fp16.py     # 140p FP16 — FP16 emulator datapath
└── training/            # Training scripts + sweep logs
```

## Quick Start

```bash
# Build firmware
make -C firmware TARGET=adder NATIVE_DIM=4 SEQ_LEN=24
make -C firmware TARGET=adder_140p BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24

# Run all tests (~10 min)
python3 -m pytest adderboard/tests/ -v
```

See `docs/how-to-run.md` for detailed per-model test commands.
