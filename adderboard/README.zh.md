> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU 上的添加程序模型

自足附加板项目:模型定义、培训、黄金
参考文献、 DRAM 版式、 RISC- V 固件以及 NPU 模拟器的测试
堆栈。

## 模型

|型号|参数|建筑|NPU 路径|
|---|---|---|---|
|页:1| 130 |1L GPT, d=4, 2h, ReLU, 罪PE|FP32 电话|
|缩写  140p| 140 |1L Qune3, d=4, 1h, SwiGLU, RMSNorm, RoPE|FP32 + FP16|

## 目录结构

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

## 快速启动

```bash
# Build firmware
make -C firmware TARGET=adder NATIVE_DIM=4 SEQ_LEN=24
make -C firmware TARGET=adder_140p BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24

# Run all tests (~10 min)
python3 -m pytest adderboard/tests/ -v
```

请参看 QQZPROT000XQZ 详细的每个型号的测试命令.
