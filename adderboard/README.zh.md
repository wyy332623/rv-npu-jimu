# NPU 上的 AdderBoard 模型

一个自包含的 AdderBoard 项目，包含模型定义、训练、golden reference、DRAM 布局、RISC-V 固件以及 NPU 模拟器栈的测试。

## 模型

| 模型 | 参数量 | 架构 | NPU 路径 |
|------|--------|------|----------|
| cosminscn_130p | 130 | 1 层 GPT，d=4、2 个 head、ReLU、正弦位置编码 | FP32 |
| dimopep_140p | 140 | 1 层 Qwen3，d=4、1 个 head、SwiGLU、RMSNorm、RoPE | FP32 + FP16 |

## 目录结构

```text
adderboard/
├── docs/                # 项目文档
│   ├── compatibility.md #   可运行模型、架构和约束
│   ├── porting-log.md   #   分步移植记录、设计决策和错误
│   └── how-to-run.md    #   三种配置的测试命令
├── models/              # 训练权重
│   └── 140p/            #   dimopep_140p 训练权重
├── golden/              # 纯 NumPy golden reference 实现
├── layout/              # DRAM 权重布局
├── firmware/            # 两个模型的 RISC-V 固件
├── tests/               # 测试文件
│   ├── test_130p_phase1.py   # 130p FP32：Python 驱动的 NPU 指令
│   ├── test_130p_phase2.py   # 130p FP32：ISS + 固件
│   ├── test_130p_cross.py     # 130p 交叉校验（阶段 1 与阶段 2）
│   ├── test_140p_phase1.py    # 140p FP32：Python 驱动的 NPU 指令
│   ├── test_140p_phase2.py    # 140p FP32：ISS + 两阶段固件
│   └── test_140p_fp16.py      # 140p FP16：FP16 模拟器数据通路
└── training/            # 训练脚本和 sweep 日志
```

## 快速开始

```bash
# 构建固件
make -C firmware TARGET=adder NATIVE_DIM=4 SEQ_LEN=24
make -C firmware TARGET=adder_140p BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24

# 运行全部测试（约 10 分钟）
python3 -m pytest adderboard/tests/ -v
```

各模型的详细测试命令见 `docs/how-to-run.md`。
