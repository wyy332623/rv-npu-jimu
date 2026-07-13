# AdderBoard 模型在 NPU 上的兼容性与状态

本文总结 AdderBoard 挑战提交在 NPU 模拟器 + RISC-V 固件栈上的运行情况。

## 当前状态

| 模型 | 参数量 | 架构 | NPU 状态 | 测试数 |
|------|--------|------|----------|--------|
| **cosminscn_130p** | 130 | 1 层 GPT，d=4、2h、ReLU、正弦位置编码 | 已移植，仅 FP32 | 20 |
| **dimopep_140p** | 140 | 1 层 Qwen3，d=4、1h、SwiGLU、RMSNorm、RoPE | 已移植，FP32 + FP16 | 28 |
| cosminscn_66p | 66 | 1 层 GPT，d=4、2h、ReLU、正弦位置编码 | 尚未尝试 | — |
| lichengliu03_50p | 50 | 1 层 GPT，d=4、2h、ReLU、正弦位置编码、秩 1 | 尚未尝试 | — |
| zcbtrak_6p | 6 | 1 层 Qwen，d=2、SiLU、RoPE、float64 | 尚未尝试 | — |

测试整合在 3 个参数化文件中，共 49 项。所有模型共享 6 个确定性测试用例，并在各阶段执行随机批量测试。

## 已移植模型

### cosminscn_130p

这是 d=4、2 个 head、head_dim=2、ff_dim=4 的手工构造 FP32 模型。全部计算都在 NPU 上执行：attention 使用 VecToMatRow 和 SPU Softmax，MLP 使用 ReLU，秩 1 c_proj 使用 MV_MUL + VV_MUL，前 4 个 LM logit 使用 NPU，最后 6 个 logit 使用 RISC-V 回退路径。

固件为 `adderboard/firmware/adder_130p.c`，采用单阶段 MMIO 数据交换。该模型不适合 FP16：1000 量级的进位检测 MLP 在 FP16 下会产生严重误差。

### dimopep_140p

这是 d=4、1 个 head、head_dim=4、ff_dim=4 的训练模型，使用 `NpuFP32` 和带 FP16 截断的 `NpuDeviceMini`。

数据路径为：

1. ISS 预填充 embedding、RMSNorm、Q/KV 投影、QK 归一化和 RoPE；
2. 阶段 1 固件执行分块 attention、O=Q^T 和 residual；
3. ISS 在两个阶段之间执行 norm2、gate/up 投影；
4. 阶段 2 固件执行 SiLU、gate×up、W_down 和 residual；
5. ISS 标量路径执行 norm_final、LM head 和 argmax。

W_q^T 单独保存在 DRAM 0xD00，以满足 MV_MUL 的矩阵方向。SiLU 使用 V_SIGM + VV_MUL，不需要新增硬件。RMSNorm 和 LM head 使用 RISC-V 标量回退。FP16 准确率约为 99%，与训练模型的 FP16 准确率一致。

## 前向计算概要

### 130p

```text
Embedding → PE → Q/K/V → tiled Attention → c_proj → residual
→ c_fc + bias + ReLU → rank-1 c_proj → residual
→ LM head → argmax
```

关键操作包括 `MV_MUL`、`VV_ADD`、`VV_MUL`、`V_RELU`、`VV_B_SUB_A`、`V_EXP` 和 `S_RECIP`。每个前向步骤约需 120–150 条 NPU 指令。

### 140p

```text
Embedding → RMSNorm → Q/KV → QK Norm → RoPE
→ tiled Attention → O projection → residual
→ RMSNorm → gate/up → SiLU → gate×up → W_down
→ residual → RMSNorm → LM head → argmax
```

每个前向步骤约需 80 条 NPU 指令，主要操作包括 `MV_MUL`、`VV_ADD`、`VV_MUL`、`V_SIGM`、`VV_B_SUB_A`、`V_EXP` 和 `S_RECIP`。

## 参数组成

| 模型 | 主要参数 |
|------|----------|
| 130p | embed_A、embed_B、c_attn、c_proj、c_fc、bias、秩 1 投影和 LM head，共 130 个参数 |
| 140p | embedding、norm1/norm2/norm_final、W_q、W_kv、q_norm、k_norm、W_gate、W_up、W_down 以及 RoPE 表，共 140 个参数 |

W_q^T 是 W_q 的转置视图，不是新增参数，但为了满足 O 投影方向会单独存放在 DRAM 0xD00。

## 使用的 NPU 指令

| Opcode | 名称 | 用途 |
|--------|------|------|
| 7 | `MV_MUL` | 矩阵-向量乘 |
| 8 | `VV_ADD` | 逐元素加法 |
| 11 | `VV_MUL` | 逐元素乘法 |
| 12 | `V_SIGM` | Sigmoid，用于 SiLU |
| 14 | `V_RELU` | ReLU |
| 15 | `VV_B_SUB_A` | score−max |
| 20/21 | `V_RD_DRAM` / `V_WR_DRAM` | DRAM 向量读写 |
| 24 | `M_RD_DRAM` | DRAM 矩阵 tile 加载 |
| 35 | `S_RECIP` | Softmax 中的倒数 |
| 37 | `V_EXP` | 通过 LUT 计算 Exp |

当前固件不使用 `V_TANH`、`V_GELU`、`V_FUNC(SOFTMAX)` 和 `V_FUNC(LAYERNORM)`：分别因为模型不需要 Tanh、SiLU 用已有操作组合、Softmax 使用分块 SPU 路径、RMSNorm 使用 RISC-V 标量路径。

## 关键约束

- pipeline 宽度为 4 个 float，10 类 logit 需要多次遍历或 RISC-V 回退；
- `NATIVE_DIM=4`，矩阵 tile 按 4×4 块加载；
- SiLU 由 V_SIGM + VV_MUL 模拟，RMSNorm 由 RISC-V 标量计算；
- `SPU_ADD_REDUCE` 是累加操作，查询之间必须清零 SRF[1]；
- FP16 下 -1e30 mask 溢出为 -inf 属于正确行为；
- DRAM[0x1F00] 的阶段标志必须写原始 uint32 位模式，不能写 float32。

