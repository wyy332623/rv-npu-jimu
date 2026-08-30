# NPU 移植计划——已完成

两个 AdderBoard 挑战模型都已移植到 NPU 模拟器 + RISC-V 固件栈。本文记录最终架构和各步骤的完成情况。

## 状态：完成 ✓

| 步骤 | 130p（手工构造） | 140p（训练得到） |
|------|------------------|------------------|
| 1. DRAM 布局 | `npu_dram_layout.py` | `npu_dram_layout_140p.py` |
| 2. Golden reference | `golden_130p.py` | `golden_140p.py`（纯 NumPy Qwen3） |
| 3. NPU 指令流 | `test_phase1b_full.py`（9 项） | `test_140p_phase1.py`（8 项） |
| 4. RISC-V 固件 | `adder_130p.c`（单阶段） | `adder_140p.c`（两阶段） |
| 5. ISS 集成 | `test_phase2_iss.py`（6 项） | `test_140p_phase2.py`（5 项） |
| 加分项：FP16 路径 | 不满足 FP16 安全要求 | `test_140p_fp16.py`（4 项） |

## 130p 架构（仅 FP32）

```text
输入：22 个 token 的提示（a=0..9，LSB-first；b=0..9，LSB-first）
输出：11 个自回归步骤（和，LSB-first）

数据流（最后 6 个 LM logit 除外都在 NPU 上执行）：
Embedding → PE → Q/K/V → Attention → c_proj → residual
→ MLP c_fc + ReLU + bias → MLP 秩 1 c_proj → residual
→ LM head（4 个 logit 使用 NPU，6 个使用 RISC-V）→ argmax

固件：单阶段，通过 MMIO 窗口交换数据
关键操作：MV_MUL、VV_ADD、VV_MUL、V_RELU、VV_B_SUB_A、V_EXP、S_RECIP
```

## 140p 架构（FP32 + FP16）

```text
输入：24 个 token 的提示（a=0..9 + 2 个分隔符 + b=0..9）
输出：11 个自回归步骤

ISS 预填充（Python）：Embedding → RMSNorm(norm1) → W_q、W_kv → QK 归一化 → RoPE
阶段 1 固件：Attention → O=Q^T → residual → 将 attn_res 写入 DRAM
ISS 间隔：从 DRAM 读取 attn_res → RMSNorm(norm2) → W_gate、W_up → 写入 S_BASE2
阶段 2 固件：SiLU → gate×up → W_down → residual → 写入 FW_LAST_H
ISS 标量路径：读取 last_h → RMSNorm(norm_final) → LM head → argmax

固件：两阶段，阶段标志位于 DRAM[0x1F00]，使用原始 uint32
关键操作：MV_MUL、VV_ADD、VV_MUL、V_SIGM、VV_B_SUB_A、V_EXP、S_RECIP
```

## 140p 的关键设计决策

### 1. 单独保存 W_q^T（DRAM 0xD00）

`MV_MUL` 计算 `MRF @ pipeline`，而 golden reference 的 O 投影计算 `ctx @ W_q`。由于 W_q 不一定对称，`W_q @ ctx` 不等于 `ctx @ W_q`。单独保存 `W_q^T` 后，`W_q^T @ ctx = ctx @ W_q`。

### 2. 两阶段固件

ISS 在阶段 1 运行前不知道 attention residual，因此不能预先计算 norm2/gate/up。Python ISS 在两个阶段之间从 DRAM 读取 `attn_res`，计算 norm2/gate/up，写入 `S_BASE2`（0x3000），然后启动阶段 2。

### 3. 阶段标志使用原始 uint32

C 固件通过 `npu_read_reg()` 将 DRAM 字节解释为原始整数。写入 `1.0f` 得到的是位模式 `0x3F800000`，不是整数 1。阶段 1 必须写 `0x00000000`，阶段 2 必须写 `0x00000001`。

### 4. SiLU = V_SIGM + VV_MUL

无需新增硬件，两个已有 NPU 操作即可计算 `sigmoid(x) × x`。NpuFP32 测试已验证结果与 NumPy SiLU 精确匹配。

### 5. 查询之间重置 SRF

`SPU_ADD_REDUCE` 和 `SPU_MAX_REDUCE` 是累加操作，值会跨查询保留：

- SRF[0]（最大值）：用 `SPU_ADD_REDUCE(-inf)` 重置；
- SRF[1]（总和）：通过 `npu_write_reg(NPU_SRF_BASE + 4, 0)` 清零。

## 模拟器和固件改动

| 改动 | 文件 | 原因 |
|------|------|------|
| 添加 `OP_V_SIGM` 处理器 | `emulator/npu_device_mini.py` | 原先会被静默忽略 |
| 添加 `OP_V_TANH` 处理器 | `emulator/npu_device_mini.py` | 完整性 |
| 在 ctypes 初始化中加入 `sigmoid` | `emulator/npu_device_mini.py` | V_SIGM 需要调用 |
| 通过 MMIO 重置 SRF | `adderboard/firmware/adder_140p.c` | SPU 归约会累加 |
| 添加 `NpuFP32` 类 | `emulator/npu_fp32.py` | FP32 验证模式 |

## 开发期间修复的错误

| 错误 | 表现 | 修复 |
|------|------|------|
| VV_A_SUB_B / VV_B_SUB_A 总是执行加法 | `scores - max` 变成 `scores + max` | 修复 `_vv_add_sub()` |
| V_EXP 为空操作 | `exp()` 返回 0 | 将 V_EXP 加入 `_v_activation()` |
| SPU 归约覆盖而非累加 | 只保留最后一个 tile 的最大值/总和 | 改为累加 |
| S_RECIP、S_SQRT 是空实现 | `inv_sum` 始终为 0 | 实现 `_spu_func()` |
| V_SIGM 未实现 | SiLU 结果错误 | 添加 V_SIGM 处理器 |
| 140p 查询间未重置 SRF[0] | CTX[1] 使用错误最大值 | 用 `SPU_ADD_REDUCE(-inf)` 重置 |
| 140p 查询间未清零 SRF[1] | Softmax 总和错误 | 查询间增加 MMIO 清零 |
| 阶段标志写为 float32 | C 的 `!= 1` 判断失败 | 写入原始 uint32 位模式 |
| ISS MMIO 窗口过小 | SRF 窗口与 DRAM 重叠 | 扩展到 0x10000 |

