> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU 移植计划——已完成

两个AdderBoard挑战模型都移植到 NPU 模拟器 + RISC - 五
固件堆叠。 该文件记录了最后架构和
每一个步骤完成。

## 现况:完成

|步骤|130p(手工制作)|140p(训练)|
|---|---|---|
|1. DRAM布局|津巴布韦|津巴布韦|
|2. 黄金参考|津巴布韦|ZPROT000XZ( 纯数字 Quen3)|
|3. NPU教学流|ZPROT000Z(9次测试)|ZPROT000Z(8次测试)|
|4. RISC-V固件|ZPROT000Z(单相)|ZPROT000Z(两阶段)|
|5. 基础设施服务部门一体化|ZPROT000Z(6次测试)|ZPROT000XQZ(5个测试)|
|奖励: FP16 路径|❌(非FP16安全)|ZPROT000Z(4次测试)|

---

## 130p 建筑(仅限FP32)

```
Input: 22-token prompt (a=0..9 LSB-first, b=0..9 LSB-first)
Output: 11 autoregressive steps (sum LSB-first)

Data flow (all on NPU except last 6 LM logits):
  Embedding → PE → Q/K/V → Attention(tiled, 2h×hd2) → c_proj → residual
  → MLP c_fc + ReLU + bias → MLP rank-1 c_proj → residual
  → LM head (4 logits on NPU, 6 via RISC-V) → argmax

Firmware: single-phase, MMIO window for data exchange
Key ops: MV_MUL, VV_ADD, VV_MUL, V_RELU, VV_B_SUB_A, V_EXP, S_RECIP
```

## 140p 建筑 (FP32 + FP16)

```
Input: 24-token prompt (a=0..9 LSB-first + 2 separators + b=0..9 LSB-first)
Output: 11 autoregressive steps (sum LSB-first)

ISS pre-fill (Python):
  Embedding → RMSNorm(norm1) → W_q, W_kv → QK norms → RoPE

Phase 1 firmware (NPU):
  Attention (tiled, 1h×hd4) → O=Q^T → residual → write attn_res to DRAM

ISS gap (Python):
  Read attn_res from DRAM → RMSNorm(norm2) → W_gate, W_up → write to S_BASE2

Phase 2 firmware (NPU):
  SiLU(gate) via V_SIGM+VV_MUL → gate×up via VV_MUL → W_down via MV_MUL
  → residual + write last_h to FW_LAST_H

ISS scalar (Python):
  Read last_h → RMSNorm(norm_final) → LM head (embedding.T) → argmax

Firmware: two-phase, phase flag at DRAM[0x1F00] (raw uint32)
Key ops: MV_MUL, VV_ADD, VV_MUL, V_SIGM, VV_B_SUB_A, V_EXP, S_RECIP
```

---

## 140p的关键设计决定

### 1. W q^T 单独存储( DRAM 0xD00)

MV MUL 计算QQZPROT000XZ,但黄金参考计算
O型投影机的ZPROT000XQZ. 由于W q不是对称的,
(原始内容存档于2019-09-03). ZZPROT000 QZ. 在单独的 DRAM 地址上存储 QZPROT0001Z
校正: XQZPROT000XQZ.

### 2. 双阶段固件

ISS无法在第一阶段运行前预先计算QQZPROT000XXZ,因为它
不知所终 Python 国际空间站的缺口读作
来自DRAM的ZPROT000XXZ,
至 ZPROT000XQZ(0x3000),然后发射第2阶段.

### 3. 阶段旗为 Raw uint32

C固件使用将 DRAM 字节解释为 QZPROT000XQZ
原始整数。 写入QQZPROT000XXZ(位图型为0x3F800000) QQ 1 in C.
必须写入原始的 uint32 位图案: 0x000000000 用于第一阶段 。
0x00000001 用于第2阶段.

### 4. 西卢 = V SIGM + VV MUL

不需要新的硬件。 两个现有的NPU操作计算QQZPROT000XZ.
通过NpuFP32测试(精确匹配于numpy silu)验证无误.

### 5. 查询之间的 SRF 初始化

SPU ADD REDUCE 和 SPU MA REDCE 是累积的—— 数值持续
交叉查询。 修补 :
- SRF[0](最大值):用 SPU ADD REDUCE(-inf) 部队重置到-inf
- 战略成果框架[1](和数):0通过XZPROT000XZ

---

## 模拟器更改

|变动|文件|原因|
|---|---|---|
|添加了 QZPROT000 QZ 处理器|津巴布韦|被无声忽略了|
|添加了 QZPROT000 QZ 处理器|津巴布韦|完整性|
|类型设置中的 ZPROT000XZ|津巴布韦|V SIGM需要这个|
|通过MMIO重置战略成果框架|津巴布韦|累积SPU 减少|
|NpuFP32级|津巴布韦|FP32 核查模式|

---

## 开发过程中修复错误

|错误|示意图|修补|
|---|---|---|
|VV A SUB B/VV B SUB A总是添加|ZPROT000Z 计算 ZPROT00001Z|固定的 ZPROT000Z|
|V EXP 无效|ZPROT000Z 返回 0|将 V EXP 添加到 ZPROT000 Z|
|SPU 减少过度写作而不是积累|只有最后一块牌子的ZPROT000Z|累计制造 ZPROT000Z|
|S RECIP, S SQRT 是根根|ZPROT000Z 总是0|已执行|
|V SIGM 未执行|SILU计算错误|添加了 V  SIGM 处理器|
|SRF[0] 不在查询之间重置(140p)|CTX[1] 使用错误的最大值(3.357 vs 3.263)|使用 SPU ADD REDUCE(-inf) 代替 SPU MA  REDUCE(-inf)|
|SRF[1] 查询之间没有零(140p)|不正确的软马克总和(2.196对1.196)|在查询间添加 MMIO 0|
|相位旗为浮点32 (140p)|C 比较 QZPROT000 Z 失败|写入原始的 uint32 位图案|
|国际空间站MMIO窗口太小|SRF 窗口重叠 DRAM|延伸至 0x1000|
