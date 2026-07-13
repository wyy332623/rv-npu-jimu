> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# 在 NPU 上的添加文件夹模型——兼容性和状态

AderBoard 质疑呈件可在 NPU 上运行的摘要
模拟器+RISC-V固件堆栈.

## 当前状况

|型号|参数|建筑|NPU 状态|测试|
|---|---|---|---|---|
|** 科斯明斯克语 130p**| 130 |1L GPT, d=4, 2h, ReLU, 罪PE|** 仅端口,FP32|20次测试|
|** 缩略语 140p**| 140 |1L Qune3, d=4, 1h, SwiGLU, RMSNorm, RoPE|++ 端口,FP32+FP16|28次测试|
|科斯明斯肯 66p| 66 |1L GPT, d=4, 2h, ReLU, 罪PE|未尝试| — |
|衣原体03 50p| 50 |1L GPT, d=4, 2h, ReLU, sin PE, 级别... 页:1|未尝试| — |
|xcbtrak  6p (韩语)| 6 |1L Quen, d=2, SiLU, RoPE, 浮点64|未尝试| — |

测试被整合为3个参数化文件(共计49个,参见QQZPROT000XQZ).
所有型号共有6个定型测试案例,每个阶段还有随机的散装测试.

---

## 详细移植模型

### cosminscn 130p(FP32,手工制作)

- 页:1
- ** 计算器类**: QZPROT000XZ(无 FP16 切换)
- ** 建筑**:d=4,2个头×头 dim=2,ff dim=4
- ** 数据路径**: 在NPU上的所有操作——注意(tilled VectoMatRow + SPU softmax),
  MLP(RELU),级别1 c proj(MV MUL + VV MUL),LM头先4个对数(NPU),
  最后6个对数( RISC- V 倒计时)
- ** 软件**:ZPROT000XZ——单相,MMIO数据交换
- ** FP16 可行性**:没有FP16-安全——1000级载体探测MLP
  在FP16精度下产生99.2%的故障
- ** 资料来源**:
  - ZPROT000Z ——黄金参考
  - — DRAM布局
  - ZPROT000Z — RISC-V 固件

### dimopep 140p(FP32 + FP16,已培训)

- 页:1
- ** 重量**:ZPROT000XQZ(140个参数,全部<74 FP16安全)
- ** 计算器类**:ZPROT000XQZ(FP32)和ZPROT0001XQZ(FP16调校)
- ** 建筑**: d=4, 1个头×头 dim=4,ff dim=4
- ** 数据路径**:
  - 国际空间站预填:嵌入式、RMSNorm、XXZPROT000XZ预测、QK规范、ROPE
  - 第1阶段固件:平板注意(VecToMatRow + SPU softmax)+ OQQT + 剩余
  - 基础设施服务缺口:RMSNorm(规范2)、ZPROT000XZ预测
  - 第2阶段固件:SiLU(V SIGM+VV MUL),门×(VV MUL),W 下(MV MUL)+剩余
  - 国际空间站scalar: RMSNorm(标准 最终标准),LM头(嵌入式.T), argmax
- ** 软件**:ZPROT000XZ——两阶段(注意后改为FFN),
  相位旗在 DRAM [0x1F00]
- ** FP16精度**:~99% −火柴模型在FP16训练精度.
- ** 关键设计决定**:
  - W q^T 单独存储于 0xD00 用于 O 投影( MV  MUL 需要移植 W)
  - SiLU = V SIGM + VV MUL(两个已有的操作,没有新的硬件)
  - RMSNorm 和 LM 头使用 RISC- V 缩放
  - 两阶段固件:国际空间站在各阶段之间计算XXZPROT000XZ
- ** 资料来源**:
  - XQ ZPROT000XQZ — 金色参考(纯数字 Quen3)
  - — DRAM布局
  - ZPROT000Z — RISC-V 固件

---

## 前进通道——步进运算符追踪

### 130p(cosminscn 130p) — d=4, 2hxhd2, ff=4, ReLU, 罪PE

|步骤|操作|什麽?|尺寸|重量|
|---|---|---|---|---|
| 1 |V RD DRAM + MV MUL|嵌入的因数化A-B|10×1 — 1×4 / 符号|嵌入 A(10×1),嵌入 B(1×4)|
| 1 |VV ADD 软件|斜体PE| 4 |PE表 (T×4)|
| 2 |M RD DRAM + MV MUL|QKV 预测| 12×4 @ 4 → 12 |c atn(12x4) (中文(简体) ).|
| 3 |VecToMatRow 视频|K^T 瓷砖构造( 每根)|2 — 4×4 管理成果框架| — |
| 3 |MV MUL 音乐|注意分数| 4×4 @ 2 → 4 | — |
| 3 |VV MUL 变压器|缩放x0.25| 4 | — |
| 3 |VV ADD 软件|致病面具| 4 | — |
| 3 |SPU MAX  REDUCE|全球最大值|斜线| — |
| 3 |VV B SUB A + V EXP|分数 - 最大, exp| 4 | — |
| 3 |SPU ADD REDUCDE + S RECIP 软件|全球总和,ZPROT000Z|斜线| — |
| 3 |VV MUL 变压器|Prob = exp × inv sum (中文(简体) ).| 4 | — |
| 3 |VecToMatRow 视频|V^T 瓦片构造|2 — 4×4 管理成果框架| — |
| 3 |MV MUL 音乐|V^T @ prob → ctx (英语).| 4×4 @ 4 → 2 | — |
| 4 |M RD DRAM + MV MUL|O 预测| 4×4 @ 4 → 4 |c proj( 4×4)|
| 4 |VV ADD 软件|剩余添加| 4 | — |
| 5 |M RD DRAM + MV MUL|MLP 输入| 4×4 @ 4 → 4 |c fc(4×4) (中文(简体) ).|
| 5 |VV ADD + V RELU|Bis + ReLU| 4 |c fc bias(4) (中文(简体) ).|
| 6 |M RD DRAM + MV MUL|排名-1点(MLP退出)| 4×1 @ 4 → 1 |c proj u (4×1) (中文(简体) ).|
| 6 |VV MUL 变压器|广播 × v| 4 |c proj v(1×4) (中文(简体) ).|
| 6 |VV ADD 软件|剩余添加| 4 | — |
| 7 |M RD DRAM + MV MUL|排名-1点(LM)| 10×1 @ 4 → 1 |lm u (10×1) (中文(简体) ).|
| 7 |VV MUL 变压器|广播 × v| 4 |lm v(1×4) (中文(简体) ).|
| 7 |RISC-V 缩写|最后6 个对数|共计10|斜角(10)|

** 前进步骤**:~120-150 NPU指示,8个结构化的MV MUL + 平板注意.

### 140p(dimopp 140p) — d=4, 1hxhd4, ff=4, SwiGLU, RMSNorm, RoPE

|步骤|操作|什麽?|尺寸|重量|
|---|---|---|---|---|
| 1 |V RD 内存|嵌入式搜索|4 个/ 符号|嵌入( 10×4)|
| 2 |RISC-V 缩写|RMSNorm (规范1)| 4 |规范1(4)|
| 3 |M RD DRAM + MV MUL|Q 预测| 4×4 @ 4 → 4 |W q( 4×4)|
| 3 |M RD DRAM + MV MUL|KV 投影( K=V)| 4×4 @ 4 → 4 |W kv( 4×4)|
| 4 |RISC-V 缩写|RMSNorm (q  Norm) (中文(简体) ).| 4 |q 规范(4)|
| 4 |RISC-V 缩写|RMSNorm (k Norm) (韩语).| 4 |k 规范 (4)|
| 5 |VV MUL + VV ADD|ROPE 应用|4 个/ 符号|cos  table, sin  table (T×2) (中文(简体) ).|
| 6 |VecToMatRow 视频|K^T 砖块构造|4 — 4×4 管理成果框架| — |
| 6 |MV MUL 音乐|注意分数| 4×4 @ 4 → 4 | — |
| 6 |VV MUL 变压器|缩放x0.5| 4 | — |
| 6 |VV ADD 软件|致病面具| 4 | — |
| 6 |SPU MAX  REDUCE|全球最大值|斜线| — |
| 6 |VV B SUB A + V EXP|分数 - 最大, exp| 4 | — |
| 6 |SPU ADD REDUCDE + S RECIP 软件|全球总和,ZPROT000Z|斜线| — |
| 6 |VV MUL 变压器|Prob = exp × inv sum (中文(简体) ).| 4 | — |
| 6 |VecToMatRow 视频|V^T 瓦片构造|4 — 4×4 管理成果框架| — |
| 6 |MV MUL 音乐|V^T @ prob → ctx (英语).| 4×4 @ 4 → 4 | — |
| 7 |M RD DRAM + MV MUL|O 投影(带QQT)| 4×4 @ 4 → 4 |W q^T (4×4, DRAM 0xD00) (中文(简体) ).|
| 7 |VV ADD 软件|剩余添加| 4 | — |
| 8 |RISC-V 缩写|RMSNorm( 规范2)| 4 |规范2(4)|
| 9 |MV MUL 音乐|门投影| 4×4 @ 4 → 4 |W 门( 4×4)|
| 9 |MV MUL 音乐|向上预测| 4×4 @ 4 → 4 |向上( 4x4)|
| 10 |V SIGM 图像|sigmoid( 门)| 4 | — |
| 10 |VV MUL 变压器|上门| 4 | — |
| 10 |M RD DRAM + MV MUL|向下投影| 4×4 @ 4 → 4 |下行( 4×4 )|
| 10 |VV ADD 软件|剩余添加| 4 | — |
| 11 |RISC-V 缩写|RMSNorm (标准 决赛)| 4 |规范 最终(4)|
| 12 |RISC-V 缩写|LM 头( 嵌入)|4×10 — 10 个对数|嵌入式.T(4×10)|

** 前进步骤**:~80 NPU指示,7 MV MUL + 平板注意.

---

## 重角分解

### 130p - 10倍径,130段

|日记|形状|要件|
|---|---|---|
|内嵌 A| 10×1 | 10 |
|内嵌 B| 1×4 | 4 |
|c 吨数| 12×4 | 48 |
|c 普罗日语| 4×4 | 16 |
|c/ fc 数据| 4×4 | 16 |
|c  fc bias (中文(简体) ).| 4 | 4 |
|c proj u (中文(简体) ).| 4×1 | 4 |
|c proj v (中文(简体) ).| 1×4 | 4 |
|时间( u)| 10×1 | 10 |
|lm v / lm bias 语言| 1×4 + 10 | 14 |
|** 共计**| | **130** |

### 140p - 11个抗震器,140个参数

|日记|形状|要件|
|---|---|---|
|嵌入| 10×4 | 40 |
|规范1| 4 | 4 |
|规范2| 4 | 4 |
|规范( L)| 4 | 4 |
|W q 时| 4×4 | 16 |
|维基月球( kv)| 4×4 | 16 |
|q 规范| 4 | 4 |
|k 规范| 4 | 4 |
|W 门| 4×4 | 16 |
|上调( W)| 4×4 | 16 |
|下调( D)| 4×4 | 16 |
|W q^T (在 DRAM 中)| 4×4 | 16* |
|cos  table + sin  table|Tx4 电话|记录|
|** 共计**| | **140** |

QQ W q^T是W q的视图,不是附加的权重参数. 单独储存
在 0xD00 的 DRAM 中满足 MV MUL 方向的 O 投影.

---

## NPU 指令集( 使用的关联操作)

|操作码|名称|用户|目的|
|---|---|---|---|
| 7 |MV MUL 音乐|两者|MRF × 管道 → 管道(母体)|
| 8 |VV ADD 软件|两者|元素添加|
| 11 |VV MUL 变压器|两者|元素乘数|
| 12 |V SIGM 图像|140页|Sigmoid(用于SiLU计算)|
| 14 |V RELU (法语)|130页|ReLU 激活|
| 15 |VV B SUB A|两者|分数 - 最大( 软马克)|
| 20 |V RD 内存|两者|来自 DRAM 的矢量负载|
| 21 |V WR 数据记录|两者|向量存储到 DRAM|
| 24 |M RD 内存|两者|从 DRAM 装入矩阵牌|
| 35 |S RECIP 软件|两者|对等( 柔软的 ZPROT000XZ)|
| 37 |V EXP 数据|两者|通过 256 进场LUT(软马克斯)退出|

** 未使用**(已有但固件绕行):
- 13 V TANH — 没有模型使用tanh
- 42 V GELU — 140p 代替使用 V SIGM+VV MUL
- 43 XQ ZPROT000XQZ – 硬件软max未使用;固件使用瓷砖 SPU 软max
- 43 XQ ZPROT000XQZ — 不用于 RMSNorm; RISC- V 缩放

---

## 主要制约因素

- ** 管道宽度**:4个浮点=10级对数需要3个通行证或RISC-V倒置
- ** NATIONAL DIM**: 4 (磁盘装载4×4块)
- ** 无SILU操作码**: 通过V SIGM + VV MUL 模拟
- ** 无 RMSNorm 缩写**:RISC-V 缩写计算
- ** SPU ADD REDUCE是累积的**:在查询之间必须零 SRF[1](MMIO写入)
- ** 无 FP16 溢出**: - 1e30 遮盖值溢出至 -inf 在 FP16 (正确行为)
- ** 阶段旗帜编码**: DRAM [0x1F00] 必须使用原始的 uint32 位图案,而不是浮点32

---

## 结构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Test Script (Python)                      │
│  Loads weights, pre-computes embedding/norms/RoPE, fills DRAM   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Phase 1     │  │   Phase 2     │  │  Phase 2+ISS  │
│ Python driver │  │ Replay stream │  │  RISC-V CPU   │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NPU Emulator                                │
│  NpuFP32 (FP32, no truncation)  or  NpuDeviceMini (FP16)        │
│  ~80-150 instructions per forward step                           │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Python directly sends NPU instructions via _push_instruction().
Phase 2 (replay): Captures firmware's instruction stream, replays in Python.
Phase 2 (ISS): ISS runs compiled RISC-V firmware → MMIO → NPU.
```

## 精度:FP32对FP16

|型号|FP32 逻辑 Diff|FP16 逻辑 Diff|FP32 散装|FP16 散装|
|---|---|---|---|---|
|130页| <0.001 |ZPROT000XZ(非FP16安全)| 50/50 (100%) |99%失败|
|140页| <0.001 | <0.76 | 50/50 (~98%) |~90-100%(~99%模型精度)|

140p FP16误差率与模型固有的99.0% FP16精度相符.
在10K测试。 所有决定性( 边界- 接近) 数字都是硬的 —— 模型
通过软马克斯的注意,而不是1000级,学会了FP16安全载体检测
MLP阈值. 130p 完全无法在 FP16 中运行, 因为 1000 级
携带饱和FP16精度的MLP.
