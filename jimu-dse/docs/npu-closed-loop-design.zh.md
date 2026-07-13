> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU 闭环 FW-HW 共振:设计与技能

> ** 观众**:初级——优化代理人(AI);中级——审查、更新和信任该系统的人类专家
> ** 现状**:执行v2
> ** 日期**:2026-06-22

---

## 摘要

此文档描述了自动优化NPU固件的闭路系统. 系统将每个固件优化作为**imbulity supposition**:候选固件表达出如何更好地利用NPU资源的猜测(DRAM带宽,矢量寄存器文件容量,SPU scalar ops,在芯片上MRF瓷砖存储). ** **3轮验证程序**(金本位参考 + 模拟 → DAG审计)是接受或拒绝每个假设的评价职能。

除了指令级优化外,该文件还引入了**意向近似+补偿**的正式框架——一类优化,在进行计算时故意错误地进行,以促成板块化或聚变,然后是更正通过。 这概括了NPU架构的FlashAtthind式推理.

该文件的结构为**双读**:AI代理读作是自主推理的知识库和技能参考;人类专家读作理解,信任,更新系统的能力.

---

## 目录

1. [闭环建筑] (ZPROT000XZ).
2. [NPU数据流模型(代理物理进取)](XXZPROT000XZ).
3. [优化目标与度量 (XXZPROT000XZ).
4. [设计探索空间] (ZPROT000XZ).
5. [意向说明+赔偿](ZPROT000XZ)
6. (ZPROT000XXZ) 互联网档案馆的存檔,存档日期2014-03-05.
7. [技能库设计与导入管道] (XXZPROT000 XZ).
8. [变异协议和回转探测] (XXZPROT000XZ).
9. [附录A:BERT编码器层 Tensor 流图] (XXZPROT000XXZ).
10. [附录B:技能库的DRAM布局] (XXZPROT000XZ).
11. [附录C:赔偿公式参考] (XXZPROT000XZ).

---

## 1. 闭环结构

### 1.1 概览

```
PROBE → ANALYZE → AGENT → VALIDATE → DEPLOY → LOOP
```

|阶段|发生什么事|后端|
|-------|-------------|---------|
|页:1|建立固件, 运行在模拟器上, 时间为以下=2和以下=6, 测量 DRAM 流量 + 指令跟踪 + DAG 集群|模拟器 (XQZPROT000XZ) + DAG (XZPROT00001Z) 模拟器 (ZPROT00001Z)|
|** 加拿大**|比较 XZPROT000XZ DRAM 比率,从 DAG 检测保存的负载对, 计算每个集群的算术强度|津巴布韦|
|页:1|AI代理(QQZPROT00000XQZ)读取DAG + DRAM集群+技能,生成候选补丁到QZPROT00001XQZ|pi + 技能库 (ZPROT000XZ)|
|** 估价**|用候选软件重建固件,再运行模拟器,测量DRAM改进与运行启动基线|模拟器|
|** 就业**|保存候选人以运行目录, 生成 opt DAG 图表以供审计, 恢复下次重复的基准|津巴布韦|

### 1.2 详细循环

```
┌──────────────────────────────────────────────┐
│  Initialize: copy baseline bert_layer.c       │
│  (from jimu-dse/baseline/bert_layer.c)        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  PROBE: build firmware + run emulator        │
│  → DRAM stats at seq=2 and seq=6            │
│  → DAG micro-op graph (dag_agent/)           │
│  → DRAM cluster analysis (Load/Store/FLOPs)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  ANALYZE: classify bottleneck                │
│  → seq6/seq2 DRAM ratio                      │
│  → match to skill trigger pattern            │
│  → build agent context                       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  AGENT: generate candidate firmware patch     │
│  → read DAG: identify save-load pairs        │
│    (dag-analyze.md skill)                    │
│  → apply VRF cache transformation            │
│    (vrf-cache.md skill)                      │
│  → output modified bert_layer.c              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  VALIDATE: rebuild + re-probe                │
│  → DRAM reduction vs run-start baseline      │
│  → save diff against baseline file           │
│  → generate post-opt DAG (dag_iterN/)        │
└──────────────────────┬───────────────────────┘
                       │
               ┌───────┴───────┐
               │               │
           IMPROVED         CONVERGED
               │               │
               ▼               ▼
┌──────────────────┐  ┌──────────────────────┐
│  Deploy: save    │  │  Print summary,      │
│  candidate to    │  │  exit loop           │
│  results/run-*/  │  │                      │
│  Continue iter   │  │  Cleanup: restore    │
│  (incremental)   │  │  baseline file       │
└──────────────────┘  └──────────────────────┘
```

### 1.3 在运行与运行之间

| |运行中|运行之间|
|--|-------------|--------------|
|** 行为**|迭代 N+1 起始于迭代 N 的结果( 递增)|默认 : 复制 QZPROT000XZ 开始|
|** 需要的旗帜**|无——总是递增|ZPROT000Z 继续从上届竞选最佳候选人|
|** 基准**|在运行开始时测量( 第一次重复)|总是与基于文件的基准进行比较|

### 1.4 关键设计点

1. ** 无git依赖**: 基线管理使用ZPROT000XQZ,而非ZPROT0001XQZ. 用于输出代码
2. ** 以文件为基础的基准**:ZPROT000XZ是未优化固件的承诺副本。
3. ** DAG 制导**: 代理读作QQZPROT00000XQZ,在应用优化前识别保存的负载对.
4. ** 时间标记运行目录** : 每跑一次都会创造出所有文物的ZPROT000Z.
5. ** 仅修改了ZPROT000Z**: 特工从不接触仿真器、国际空间站或测试码。

---

## 2. NPU 数据流模型( Agent 的物理透视)

这是代理商对数据如何通过NPU流动的心理模型. 每个指令都会修改这种状态;代理者必须跟踪它以了解优化的原因.

### 2.1 任何点的状态

```
State after N instructions executed:

DRAM[addr]:     [x0 .. x_{H*SL-1}]     — Input X, Q/K/V/W/b matrices, output buffer
                [tiled Q proj]         — Q=Wq×X in VRF_ACC at 0x200
                [tiled K proj]         — K=Wk×X in VRF_ACC at 0x210
                [tiled V proj]         — V=Wv×X in VRF_ACC at 0x220
                
MRF (on-chip):  [W_tile]               — Currently loaded weight tile (NATIVE_DIM²)
                
VRF (on-chip):
  IVRF (mem 5): [NATIVE_DIM floats]    — Current input vector chunk
  AS0  (mem 7): [NATIVE_DIM floats]    — Tile row 0 accumulator
  AS1  (mem 8): [NATIVE_DIM floats]    — Tile row 1 accumulator (if NUM_TILES > 1)
  MUL  (mem 1): [NATIVE_DIM floats]    — Temporary multiply result
  ACC  (mem13): [NATIVE_DIM floats]    — MVM accumulator (bias pre-loaded)
  MFU  (mem 6): [4096 floats]          — VRF cache for intermediates (K, V, Q, Z, SO, etc.)

SRF (SPU):      [64 floats]            — Scalar register file
  SRF[0]: current tile max (for tiled softmax)
  SRF[1]: current tile sum

Pipe:           [P floats]             — Current pipeline value (result of last compute op)
vpipe_a:        [P floats]             — Saved operand A for VV ops

Registers:
  read_vector_mask:  0xFF              — Per-lane read mask
  write_vector_mask: 0xFF              — Per-lane write mask
  precision_mode:    1                 — BFP enabled
  tile_rows:         2                 — NUM_TILES
  tile_cols:         2
  iterations:        seq_len
```

### 2.2 指令语义(数据流视图)

每项指令按其对以上状态** 的影响加以界定:

```
OP_V_RD_DRAM(addr):
  State before: DRAM[addr..addr+P-1] = data
  State after:  pipe[0..P-1] = masked(DRAM[addr..addr+P-1], read_vector_mask)
                vpipe_a = old_pipe (save current pipe as operand A)
  DRAM cost:    P elements read
  Cycles:       ~3 (VMM latency)

OP_MV_MUL:
  State before: pipe = vector v, MRF = matrix W (NATIVE_DIM×NATIVE_DIM)
  State after:  pipe[i] = Σⱼ W[i][j] × v[j]   (dot product per MRF row)
  DRAM cost:    0 (MRF is on-chip)
  Cycles:       ~3 (MVU latency)

OP_VV_MUL:
  State before: pipe = v, vpipe_a = u
  State after:  pipe[i] = v[i] × u[i]
  DRAM cost:    0
  Cycles:       1 (MFU combinational)

OP_V_FUNC(SUB_SOFTMAX):
  State before: pipe = score vector
  State after:  pipe[i] = exp(score[i]) / Σⱼ exp(score[j])
  DRAM cost:    0
  Cycles:       ~6 (SLU pipeline)

OP_V_WR(mem, addr):
  State before: pipe = data
  State after:  VRF[mem][addr..addr+P-1] = masked(pipe, write_vector_mask)
  DRAM cost:    0

OP_SPU_MAX_REDUCE(addr):
  State before: pipe = vector v
  State after:  SRF[addr] = max(v[0..P-1])
  DRAM cost:    0
  Cycles:       1 (combinational tree, register result on next cycle)

OP_SPU_BROADCAST(addr):
  State before: SRF[addr] = scalar s
  State after:  pipe[i] = s (all P elements)
  DRAM cost:    0
  Cycles:       1 (broadcast)
```

### 2.3 代理人必须追踪的资源制约

|资源|大小|类型|页:1|
|----------|------|------|-------|
|记录|524288 浮点数|离芯片|主记. 每 ZPROT000XQZ花费能量+时间.|
|管理成果框架|1 瓦(ND2)|芯片存储器|握着一个基质瓦片 装入新瓷砖覆盖旧瓷砖。|
|VRF (IVRF) (英语).|20480个浮点数|芯片存储器|MVM初始矢量RF. 最大VRF.|
|VRF (AS0) (英语).|1024个浮点数|芯片存储器|添加Sub VRF 0. 用于瓷砖堆积器。|
|VRF (AS1) (英语).|4096个浮点数|芯片存储器|添加Sub VRF 1. 第二瓦堆积器。|
|VRF( 微额供资)|4096个浮点数|芯片存储器|MFU初创VRF. 用作中间件的VRF缓存.|
|VRF( 毛里求斯)|64个浮点数|芯片存储器|临时乘法结果存储.|
|VRF(行政协调会)|256个浮点数|芯片存储器|MVM累积器(bias pre-load).|
|战略成果框架|64个浮点数|芯片记录|SPU scalar 注册文件 。 有点小但很快|
|管道|P 浮点数|管道正则|单向量. 每一个计算操作在这里写。|
|风笛 a|P 浮点数|管道正则|已保存操作 A. 由任何 QQZPROT000XZ 设置|

** 关键限制**: 一次只有一块瓷砖可以在MRF中. 如果需要两块不同的重瓦(如Q和K预测互换),则必须重新装入.

---

## 3. 优化目标与测量

### 3.1 等级目标

|优先权|目标|度量衡|接受情况|何时|
|----------|------|--------|------------|------|
|页:1|数值正确性|ZPROT000Z对黄金|ZPROT000Z(1层FP16)|每个候选人|
|页:1|无回归|现有测试套件|全部通过|每个候选人|
|页:1|DRAM 减少交通量|津巴布韦|减少ZPROT000Z|每个候选人|
|** P2 **|指令计数减少|津巴布韦|减少ZPROT000Z|中学|
|** P3 ** 。|周期数|HDL 模拟周期|任何改进|可用时|

### 3.2 DRAM 成本模型

每个指令的 DRAM 成本在 FP16 元素中 (每个2 字节):

|操作码|每个问题的要素|何时|
|--------|-------------------|------|
|津巴布韦|P(= 自然数据)|每个问题|
|津巴布韦|页:1|每个问题|
|津巴布韦|P× vec 计数|如果 ZPROT000ZZ 则|
|津巴布韦|P× vec 计数|如果 ZPROT000ZZ 则|
|津巴布韦|P2(=瓦)|每个问题|
|津巴布韦|临2|每个问题|
|所有其他国家| 0 |仅限芯片|

总计=P×(V RD count + V WR count)+P2×(M RD count + M WR count)

### 3.3 典型基线DRAM(dim=2,隐藏=4)

|度量衡|下级=2|下级=6|它所揭示的|
|--------|-------|-------|-----------------|
|V RD DRAM 行动| ~180 | ~456 |带有以下 len的缩放(每个位置负载)|
|V WR DRAM 行动| ~60 | ~156 |带有以下 len的缩放(每个位置保存)|
|M RD DRAM 行动| 144 | 144 |常数(重量矩阵,独立于下列)|
|总字节| ~2,144 | ~6,240 |2.9× 比率 -- -- 以下各段的VRF溢额=6|

优化通过缓存中间体减少每个位置的矢量流量
在ZPROT000XQZ(第6点)中。 完全优化后:

|度量衡|下级=2|下级=6|
|--------|-------|-------|
|V RD DRAM 行动| ~60 | ~168 |
|V WR DRAM 行动| ~12 | ~12 |
|总字节| ~1,248 | ~3,744 |

其余的矢量流量是不可减少的: LN QQZPROT000XQZ负载,单位矢量
用于 V.T 重转换,和重量矩阵负载。

---

## 4. 设计探索空间

### 4.1 一级:指令时间安排

|尺寸|检测模式|转变|DRAM 保存|技能|
|-----------|------------------|----------------|-------------|-------|
|** INC折叠**|津巴布韦|用 INC 变体替换配对|~1 ~ ZPROT000 ~%Z 每对|津巴布韦|
|** 循环交换**|磁共振再利用不良的磁共振循环 QZPROT000XZ|交换到 ZPROT000Z|取决于准入模式|津巴布韦|
|** 业务重排**|依赖性分析显示机会停滞|移动 M  RD  DRAM 更早|隐藏延迟而非 DRAM|津巴布韦|
|** 消除死亡代码**|输出从未读取的指令|删除整个序列|变量|津巴布韦|

### 4.2 第二级:VRF缓存(初级优化)

VRF 缓存转换用芯片取代 DRAM 保存的圆盘
复制至 QQZPROT00000XQZ(mem 6,4096元素). 这是主要的
优化管道目标.

|中级|已删除|缓存位置|技能|
|-------------|----------------|----------------|-------|
|**K**每个职位|2 V WR + 2 V RD 每个位置|VRF[6][有×步]|津巴布韦|
|**V**每个职位|2 V WR + 2 V RD 每个位置|VRF[6][K 大小 + pos × 步|津巴布韦|
|每个职位|每个位置2 V RD|VRF[6][K 大小+V 大小+...]|津巴布韦|
|** Z**(注意上下文)|2 V WR + 2 V RD 每个位置|弗朗索瓦[6]|津巴布韦|
|** SO**(自产出)|2 V WR + 2 V RD 每个位置|弗朗索瓦[6]|津巴布韦|
|** LN 刮伤**|2 V WR + 2 V RD 每磅|弗朗索瓦[6]|津巴布韦|
|** GELU产出**|2 V WR + 2 V RD 每个位置|弗朗索瓦[6]|津巴布韦|
|输入**|每个位置重装 3× X|VRF[6] 通过VRF ADDSUB 1|津巴布韦|

### 4.3 第3级:微结构构造

|参数|登记册|勘探范围|什么样的变化|技能|
|-----------|----------|-------------------|--------------|-------|
|精度模式|第20条|0 (FP16) / 1 (BFP) (英语).|DRAM 对准确性|津巴布韦|
|平铺行|第1条|1. 无|MRF 瓦片计数|津巴布韦|
|拼贴|区域组2|1. 无|MRF 瓦片计数|津巴布韦|
|迭代数|区域行动方案3|1..seq len (中文(简体) ).|批量大小|津巴布韦|

### 4.4 第4级:有意估计+赔偿

见[第5节](ZPROT000XZ)。

---

## 5. 意向说明+ 赔偿

### 5.1 一般模式

```
Let O(x) be a gold-standard operator that cannot be tiled.
Let x = [x₀, x₁, ..., x_{T-1}] be tiled input.

Phase 1 — Approximate (per tile, intentionally wrong):
  y_t = tile_O(x_t)        # Compute as if this tile were the whole
  s_t = g(y_t)             # Per-tile statistics (e.g., max, sum)

Phase 2 — Merge (global aggregation):
  global_s = merge(s₀, ..., s_{T-1})   # Combine statistics across tiles
  Uses: SPU.SS_ADD, SPU.MAX_REDUCE     # (accumulate, compare-and-swap)

Phase 3 — Correct (per tile, using global stats):
  y_corrected_t = h(y_t, global_s)     # Apply correction factor
  Uses: SPU.broadcast, VV_MUL, VV_ADD # (scalar→vector, elementwise)

Phase 4 — Verify:
  max_diff(y_corrected, O(x)) < tolerance  # Emulator comparison
```

### 5.2 混凝土:倾斜软max(NPU闪烁模式)

** 标准软体**:ZPROT000XZ
需要全球最大值(数字稳定性)和全球总和。 不能天真一点

** 附带更正的软顶**:

```
Per tile t (processing elements i ∈ tile_t):
  1. tile_max[t] = max(S[i] for i ∈ tile_t)
  2. exp_sum[t] = Σ exp(S[i] - tile_max[t])
  3. Save exp_vals[t][i] = exp(S[i] - tile_max[t]) to DRAM

Global merge (after all tiles processed):
  4. global_max = max(tile_max[0], ..., tile_max[T-1])
  5. For each tile t: correction[t] = exp(tile_max[t] - global_max)
  6. global_sum = Σ correction[t] × exp_sum[t]

Per tile correction (reload exp_vals from DRAM):
  7. P(i) = exp_vals[t][i] × correction[t] / global_sum
      = exp(S[i] - tile_max[t]) × exp(tile_max[t] - global_max) / global_sum
      = exp(S[i] - global_max) / global_sum                           ✓
```

** NPU 要求 **:SPU (XQZPROT000XXZ for tile max,XZPROT0001XZ for exp sum,XZPROT00002Z for 全球合并,XZPROT0003XZ for 更正因子),V EXP(或DRAM中预算的Exp LUT).

### 5.3 赔偿模式目录

|算法|故意错误|更正|NPU 先决条件|状态|
|-----------|------------------|------------|-------------------|--------|
|** 图层|局部向量上的单瓦 QZPROT000XZ|使用 Global QQZPROT000XZ 重算|苏维埃|*                              |
|** 附带软max**|局部矢量的每片软max|使用全局 QQZPROT000XZ 乘以校正因数|SPU 减少+广播|&\ SPU 完成; 需要 V  EXP|
|** 注意**|单片注意QQK|将每瓦的上下文累积到恢复正常状态|SPU + VRF 累积器|软马克之后|
|** BFP 瓦片近似值**|每瓦的精度降低|累积在FP32中,在边界重命名|BFP 模式切换 + SRF 累积器|• 可行|
|** 在瓦片边界上接近GELU**|瓦片边缘的边界错误|预计算校正 LUT|在斯卢的LUT|• 可行|

### 5.4 代理人如何发现赔偿

鉴于:
- 运算符的数学定义( XZPROT000XZ)
- NPU的资源限制(瓦片大小,SRF深度,SPU操作)
- 可用的原始( SPU 减少、 SPU 广播、 元素算术)

特工可以说明:

> "Softmax需要全球统计. 我只能拿一块砖块在MRF。 如果我把分数向量打平,每块瓷砖的软max都错了. 但我可以计算每个瓦片的统计, 把它们在瓦片上合并, 并应用每个元素的校正。 让我得出校正公式..."

这是** 不匹配的模板** 。 代理人的公式来源于:
1. ** 操作员的数学定义**(来自技能库)
2. ** 原始语义**(SPU减少=========== 最大超过矢量;广播=scalar==vector)
3. ** 组成规则**(从内容上看,组成是:XZPROT000XZ)

衍生词在代理输出中是明确的,由仿真器进行**数学验证**.

---

## 6. 代理提示( S)

### 6.1 快速结构(四层)

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Current State (from analyze)                              │
│                                                                      │
│  Firmware source: bert_layer.c (current version, line-numbered)     │
│  Instruction trace: last 200 MMIO ops (annotated with DRAM cost)    │
│  Bottleneck: "V_WR_DRAM + V_RD_DRAM pairs consume 40% of DRAM"     │
│  Resource state at bottleneck point:                                │
│    MRF = W_tile[0][0],  VRF_AS0 = acc_row_0,  VRF_AS1 = acc_row_1  │
│    Pipe = last compute result,  SRF = [0, 0, ...]                   │
│                                                                      │
│  Prior attempts: iter1=FAIL(max_diff=0.5), iter2=PASS(DRAM -8%)     │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: NPU Dataflow Model (the "physics engine")                 │
│                                                                      │
│  (The full dataflow model from Section 2 of this document)          │
│  Key: every instruction transforms the state. Track it.             │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Skill Library (the "toolbox")                             │
│                                                                      │
│  Available skills (matching the bottleneck):                         │
│    dag-analyze — read DAG to identify save-load pairs               │
│    vrf-cache   — replace DRAM save/load pairs with VRF cache        │
│    self-verify — run pytest to check correctness                    │
│                                                                      │
│  Each skill includes: trigger, preconditions, transformation,        │
│  validation hook, cost model, and prior success rate.               │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: Task Specification                                        │
│                                                                      │
│  Goal: Generate candidate firmware patch that:                      │
│    (1) Preserves correctness (P0: max_diff < 0.05)                 │
│    (2) Reduces DRAM traffic or instruction count (P1/P2)            │
│    (3) Explains the reasoning explicitly                            │
│                                                                      │
│  Output format: diff-style patch OR full file                        │
│  State your compensation derivation if using intentional approximation│
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 技能构成

技能可调和。 特工把他们锁起来:

```
Skill A: dag-analyze
  Reads:  jimu-dse/results/dag_agent/micro_op_dag.txt
  Action: Identify DRAM_STORE → DRAM_LOAD pairs by following edges
  
Skill B: vrf-cache
  Depends: A (identified save-load pairs)
  Action: Replace DRAM_WR + DRAM_RD with VREG_MOVE + V_RD from MFU_VRF
```

毒剂探索**技能依赖图**找到影响最高的组合.

### 6.3 透明度

代理人必须明确表示其推理:

```
Step 1: DAG analysis
  Found edge: node 33 (DRAM_STORE to 0x300) → node 150 (DRAM_LOAD from 0x300)
  This is K[0] — saved to DRAM then read back for attention
  
Step 2: Skill matching
  Pattern matches vrf-cache skill (trigger: DRAM save-load pair)
  Precondition check: no intervening write to same address → OK
  
Step 3: Apply transformation
  Replace V_WR_DRAM(0x300) + ... + V_RD_DRAM(0x300) with
  VREG_MOVE(ADDSUB_VRF_0 → VRF[6], offset) + ... + V_RD(VRF[6], offset)
  
Step 4: Cost model
  Before: 2 × P = 16 elements per position
  After: 0 DRAM elements per position
  For seq=6: 96 elements saved = 384 bytes
```

---

## 7. 技能库设计和导入管道

### 7.1 技能格式

技能是 Markdown 文件在 QQZPROT000XZ 。 每个技能包括:

```yaml
name: vrf-cache
version: 1.0.0
category: [isa, fusion]
description: >
  Replace DRAM save-load pairs with on-chip VRF cache operations.
  Used for K, V, Q, Z, SO, LN, GELU, and X intermediates.

trigger:
  pattern: [DRAM_STORE, *, DRAM_LOAD]  # * = any intervening ops
  constraint: |
    addr(store) == addr(load)
    AND no_write_between(store, load, addr)
    AND load_consumes_once(store, load)

preconditions:
  - "The intervening ops do not modify the saved data"
  - "MFU_INITIAL_VRF (mem 6) has sufficient capacity"

transformation:
  before: |
    V_WR_DRAM(addr)
    <intervening ops>
    V_RD_DRAM(addr)
  after: |
    VREG_MOVE(ADDSUB_VRF → VRF[6], offset)
    <same intervening ops>
    V_RD(VRF[6], offset)

cost_model:
  dram_elements_before: 2 × P
  dram_elements_after: 0
  saving: "2P elements per save-load pair"

dependencies: [dag-analyze]
conflicts: []
```

### 7.2 技能文件

|技能|文件|说明|
|-------|------|-------------|
|dag 分析|津巴布韦|读取微操作 DAG 以识别保存的负载对|
|vrf 缓存|津巴布韦|用芯片 VRF 缓存替换 DRAM 圆路|
|自我验证|津巴布韦|固件修改后的自我验证|

### 7.3 如何添加新技能

```
1. Write the skill as a Markdown file in jimu-dse/docs/skills/<category>/
2. Include: name, trigger pattern, preconditions, transformation, cost model
3. Reference any base skills in `based_on:` metadata
4. The agent discovers the skill via the file listing in its prompt
```

---

## 8. 验证协议和后退检测

### 8.1 验证管道

|圆|后端|速度|验证什么|使用|
|-------|---------|-------|-------------------|-----|
| 0 |数字金色|即时|算术正确性|每个候选人|
| 1 |模拟器|~分钟|指令语义、 DRAM 版式、 微量|每个候选人|
|审计|DAG 图表|~s 时|优化DAG后用于ZPROT000Z审查|每个候选人|

HDL弹(Amaranth cycle-accessate sim)可在 QQZPROT000XQZ 上获得.
分支,但不在此 FW 优化管道中。

### 8.2 每个优化类型的接受标准

|优化类型|容忍(最大位元)|页:1|
|-------------------|---------------------|-------|
|指令重排顺序|< 1e-6 (英语)|没有新的数据路径|
|INC 折叠|< 1e-6 (英语)|校验 INC 地址|
|VRF 缓存| < 0.05 |累积顺序更改|
|操作器聚变| < 0.05 |累积顺序更改|
|倾斜计算| < 0.1 |FP16 横扫瓷砖|
|意向性近似值+赔偿| < 0.2 |最有攻击性|

### 8.3 后退探测

```
if max_diff > 0.05:                   REJECT (numerical regression)
if dram_bytes > baseline × 1.1:      REJECT (DRAM regression)
if len(trace) > baseline × 1.1:      WARN (instruction count increased)
```

### 8.4 趋同

当运行启动基线的 DRAM 改进为
少于 QZPROT000XZ( 默认为 15%) 。 典型的趋同发生在
3-5次迭代后总DRAM减少40%.

---

## 附录A:BERT编码器层

这是特工的精神模型 伸缩器流经一个BERT位置。

```
                          X (DRAM[0..hidden_size-1])
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    Q = Wq×X + bq   K = Wk×X + bk   V = Wv×X + bv
    (save Q→0x200)  (save K→0x210)  (save V→0x220)
          │               │               │
          ▼               ▼               │
    load Q (mask=0x0F)  load K (mask=0x0F)│
          │               │               │
          └─────── VV_MUL ─┘               │
                      │                    │
                  score_vec                │
                      │                    │
              V_FUNC(SOFTMAX)              │
                      │                    │
                   prob_vec                │
                      │                    │
                      └─── VV_MUL ─────────┘
                               │
                          context_vec
                               │
                    ┌──────────┘
                    ▼
             self_output = Wso × context + bso
                    │
                    ▼
             res1 = self_output + X
                    │
                    ▼
             LN1: layernorm(res1)
                    │
                    ▼
             FFN_intermediate = GELU(Wi × LN1_out + bi)
                    │
                    ▼
             FFN_output = Wo × FFN_intermediate + bo
                    │
                    ▼
             res2 = FFN_output + X
                    │
                    ▼
             LN2: layernorm(res2) = final output
```

### 临界优化点(在流中标注)

|标签|优化|技能|
|-------|-------------|-------|
|A: X 装入 3x 用于 QQZPROT000XZ|跨预测的 VRF 缓存 X|津巴布韦|
|B: K, V 省载对|在 VRF 中缓存 QZPROT00000Z 而不是 DRAM|津巴布韦|
|C:得分 = 柔和度 = 上下文|通过软max输出流 V|津巴布韦|
|D:上下文 自产出|消除冗余的 ZPROT000ZZ|津巴布韦|
|E: 剩余 = LN|将倾斜投向LN ZPROT000Z|津巴布韦|
|F:FFN中间体 − GELU|将 GELU 输入 FFN MVM|津巴布韦|

---

## 附录B:DRAM 技能库布局

```
DRAM layout for BERT encoder layer (hidden_size=16, NATIVE_DIM=8, num_tiles=2):

Offset  | Content
────────┼──────────────────────────────────────────
0x0000  | X[0..15]  (input vector, seq_len × hidden_size)
0x0010  | X[16..31] (next position, if seq_len > 1)
...     |
0x0014  | Wq (16×16 = 256 elements, tiled as 2×2 tiles of 8×8)
0x0114  | bq (16 elements)
0x0124  | Wk (256 elements, tiled)
0x0224  | bk (16 elements)
0x0234  | Wv (256 elements, tiled)
0x0334  | bv (16 elements)
0x0344  | Wso (256 elements, tiled)
0x0444  | bso (16 elements)
0x0454  | Wi (intermediate FFN, 256 elements, tiled)
0x0554  | bi (16 elements)
0x0564  | Wo (output FFN, 256 elements, tiled)
0x0664  | bo (16 elements)
0x0674  | LN1_gamma (16 elements)
0x0684  | LN1_beta (16 elements)
0x0694  | LN2_gamma (16 elements)
0x06A4  | LN2_beta (16 elements)
...     |
0x0200  | Save area: Q projection result
0x0210  | Save area: K projection result
0x0220  | Save area: V projection result
0x0300  | Save area: self_output + residual
0x0400  | Save area: final output
```

** 长期公约**: 每个 W 矩阵( 隐藏  大小  隐藏  大小) 保存为
(原始内容存档于2019-09-01). ZPROT000Z 子地图 QZPROT00001Z. 平面图:ZPROT0002Z
(原始内容存档于2019-09-21). at DRAM efficial QQZPROT000XZ.

---

## 附录C:赔偿公式

### C.1 倾斜层

```
Given: x = [x_0, ..., x_{T-1}]  (T tiles)

Per tile t:
  sum[t] = Σ x_i
  sumsq[t] = Σ x_i²
  n[t] = len(x_t)

Global:
  N = Σ n[t]
  mean = Σ sum[t] / N
  var = Σ sumsq[t] / N - mean²
  inv_std = 1 / sqrt(var + ε)

Per element:
  y_i = gamma × (x_i - mean) × inv_std + beta
```

### C.2 倾斜软max

```
Given: S = [S_0, ..., S_{T-1}]  (T tiles of score)

Per tile t:
  tile_max[t] = max(S_i)
  exp_sum[t] = Σ exp(S_i - tile_max[t])
  Save: exp_vals[t][i] = exp(S_i - tile_max[t])

Global:
  global_max = max(tile_max[0], ..., tile_max[T-1])
  correction[t] = exp(tile_max[t] - global_max)
  global_sum = Σ correction[t] × exp_sum[t]

Per element (reload exp_vals):
  P(i) = exp_vals[t][i] × correction[t] / global_sum
       = exp(S_i - global_max) / Σⱼ exp(S_j - global_max)
       = exp(S_i) / Σⱼ exp(S_j)  ✓
```

### C.3 一般模式

```
For any reduction operator O that needs global statistics to normalize:

1. Identify the statistics needed: g = {g_0, g_1, ..., g_{k-1}}
   Example: softmax needs g = {max, sum}

2. Identify the per-tile computation of g:
   tile_g[t] = per_tile_g(x_t)

3. Identify the merge function M:
   global_g = M(tile_g[0], ..., tile_g[T-1])
   Example: max merge = max over tiles; sum merge = sum over tiles

4. Identify the correction function H:
   corrected_tile = H(O_tile(x_t), tile_g[t], global_g)
   Example: softmax correction = exp(S - tile_max) × exp(tile_max - global_max) / global_sum

5. Verify: H(O_tile(x), g_tile, M(g_tile, ..., g_T)) == O(x)
   This must hold mathematically. If it doesn't, the compensation is wrong.
```

---

## 参考资料

1. rv-npu 建筑学 — — ZPROT000XZ
2. rv-npu 公司软件指南——ZPROT000XZ
3. Dao等人著"FlashAttention:快速和记忆力 Exactive Exception with IO-Awardness" (2022).
4. JIMU 技术报告——在pyspike-fpga的ZPROT000XZ
