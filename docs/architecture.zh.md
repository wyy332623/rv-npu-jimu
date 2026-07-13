> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU 架构

## 概览

NPU是用于变压器推论的固件驱动SIMD加速器.
RISC-V控制处理器向NPU发送32位MMIO指令词;
NPU通过功能单位执行它们.
该架构对变压器模型中发现的图案进行了优化:
矩阵向量乘法,软马克,层常态,GELU,和剩余加法.

### 两个指令发布模式

|模式|机制|使用大小写|
|------|-----------|----------|
|** 指示**|每人写到ZPROT00000Z摊位,直到ZPROT00001Z. 公司民意测验,|配置 (S WR),简单的序列,调试. (原始内容存档于2019-09-29). ZPROT0001XQ Z.|
|** 基于钱的**|多个指令是在没有按指示设置摊位的情况下写成的,然后通过QQZPROT000XXZ进行解剖. ZZPROT0001Z 登记每个单位繁忙状态的音轨.|计算重序(MVM,软max,剩余). (原始内容存档于2019-09-02). ZPROT000+ZPROT0001+ZPROT00002+Z.|

在每门教学模式中,每门教学必须在下门教学之前完成
开始。 在链式模式中,一个链式流经
没有 DRAM 保存的圆路和独立的管道注册
链条可以在硬件中重叠(SMC——同步多切宁).

### 关键设计原则

- ** 软件驱动**:所有计算均用MMIO写法表示。
  RISC-V CPU. 互联网档案馆的存檔,存档日期2013-03-02. NPU没有指令缓存或序列器.
- ** 同步执行(执行指令模式)**:每个指令运行
  在下一个开始前完成。 没有管道,没有平行的调度。
- ** 同步执行(链模式)**:链内指令
  通过管道登记; 多个链条可以运行
  不同功能单位同时使用.
- ** 数据并行**:所有单位都使用QQZPROT000XQZ元素的矢量。
- ** 过渡优化**:《国际审计准则》直接支持为以下目的开展的行动:
  注意,进取, 和正常化层。

### 系统图表

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RISC-V CPU (firmware)                         │
│                                                                          │
│    Mode A: Per-instruction                          Mode B: Chain-based │
│    ──────────────────────                          ──────────────────── │
│    send(SI(OP_M_RD_DRAM))                          npu_chain_begin()    │
│    while(STATUS != DONE);     ← stal               send(...)            │
│    send(SI(OP_MV_MUL, 0, 0))                       send(...)            │
│    while(STATUS != DONE);     ← stal               npu_chain_commit()   │
│    send(SI(OP_V_WR, ...))                          while(CHAIN_STATUS)  │
│    while(STATUS != DONE);     ← stal               ← per-unit busy      │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │ MMIO (32-bit instruction words)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Instruction Decoder                          │
│                                                                  │
│  Per-instruction path:  decode → execute → STATUS=DONE           │
│  Chain path:            decode → chain FIFO → INST_ISSUE commits │
│                                                                  │
│  Pipeline register threads values between instructions:          │
│    V_RD → pipe, MV_MUL → pipe, V_FUNC → pipe, V_WR captures    │
│                                                                  │
│  CHAIN_STATUS tracks per-unit busy (bit0=VMM, bit1=MMM, bit2=   │
│  MVU).  No scoreboard, no hazard checking in Python emulator.    │
└──────┬─────────────────────┬────────────────┬──────────┬──────────┘
       │                     │                │          │
       ▼                     ▼                ▼          ▼
┌──────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐
│  TMM         │   │  MVU         │   │  MFU     │   │  SLU         │
│  (DRAM ↔     │   │  (matrix-    │   │  (GELU,  │   │  (softmax,   │
│   register)  │   │   vector)    │   │   add,   │   │   layernorm) │
│  ┌─────────┐ │   │  ┌─────────┐ │   │  ┌─────┐ │   │  ┌─────────┐ │
│  │ V_RD    │ │   │  │ MV_MUL  │ │   │  │GELU │ │   │  │ Max     │ │
│  │ V_WR    │ │   │  │         │ │   │  └─────┘ │   │  │ │       │ │
│  │         │ │   │  │ (C lib  │ │   │  ┌─────┐ │   │  │ Exp    │ │
│  │ M_RD    │ │   │  │  call)  │ │   │  │add/ │ │   │  │ │       │ │
│  │ M_WR    │ │   │  └─────────┘ │   │  │sub  │ │   │  │ Sum    │ │
│  └─────────┘ │   │              │   │  └─────┘ │   │  │ │       │ │
│              │   │              │   │  ┌─────┐ │   │  │Normal  │ │
│              │   │              │   │  │tanh/ │ │   │  └─────────┘ │
└──────────────┘   │              │   │  │mul  │ │   │              │
                    └──────────────┘   │  └─────┘ │   └──────────────┘
                                       └──────────┘

Register Files (VRF):
┌────────────┬─────┬──────────┬────────────────────────────┐
│ Bank       │ ID  │ Elements │ Connected To               │
├────────────┼─────┼──────────┼────────────────────────────┤
│ DRAM       │  0  │  524288  │ Off-chip memory            │
│ MULTIPLY   │  1  │      64  │ Temporary multiply storage │
│ MATRIX_RF  │  4  │ 128×128  │ Weight tile (MRF)          │
│ MVM_INIT   │  5  │   20480  │ MVU input vector           │
│ MFU_INIT   │  6  │    4096  │ MFU input / VRF cache      │
│ ADDSUB_0   │  7  │    1024  │ MFU AddSub operand A       │
│ ADDSUB_1   │  8  │    4096  │ MFU AddSub operand B       │
│ MVM_ACC    │ 13  │     256  │ MVM accumulator            │
│ VEC_TO_MAT │ 18  │       —  │ Vector → matrix row buffer │
└────────────┴─────┴──────────┴────────────────────────────┘

Data Flow:
  DRAM ──TMM──► VRF │ MRF ──MVU──► pipeline ──MFU/SLU──► VRF ──TMM──► DRAM
                     │                  │
                     └── VREG_MOVE ─────┘  (VRF-to-VRF moves)
```

### 管道登记册

"Pipeline retainment"(或称QXQZPROT000XQZ / XQZPROT0001XQZ)是一个**单向矢量-宽
a. 包含指令间实际计算结果的登记册。
每一份计算指令都将其输出写入管道登记簿.
下一项指示从中读出,是其行动之一:

```
MV_MUL:  pipeline = Σ(MRF[row][c] × VRF[input][c])
VV_ADD:  pipeline = vpipe_a + pipeline   (reads previous pipeline output)
SOFTMAX: pipeline = softmax(pipeline)
V_WR:    VRF[dst] = pipeline             (capture result to VRF)
```

### SerDes 模式( SLU)

SerDes (XQZPROT000XQZ) 模式在XQZPROT0001XQZ时使用.
硬件只有 QZPROT000XZ 平行 FP16 数据路径, 但矢量可能
包含更多的元素。 在 SerDes 模式中, QQZPROT000QZ 将矢量分为
零星的ZPROT000Z元素 对于**softmax**:它累积最大和总和
跨越所有块,然后做一个最后的正常化通过。 对于** layernorm **:它
跨块累积和平方圆,计算全局
差异,然后使所有块恢复正常。

## 职能单位

### TMM — 十进制内存管理器

两个独立的子单位在 DRAM 和注册文件之间移动数据:

- ** VMM**:向量转移(ZPROT000XZ,ZPROT0001Z,INC变体).
  通过瓦片计数登记册支持多向量流。
- ** MMM**:基质瓦片转让(ZPROT000XZ,ZPROT00001Z)。

### MVU——矩阵变量乘法单元

计算MRF行和输入矢量之间的点产品.
支持用于平板马图的积累模式( XZPROT000XZ)
多个重瓦柱。 MVM累积器 VRF (XXZPROT000XXZ)
在瓦柱迭代之间持有部分金额。

数字计算委托给ZPROT000XQZ.
Python代码设置了操作器,通过ctype调用C.

### MFU——多功能股

3个具有基于能力的路由:
- MFU 0: GELU (基于LUT) (英语).
- MFU 1: 向量 XQ ZPROT000XQZ
- MFU2: tanh, XQ ZPROT000XQZ, 乘数

算术被委托给C图书馆的功能(ZPROT000XQZ,ZPROT0001XQZ,ZPROT0001XQZ,).
(原始内容存档于2018-09-29). ZPROT000XZ, ZPROT00001Z. Python 模式选择 MFU 和路线
操作。

### SLU — Softmax / LayorNorm 单位

一次ZPROT000XQZ发送中,
- ** 软件**:最大减少量 → exp(x - 最大) → 总计 → 正常
- ** LayerNorm**:总和 = 平均值 = 差异 = inv sqrt = 比例+班次

支持 SerDes 模式 当 QQZPROT000XZ, 累积中间
在最终正常化通过之前,统计跨块。

算术被下放到C库功能(ZPROT000XQZ,ZPROT0001XQZ).

## 注册文件

### 矢量注册文件( VRF)

|银行|身份证|要件|目的|
|------|----|----------|---------|
|津巴布韦| 0 | 524288 |外部内存|
|津巴布韦| 1 | 64 |临时乘数存储|
|津巴布韦| 4 | 128×128 |重量矩阵瓦(MRF)|
|津巴布韦| 5 | 20480 |MVU 输入向量|
|津巴布韦| 6 | 4096 |MFU 输入 / VRF 缓存|
|津巴布韦| 7 | 1024 |MFU AddSub 操作 A|
|津巴布韦| 8 | 4096 |MFU AddSub 操作 B|
|津巴布韦| 13 | 256 |MVM 累积器|
|津巴布韦| 18 | — |矢量 — 矩阵行缓冲器|

### 矩阵注册文件( MRF)

单张ZPROT000Z牌. 从DRAM通过QQZPROT0001XQZ装入.
一次只有一块瓷砖可以居住.

## 执行模型( Python 模拟器)

Python 模拟器执行两种指令发布模式:

### 按指示模式

1. Firmware 向 QQZPROT000XZ (MMIO 抵消 0x00) 写入指令单词.
2. 模拟器解码32位的opcode和操作.
3. 模拟器调用适当的功能单元方法,它可能
   通过ctypes调用XZPROT000XXZ来进行算术.
4. 指令同步完成 。
5. ZPROT000XZ(0x04)成为ZPROT00001Z.
6. 固件读作 QQZPROT000XXZ , 在下次写入前确认完成 。

### 链式

1. Firmware 向 QQZPROT000XZ 撰写多个指令词
   民意调查,
2. 每个指令按顺序同步执行(模拟器有
   ,与管道连接值之间
   连续指示.
3. Firmware 写着 ZPROT000Z 来执行链条。
4. 公司民意调查(ZPROT000XZ)(0x0C),
   表示所有功能单位已完成。

### 与硬件模型的差异

Python仿真器为单线形,无管状. 每个指令
运行到下一个开始前完成。 没有循环计数器
没有危险探测, 也没有平行发送。 链式模式存在于
仅在硬件执行中核查正确性
在链条内针对不同的功能单位,并在其中执行
平行(详见规格.md§3.3)。

## 执行:Python vs C

|构成部分|语言|页:1|
|-----------|----------|-------|
|** RISC-V 国际空间站**(ZPROT000XZ)|Py|~1500线. 运行 RV64IM ELF 二进制指令 。|
|** NPU 设备模型** (XXZPROT000XZ)|Py|功能模拟器. 处理MMIO,注册文件,指令解码,DRAM传输.|
|** 指示解码器**|Py|解码32位指令并发送到相应的功能单元方法.|
|** TMM** (微米+微米)|Py|DRAM QQ ZPROT000XZ 传送 地址自动递减逻辑.|
|** MVU**|Python (控制) + C (数学)|Python设置了操作符;通过ctypes调用QQZPROT000XZ.|
|** MFU**(GELU, 添加, 子, mul)|Python (控制) + C (数学)|Python选择了MFU和路由操作;算术调用C库(XQZPROT000XQZ,XQZPROT0001XQZ,XQZPROT0002XQZ,XQZPROT0003XQZ).|
|** SLU** (软体、层色)|Python (控制) + C (数学)|Python管理SerDes还原环路;算术呼叫C库(QQZPROT000XZ,XZPROT0001~Z).|
|**BERT固件**(ZPROT000XZ)|C (RV64IM) (英语).|由QQZPROT000XQZ编译,运行于Python国际空间站.|
|** Kernel 图书馆**(ZPROT000XZ ~ ZPROT0001Z)|C(第86页)|** 后端计算引擎。 ** 通过ctype调用所有数字操作.|
|** 黄金参考**(ZPROT000XZ)|Py|pytest用于端对端验证的纯numPy执行. 不是模仿者的召唤|
|** 试验**|Py|在ZPROT000XZ地区|

> ZPROT000Z是强制性的。 ** 如果它不存在, 模拟者
> 启动时加高 QZPROT000QQZ , 无法运行 。 这不是一个
> 独立的金色参考——它是核心算术后端.

## 时间模型

Python 仿真器( QQZPROT000XXZ) 是一个 ** 功能模型, 不包含
时间**。 写给 QQZPROT000QZ 的指令会同步执行
在下一个开始前完成:

```python
def _push_instruction(self, inst):
    self._status = STATUS_BUSY
    self._execute(...)      # ← entire instruction finishes here
    self._status = STATUS_DONE
```

没有循环计数器,没有管道阶段重叠,没有平行发送
跨功能单元,没有争议模型。 ZPROT000Z方法为
没有 文件顶端的评论称:

> *"用于循环精确模拟,使用sim.backend verilator代替."*

模拟器适合:
- ** 功能正确性**:验证固件逻辑产生权利
  数字产出
- ** DRAM流量分析**:总字节数
- ** DAG 分析**:从指令痕迹中绘制的依赖性图表

它的**不**型号:
- 管道危险或摊位
- 性能(周期、FLOPS、耐久性)
- 职能单位之间的平行关系
- 时间依赖错误

---

## 与相关加速架构的比较

这个NPU与几个当代和
研究加速器,但微结构、尺度和目标不同
部署。 下表比较了关键建筑特征.

### 微软脑波(ISCA 2018)

Brainwave是一个云级FPGA基于DNN的服务平台. 关键差异 :

|外观|脑波|这个 NPU|
|--------|-----------|----------|
|执行情况|Intel Stratix 10 FPGA 覆盖|Python 模拟器|
|计算核心|Systolic 阵列/ 抗热MVU|序列点产品引擎(1 XQ ZPROT000XQZ)|
|矩阵大小|任意( 高压平板)|单张ZPROT000XZ瓦(N=LANES)|
|发送主机|x86 CPU 超过 PCIe|通过MMIO的 RSC-V芯片|
|软马克/ LN|微编码序列|伪造的 ZPROT000Z 指令|
|缩放|数据中心织物|单芯片研究|
|内存等级|深 FPGA BRAM 结构|平面 DRAM + XZPROT000XZ 注册文件|

### Groq LPU (ISCA 2021 / Hot Chips 2023) (英语).

Groq的语言处理股是一个决定式的 高压电流结构
带有编译时间表的数据流结构。 关键差异 :

|外观|格罗克LPU|这个 NPU|
|--------|----------|----------|
|时间安排|编译时间静态时间表|运行时间固件发送|
|计算模型|Tensor 溪流(SIMD 横跨车道+时间)|矢量顺序( 逐次)|
|危险模式|无 — 所有依赖关系在编译时得到解决|每条链的记分板(XXZPROT000XZ)|
|内存|分布的SRAM 瓦片( 总计 ~ 230MB)|集中的 DRAM + VRF 注册文件|
|决定主义|周期确定性(无仲裁)|通过运行时危险探测依赖数据|
|汇编|全部静态排程(温度 + 组装 + 时间)|带有内置 MMIO 指令的固件|
|职能单位|320 SXM模块(每个模块都有MAC + mem)|MVU + MFU (3) + SLU + TMM|

### FSA — Fused Systolic 关注

FSA(arXiv:2507.11331)" 系统注意:在a内引信闪光注意
单音节阵列”, [XQZPROT000XQZ] (XQZPROT0001XQZ).
是一个运行整个 FlashAttention 的增强的符号数组架构
(Q×K^T → 软max → S×V)在一个没有外向量的单音节阵列上
单位。 它实现精细的元素 重叠的注意力
操作,在保留原始FP的同时最大限度地利用数组
闪存的操作顺序。

|外观|自由军|这个 NPU|
|--------|-----|----------|
|计算引擎|所有关注阶段的单音节阵列|序列式MVU + MFU + SLU|
|软马克|符号阵列上的元素|分解 QZPROT000Z 指令|
|重叠|精细的、元素化的引信|操作(载量)|
|执行情况|可合成RTL(16nm,1.5GHz)|Python 函数模拟器|
|可编程性|自定义内核( 自定义内核)|带有MMIO指令的RISC-V固件|
|倾斜战略|FlashAttenty 风格( 跨SRAM 银行)|平铺式(重瓦×位置)|

### 关键建筑差异摘要

|特性|脑波|格罗克LPU|自由军|这个 NPU|
|---------|-----------|----------|-----|----------|
|家具|FPGA 覆盖|自定义 ASIC (7nm)|自定义 ASIC( 模拟)|Python 模拟器|
|调度模式|CPU 驱动|静态表(数据流)|CUDA 类似内核发射|教条+基于链(双模式)|
|点产品|符号阵列|SIMD MAC 阵列|丝状+引信软max|序列(1 XQ ZPROT000XQZ)|
|正常化|微编码|SIMD 元素|在线引信(在GEMM中)|伪造的 ZPROT000Z HW|
|危险检测|无( 顺序)|无(编译时间)|无( 内核同步)|记分牌 (ZPROT000XZ)|
|芯片存储|BRAM银行(~10MB)|分布式SRAM(~230MB)|SRAM 等级(银行)|VRF + MRF(~64KB) 变压器|
|目标部署|云推论|LLM 推论( 数据中心)|LLM 预填( 研究)|单芯片研究|
