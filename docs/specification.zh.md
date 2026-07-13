> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU 规格

## 1. 概览

NPU是用于变压器推论的固件驱动SIMD加速器.
RISC-V控制处理器发送32位 MMIO 指令词; NPU
同步执行,并通过状态登记册发出完成信号。

### 1.1 关键设计原则

- ** 软件驱动**:所有计算均以32位的序列表示
  MMIO来自一个RISC-V CPU. NPU没有指令缓存或序列器.
- ** 同步执行**:每项指令在
  下一个开始。 没有管道,没有平行的调度。
- ** 数据并行**:计算单位在QQZPROT000XZ元素的矢量上运行.
  内部算术使用带有可选块浮点(BFP)的FP16.
- ** 转换优化**:《国际审计准则》直接支持矩阵向量乘法,
  软max,层态化,GELU,剩余添加,以及注意力遮掩.

### 1.1.1a 术语: 自然数据与语言

两个相关但不同的参数定义了矢量宽度:

|任期|范围|定义|
|------|-------|-----------|
|津巴布韦|固件( 编译时间宏)|逻辑矢量中的元素数量 。 通过 ZPROT000XXZ 设定每个固件的构造 。 所有 VRF 传输,MV MUL 迭代,和 DRAM 脚步使用此值.|
|津巴布韦|硬件(参数)|物理硬件中的平行FP16数据路径数. 设计时固定. 所有计算单元(MVU乘数,MFU管道,SLU ALUs)都有QQZPROT000XQZ副本.|

** 关系:** ZPROT000XZ必须是ZPROT0001Z的倍数. 何时
ZPROT000XZ, SLU以SerDes模式运行(第4.3节) -
它将矢量分割成 XZPROT000XZ 块并累积
在最后正常化之前减少各块的统计数字。

示例:使用 QQZPROT000XZ,固件可以编译 QZPROT0001ZZ
(没有SerDes)或ZPROT000ZZ (SerDes 活动,向量分裂为 2)
每块2个元素).

### 1.2 系统块图

```
RISC-V CPU ──MMIO──► Decoder ──┬── MVU (matrix-vector multiply)
  (firmware)         (decode)   ├── MFU (GELU, add, mul)
                                ├── SLU (softmax, layernorm)
                                └── TMM (DRAM ↔ register files)
                                       ├── VMM (vector)
                                       └── MMM (matrix)
```

---

## 2. 指令集

指令是32位单词的两种格式:

### 2.1 SI格式(标准指令)

```
Bit:  31      24  23      16  15                       0
      ├──────────┼──────────┼──────────────────────────┤
      │  OpCode  │   Opd0   │          Opd1             │
      │  (8 bit) │  (8 bit) │        (16 bit)           │
      └──────────┴──────────┴──────────────────────────┘
```

OpCode 选择操作; QQZPROT000XZ 编码操作器( 记忆目标,
sub-opcode,即时).

### 2.2 LO格式(长期偏移)

```
Bit:  31      24  23                                   0
      ├──────────┼──────────────────────────────────────┤
      │  OpCode  │          Address (24 bit)             │
      │  (8 bit) │                                      │
      └──────────┴──────────────────────────────────────┘
```

用于DRAM内存操作(opcode ≥ 20). 提供24位字节地址.

### 2.3 数字表

|十二月(半天)|名称|格式|行动|
|-----|------|--------|-----------|
| 0 |津巴布韦|高级|Scalar 寄存器写道:|
| 1 |津巴布韦|高级|读取 Scalar 寄存器|
| 2 |津巴布韦|高级|VRF的矢量载荷: XQ ZPROT000XQZ|
| 3 |津巴布韦|高级|列缓冲器的矩阵负载 : XZPROT000XZ|
| 5 |津巴布韦|高级|向量商店到 VRF : XZPROT000XZ|
| 6 |津巴布韦|高级|矩阵写入确认|
| 7 |津巴布韦|高级|矩阵向量乘法: QZPROT000XZ( 无累积)|
| 8 |津巴布韦|高级|向量增加:ZPROT000Z|
| 11 |津巴布韦|高级|元素乘法: QZPROT000XZ|
| 20 |津巴布韦|联络处|来自 DRAM 的矢量载荷 : XZPROT000XZ|
| 21 |津巴布韦|联络处|向量存储到 DRAM : XZPROT000XZ|
| 22 |津巴布韦|联络处|带有自动递增地址的矢量负载|
| 23 |津巴布韦|联络处|具有自动递增地址的矢量商店|
| 24 |津巴布韦|联络处|矩阵瓦片负载: XQ ZPROT000XQZ|
| 25 |津巴布韦|联络处|矩阵瓷砖店: XQ ZPROT000XQZ|
| 27 |津巴布韦|高级|累积 MV MUL:从QQZPROT000XZ加载预置总和,增加结果|
| 42 |津巴布韦|高级|GELU通过LUT激活|
| 43 |津巴布韦|高级|矢量函数: opd0=0 → 软max, opd0=1 → 层态|
| 44 |津巴布韦|高级|添加 Scalar-scalar|
| 45 |津巴布韦|高级|链开始: 切换链 ID( 未在模拟器中执行)|

### 2.3a 格式检测

解码器通过opcode值区分SI和LO格式.
SI格式将24位操作字段分割为8位QQZPROT000XZ ID
(opd0)和一个16位的即时或子opcode(opd1). 够了
因为内部注册文件(VRF、MRF、scalar regs)的ID小于256
它们的指数是16比特

OpCodes ~ 20 访问 DRAM,需要更大的地址范围. 这些用法
LO 格式: 完整的 24 位操作字段是一个平面字节地址, 给予
16 MB 可地址 DRAM 空间 。 8位 opd0 和 16位 opd1 字段
与地址重叠——解码器调制它们:

```python
if opcode >= 20:
    # LO format: operand is the full lower 24 bits
    addr = inst & 0xFFFFFF
else:
    # SI format
    file_id = (inst >> 16) & 0xFF      # opd0: which VRF/MRF/reg file
    index   = inst & 0xFFFF            # opd1: offset within that file
```

### 2.4 VRF 记忆目标

|名称|数值|大小|目的|
|------|-------|------|---------|
|津巴布韦| 0 |2M+ 高频|外部 DRAM( 平面 24 位地址)|
|津巴布韦| 1 | 64 |临时乘数存储|
|津巴布韦| 4 | 128×128 |重量矩阵注册文件( MRF)|
|津巴布韦| 5 | 20480 |MVU 输入矢量RF|
|津巴布韦| 6 | 4096 |MFU 输入 / VRF 缓存|
|津巴布韦| 7 | 1024 |添加Sub 操作 A|
|津巴布韦| 8 | 4096 |添加Sub 操作 B|
|津巴布韦| 13 | — |MVM 叠加器|
|津巴布韦| 18 | — |矢量 — 矩阵行缓冲器|

### 2.5 斯卡尔登记册

Scalar 注册是内部的 NPU 状态,通过 QQZPROT000XZ 写入
透过ZPROT000Z读取。 他们控制数据传输维度 车道遮盖
和精确模式。 除非注明,否则所有登记册默认为0。

|名称|添加器|默认|写为|目的|
|------|------|---------|------------|---------|
|津巴布韦| 1 | 1 |固件|多向转移的瓦片行数(INC变体 iterate QXZPROT000XZ向量). 另外为QQZPROT0001XQZ设置MRF维度:负载XZPROT00002XQZ行.|
|津巴布韦| 2 | 1 |固件|多向转移的瓦片列数 。 与 QQZPROT000XXZ 合并计算 QZPROT00001Z 用于 INC 变种.|
|津巴布韦| 3 | 1 |固件|INC变种的外环计数. 控制 QQZPROT000Z 矢量的批量转移 。|
|津巴布韦| 15 |0xFF (英语).|固件|Per-lane读作XQZPROT000XZ的口罩. Bit QZPROT0001Z控制道 QZPROT00002Z. 默认 QZPROT0003Z 启用所有车道 。 用于多头注意只加载头特异性元素.|
|津巴布韦| 16 |0xFF (英语).|固件|Per-lane为QQZPROT000XQZ写口罩. Bit QZPROT0001Z控制道 QZPROT00002Z. 默认 QZPROT0003Z 启用所有车道 。 用于多头注意,将上下文写到标题特定元素槽.|
|津巴布韦| 17 |0xFF (英语).|固件|选中 XZPROT000XZ 的掩码。 Bit QZPROT0001XZ 口罩排队 QZPROT0002Z到ZPROT0003ZZ. 零行产生零点产品.|
|津巴布韦| 20 | 0 |固件|0=FP16,1=BFP(块浮点). 在预测阶段前设定为 1, 重置为 0 , 以引起注意 。|

** 软件使用示例**(来自 QQZPROT000XZ):

```c
// Configure for tiled matrix multiply (dim=2, hidden=4, 2×2 tiles):
SEND_SI(OP_S_WR, REG_TILE_ROWS, num_tiles);      // 2 tile rows
SEND_SI(OP_S_WR, REG_TILE_COLS, num_tiles);       // 2 tile columns
SEND_SI(OP_S_WR, REG_ITERATIONS, seq_len);        // iterate per position
SEND_SI(OP_S_WR, REG_READ_MATRIX_MASK, 0xFF);     // enable all MRF rows
SEND_SI(OP_S_WR, REG_PRECISION_MODE, 1);          // enable BFP

// Multi-head attention: mask per head (heads_per_tile=2):
for (int h = 0; h < heads_per_tile; h++) {
    uint8_t head_mask = (h == 0) ? 0x03 : 0x0C;   // elements [0,1] vs [2,3]
    SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, head_mask);
    // ... load Q/K/V slice for this head ...
    SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, head_mask);
    // ... compute and store context for this head ...
}
SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);     // restore full mask
SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, 0xFF);    // restore full mask
```

---

## 3. 命令链模型

**命令链**是一个由三个微操作组成的序列,构成一个
有效的 VLIW 指令 :

```
load  operand  → VRF / MRF      (V_RD_DRAM, M_RD_DRAM, V_RD, etc.)
compute       → pipeline        (MV_MUL, VV_ADD, V_FUNC, V_GELU, etc.)
store result  → VRF / DRAM      (V_WR, V_WR_DRAM, etc.)
```

每个NPU操作都遵循这种负载计算存储模式,即
按顺序写成的32位指令
(原始内容存档于2019-09-03). ZZPROT000 QZ. 链式中的三个指令针对不同的功能
在 Python 模拟器中执行
一次一个

共同链的例子:

|装入|计算|存储|目的|
|------|---------|-------|---------|
|津巴布韦|津巴布韦|津巴布韦|矩阵向量乘法|
|津巴布韦|津巴布韦|津巴布韦|GELU 启动|
|津巴布韦|津巴布韦|津巴布韦|设置软max|
|ZPROT0001Z + ZPROT0001Z + ZPROT0001Z + ZPROT00001Z|ZPROT0001Z + ZPROT0001Z + ZPROT0001Z + ZPROT00001Z|ZPROT0001Z + ZPROT0001Z + ZPROT0001Z + ZPROT00001Z|倾斜的有偏倚的垫子|
|津巴布韦|XQ(ZPROT00000XQZ) (读音vpipe a)|津巴布韦|剩余添加|

### 3.1 执行令

Python 模拟器同步执行每个指令. 每一个字
以 QQZPROT000XZ 解码并运行指令,以便在
下一个开始 :

```python
def store(self, addr, data):
    if addr == NPU_INST_FIFO:
        self._status = STATUS_BUSY
        self._execute(opcode, opd0, opd1)   # runs to completion
        self._status = STATUS_DONE
```

没有指令队列,没有记分牌,没有危险检测,以及
没有平行发送。 训导一次处理一次.

### 3.2 状况

|登记册|折换|数值|含义|
|----------|--------|-------|---------|
|津巴布韦|0x04 (英语).| 0 |IDLE - 准备接受下一个指示|
| | | 1 |BUSY——飞行中的指示|
| | | 2 |完成- 指令完成|

在撰写下一个指令之前,

### 3.3 海峡间重叠和海峡内天线管道

链式模型可以实现两种不同的平行形式,这两种形式都是:
愿望( 在 Python 模拟器中未执行) :

#### 海峡间重叠(SMC-同时多海峡)

没有 QQZPROT000Z 危险的独立链可以同时运行。
每个链都针对不同的功能单元(TMM、ZPROT000XZ、TMM),
和每个单位繁忙状态的QQZPROT000XQZ记录器(0x0C)音轨.

例如:当链1计算QQZPROT000XZ(MVU繁忙)时,链2可以加载
进入 MRF (TMM 忙碌) 的下一个瓦片 :

```
Time ────────────────────────────────────────────────────>
     ┌─────────────────────────────────────┐
C1   │ M_RD_DRAM(K.T) │ MV_MUL  │ V_WR    │  ← chain 1: Q × K.T
     └───────┬─────────┴─────────┴─────────┘
             │ overlap: TMM loads next tile
             │ while MVU computes
     ┌───────┴──────────────────────────────┐
C2   │ M_RD_DRAM(V) │ MV_MUL  │ V_WR      │  ← chain 2: attn × V
     └──────────────┴─────────┴────────────┘
```

#### 中天天线内管道

在单一链条内,管道记录线程在
指令,没有明确的 QZPROT000XZ 字段。 这样就可以
管道数据流动,其中一个微型操作器的输出被消耗
,避免中间
VRF 检查 :

```
Cycle:    1           2           3           4           5
         V_RD_DRAM → MV_MUL → V_FUNC(SOFTMAX) → MV_MUL → V_WR
         (TMM load)  (MVU)     (SLU)            (MVU)    (TMM store)
         │           │         │                │        │
pipe:    Q           scores    attn             context  context
```

每个阶段都读到前一阶段所写的管道值,
形成** tensor 管道**,其中矢量流经多个
计算单元而不离开数据路径。 在硬件中
执行中,管道记录器为单一矢量长
注册(NATION DIM × FP16),所有单位都可以在
同样的循环。

#### 危害

|危险|条件|检测|
|--------|-----------|-----------|
|拉脱维亚|装入 MRF ~ MV  MUL 在同一链中|订购:MRF由M RD DRAM设定,由MV MUL消耗|
|拉脱维亚|V RD DRAM — MV MUL 在同一链条|订购:由 V RD DRAM 设置, MV MUL 消耗的管道|
|战争|两条链条写着相同的 VRF|链排程器: 在发行前检查 VRF 目标|
|妇联|两条链条写着相同的 VRF|链排程器: 在发行前检查 VRF 目标|

#### 与现有实例的关系

在ZPROT000XQZ中,
- ZPROT000XZ:一个链中的基本负载计算存储器
- QQZPROT000XQZ:两条独立链(MVM+偏差加)——候选人.
  链间重叠
- 通过MVU SLU MVU的ZPROT000+Z:XQK.T 软马克+Attn×V管道
  - 链内拉伸管线的犬类例子

---

## 4. 计算单位

### 4.1 MVU——矩阵-变量乘法

计算MRF行和输入矢量之间的点产品.
通过ctypes将数字计算委托给 QQZPROT000XQZ.

- QQZPROT000XXZ: 覆盖内部积分( 积=0)
- ZPROT000XZ:从ZPROT00001Z装入先前的总和,添加,存储后退

### 4.2 MFU——多功能股

3个具有基于能力的路由:

|缩略语|业务|
|-----|------------|
| 0 |GELU(1024-入口LUT)|
| 1 |向量XZPROT000ZZ|
| 2 |Tanh, QZPROT000Z, 乘数|

Python 模拟器将指令引导到相应的 C 库
函数(XZPROT000XZ,XZPROT00001Z,XZPROT00002Z,XZPROT00003XZ)通过类型.

### 4.3 SLU - Softmax / Layer Norm (软)

单ZPROT000XXZ 发送触发 引信减少和正常化:

** 软件:** 最大减少量 → exp(x- 最大) → 总分 → 正常
** LayerNorm:**总和 = 平均值 = 差异 = inv sqrt = 比例+班次

当 QQZPROT000XQZ 时, SLU 以 SerDes 模式运行, 重复
QQZPROT000XXZ 元素块和最终数据
正常化通过。

---

## 5. 内存系统

### 5.1 记录

平面24位可地址存储器(512K浮点32元素). 固件装填
在发布计算指令之前输入载荷和重量矩阵。

### 5.2 矢量登记文件

|弗朗索瓦|身份证|要件|已连接到|
|-----|----|----------|-------------|
|津巴布韦| 5 | 20480 |MVU 输入|
|津巴布韦| 6 | 4096 |MFU 输入 / VRF 缓存|
|津巴布韦| 7 | 1024 |MFU AddSub 操作 A|
|津巴布韦| 8 | 4096 |MFU AddSub 操作 B|
|津巴布韦| 13 | 256 |MVM 蓄积器(预装)|

QQZPROT000XZ 指令变体维护内部 DRAM 地址登记册
每次转移后自动增殖,使流畅无明
地址更新。

### 5.3 矩阵登记文件(MRF)

单张ZPROT000Z牌. 从DRAM通过QQZPROT0001XQZ装入.
每ZPROT000Z周期提供一行用品。 装入新瓷砖覆盖
前一个。

---

## 6. TMM — 十进制内存管理器

两个独立的分单位处理 DRAM QQ 注册文件传输:

### 6.1 VMM (变量内存管理器)

Handles XQ ZPROT000XQ Z, XQ ZPROT0001XQ Z, 及其INC变种. 密克罗尼西亚联邦
由QQZPROT000XXXZPROT00001Z控制多向量传输支持.

### 6.2 MMM( Matrix 内存管理器)

(原始内容存档于2019-09-29). Handles QZPROT0001Z和ZPROT0001Z. 按元素排列元素
横跨矩阵牌。

---

## 7. 精确度

- **FP16**: IEEE 754-2008 二进制16 在所有计算单位边界上
- ** BFP** (可选): 块浮点, 以 QQZPROT00000 QZ 组共享参数
- ** BFP 宽度**: 4位曼提萨(种子), 4位曼提萨(种子)

---

## 8. 固件接口

### 8.1 MMIO 登记册

|折换|名称|访问|目的|
|--------|------|--------|---------|
|0x00 (英语).|津巴布韦|维文|写入指令单词|
|0x04 (英语).|津巴布韦|R级|0=IDLE、1=BUSY、2=DONE|
|0x08 (英语).|津巴布韦|维文|写入非零 = 重置|
|0x0C (英语).|津巴布韦|R级|单位繁忙位数|
|0x20 (英语).|津巴布韦|津巴布韦|隐藏尺寸配置|
|0x24 (英语).|津巴布韦|津巴布韦|序列长度配置|

### 8.2 固件发送协议

```c
// Push one instruction, wait for completion:
npu_send_inst(instruction_word);
while (npu_read_reg(NPU_STATUS) != NPU_STATUS_DONE);
```
