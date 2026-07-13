> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU 链例

显示使用单链和多链固件模式
连锁-意识发送API(XQZPROT000XZ + XZPROT0001Z).

## 背景情况

一个**链**是一组连续的NPU指令被承诺
解剖学上通过QQZPROT000XQZ(opcode 45). 一连串:

- 指令流入FIFO**不按指示摊位**
  (ZPROT000Z不再投票)
- 介于指令之间的线程值:
  ZPROT000ZZ将一个矢量装入管道, 计算操作转换它,
  ZPROT000Z 把它储存到ZPROT00001Z ** 将数值保留在
  管**(广播语义).
- ZPROT000Z/ZPROT0001Z**不**消耗管道——随后
  指令仍然可以通过 QQZPROT000XXZ 读取(它保存了旧的)
  输入ZPROT000Z的二进制操作
- XZPROT000XZ 装入新值并保存上一条管道
  ZPROT0001XQZ(ZPROT0001XQZ,ZPROT00002XQZ等的首个操作).
- 津巴布韦 ** 丢弃管道**——数值没有持续
  跨越链条边界。

## 隐性诉明确行为

NPU指令以两种方式参考操作:

|作业|来源|用于|
|---------|--------|--------|
|** 管理成果框架**(明确)|由 QQZPROT000XZ 或 ZZPROT0001Z 装入(列缓冲)|仅限ZPROT000Z|
|** 管道** (隐含)|由最近 的 ZPROT000XZ / ZPROT0001Z 设定|ZPROT0001Z,ZPROT00001Z,ZPROT00002Z 激活,ZPROT00003Z,ZPROT00004Z|
|**vippe a**(隐形)|(旧管道)|ZPROT000Z,ZPROT0001Z等 (二进制行动)|
|** 特别报告**(明确)|由 QQZPROT000XZ, QZPROT0001XZ 等编写 。|津巴布韦|

ZPROT000XXZ是独一无二的:它需要**一个明确的操作**(MRF, 由
和**一个隐含的操作**(管道,由
在ZPROT000Z/ZPROT00001Z之前。 指令词本身编码
既非源——是链条中指令的 * 排序 *
保证管理成果框架和管道保持正确的价值。

因此,QQZPROT000XXZ中的单链例可以避免
不必要的 QQZPROT000Z 环行图 — 矢量生命
从ZPROT00000Z直接进入ZPROT00001Z的管道中.

横跨链条:

- `INST_ISSUE` 执行所有前述并行指令
  (以RTL计) 通过清除模拟器模式
  (原始内容存档于2019-09-02). ZPROT000XZ to 0.
- (ZPROT0001Z 民调)
  直到所有功能单元(VMM,MMM,MVU)闲置.
- ** SMC**(同时期的多笼):独立的链条
  (没有 QZPROT000XZ 危险) 可以在RTL 中同时运行 。

## CHAIN STATUS 位任务

|位数|单位|说明|
|-----|------|-------------|
| 0 |自愿|ZPROT0001Z,ZPROT0001Z,ZPROT00002Z,ZPROT00003Z,ZPROT00004Z,ZPROT00005Z(活动),ZPROT00006Z.|
| 1 |母亲|ZPROT000XZ,ZPROT00001Z (VecToMatRow → MRF),ZPROT00002XZ|
| 2 |毛里求斯|津巴布韦|

输油管登记册本身不是一个功能单位——它是一个数据
所有指示都经过的路径。 永远不会独立
"繁忙",因此在CHAIN STATUS中没有跟踪.

## 文件引用

### 津巴布韦

一个链组中的单个MVM(矩阵-活度乘数):

```
M_RD_DRAM → M_WR → V_RD_DRAM → V_WR/IVRF → V_RD/IVRF → MV_MUL → V_WR/MPV
├───────────────────── MMM ─────────────────────┤
├───────── VMM (load) ─────────┤├── MVU ──┤├─ VMM (store) ─┤
                                                      INST_ISSUE
                                                      wait_chain
```

然后第二连锁阅读VRF结果并写到DRAM.

### 津巴布韦

** 1** — MVM: ZPROT000Z

** 第2章**——比亚斯加上:ZPROT000XZ

并包含ZPROT000Z, 参考执行
根据ZPROT000XZ的ZPROT00001Z,
作为单链组(15个指令).

### 津巴布韦

证明通过ZPROT0001Z 与ZPROT00001Z 连在一起
隐含管道。 软max 输出直接流入 QQZPROT000 QZ
没有中间 DRAM 保存。

** Chain 1 ** — Q × K.T → 分数 → 软max(MRF = K.T,管 = Q):
```
M_RD_DRAM(K.T) → M_WR → V_RD_DRAM(Q) → MV_MUL → V_FUNC(SOFTMAX) → V_WR
├──── MMM ─────┤├─────────── VMM (load) ──────────┤├ MVU ─┤├ V_FUNC ┤├ VMM ┤
                                                                  INST_ISSUE
                                                                  wait_chain
```

** Chain 2 ** — atn × V → 上下文(MRF = V, 管道 = atn from VRF):
```
M_RD_DRAM(V) → M_WR → V_RD(attn) → MV_MUL → V_WR
├─── MMM ────┤├── VMM ──┤├ MVU ┤├ VMM ┤
```

** Chain 3 ** — 背景书写给 DRAM。

关键洞察力:ZPROT000XQZ读取管道(MV MUL的分数),
应用软max,并将结果写回管道上——所以
下一个指令 (V WR) 立即看到注意重。 没有
DRAM在积分和软马克之间循环保存载荷.

管理成果框架将**K.T**列入第1链,将**V**(不是V.T)列入第2链。
由于MV MUL计算 MRF × 管道,链2计算 V× attn,
上下文矢量 = 值行的加权和。

### 津巴布韦

生成用于DAG的显示伸缩流经链的图
实例。 生产两个事件级的 DAG( 每个指令都作为节点)
以及一个崩溃的微操作DAG(负载计算存储组).

运行脚本重播单链和SiLU链指令
通过模拟器的 QQZPROT000XZ 序列并写入 DOT 文件
用 Graphviz 渲染:

```bash
PYTHONPATH=. python3 firmware/examples/chain_dag.py --output /tmp/chain_dag/
# Render DOT to PNG:
dot -Tpng /tmp/chain_dag/chain_example_events.dot -o chain_events.png
dot -Tpng /tmp/chain_dag/chain_example_microops.dot -o chain_microops.png
dot -Tpng /tmp/chain_dag/silu_chain_microops.dot -o silu_chain_microops.png
dot -Tpng /tmp/chain_dag/silu_chain_events.dot -o silu_chain_events.png
dot -Tpng /tmp/chain_dag/multi_chain_events.dot -o multi_chain_events.png
dot -Tpng /tmp/chain_dag/multi_chain_microops.dot -o multi_chain_microops.png
dot -Tpng /tmp/chain_dag/softmax_chain_events.dot -o softmax_chain_events.png
dot -Tpng /tmp/chain_dag/softmax_chain_microops.dot -o softmax_chain_microops.png
```

### 生成的 DAG 图表

预发图见于 QQZPROT00000Z:

|图表|点文件|说明|
|---------|----------|-------------|
|[单链事件] (ZPROT000XZ)|津巴布韦|事件级别 DAG QQZPROT000XZ – 每个指令作为节点,边缘显示数据流(MRF → MV MUL,管道 → MV MUL,管道 → V WR)|
|(ZPROT000XXZ) (ZPROT000Z) (ZPROT000XZ) (ZPROT000Z) (ZPROT000Z) (ZPROT000Z) (ZPROT000Z) (ZPROT000Z) (ZPROT000Z) (ZPROT000Z) (ZPROT000Z) (ZPROT000Z) (ZPROTUTUZ) (ZPROTUTUTOZ) (ZPROTUTUZ) (ZPROTMZ) (ZPROTMZ)|津巴布韦|折叠的微op DAG – MAT LOAD → MV MUL → DRAM STORE 组|
|[多链事件] (ZPROT000XZ)|津巴布韦|双链序列来自 QQZPROT000XZ – 链 1 (MVM) 之后是链 2 (bias add)|
|(ZPROT000XXZ) (ZPROT000Z) (英语).|津巴布韦|跨越两个链的微op DAG – VRF[1][0] 数据依赖连接 INST ISSUE 边界|
|[silu连锁事件] (ZPROT000XZ)|津巴布韦|SiLU 事件级 DAG 上 → W  down → 残余链 – 24 个带有全脱功能边缘的指令|
|(ZPROT000XZ) (英语).|津巴布韦|同一链条的折叠微操作DAG——21个指令崩溃到9个微操作器|
|! [软链事件] (ZPROT000XZ)|津巴布韦|事件级别 DAG QQZPROT000XZ – 链1:QQK.T → 分数 → 软马克斯 → attn. 链条2:atn × V 上下文. 链条3: DRAM写.|
|(ZPROT000XXZ) (英语).|津巴布韦|Micro-op DAG——SOFTMAX是MV MUL(K.T×Q)和MV MUL(V×attn)之间的一个独立的节点. MRF持有K.T然后V.|

事件级别 DAG 显示 MV MUL 带有两个进入的边缘—— 一个 QQZPROT000XZ
(前面的"ZPROT000XZ"中的明确操作)和"ZPROT00001Z"中的1个.
(前作"ZPROT000XZ"中的隐含操作). 每个边缘
标签为资源名称,使数据流明确。

示例输出( 活动级别) :
```
  3 M_RD_DRAM          defs=[(MRF,)]          uses=[(DRAM, 1024)]
  4 M_WR               defs=[]                uses=[(MRF,)]
  5 V_RD_DRAM          defs=[(pipe,), ...]     uses=[(DRAM, 8192)]
  6 MV_MUL             defs=[(pipe,)]          uses=[(pipe,), (MRF,)]
    <- [3 M_RD_DRAM] via MRF       ← explicit
    <- [5 V_RD_DRAM] via pipe      ← implicit
  7 V_WR               defs=[(VRF, 1, 0)]      uses=[(pipe,)]
  8 INST_ISSUE
  9 V_RD               defs=[(pipe,), ...]      uses=[(VRF, 1, 0)]
 10 V_WR_DRAM          defs=[(DRAM, 8448)]     uses=[(pipe,)]
 11 INST_ISSUE
```

## 构建

这些例子使用与现有的Makefile基础设施相同的
坚固的软件。 以 :

```bash
cd firmware
make TARGET=examples/01_single_chain BUILD_DIR=build_examples
make TARGET=examples/02_multi_chain BUILD_DIR=build_examples
make TARGET=examples/03_softmax_chain BUILD_DIR=build_examples
```

## 密钥 API

|函数|目的|
|----------|---------|
|津巴布韦|按一个指令——没有FIFO摊位|
|津巴布韦|发送 QZPROT000QZ 以执行当前链|
|津巴布韦|在所有单位闲置之前,|
|津巴布韦|遗产——用链式API代替|
