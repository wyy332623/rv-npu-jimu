# RV-NPU/JIMU 项目版本与上游仓库详细比较报告

> **归档状态**：这是一次定点仓库比较，结论绑定当时的上游提交和本地工作树。它保留为迁移依据，不作为当前运行说明；当前状态请看 [`../project-status.zh.md`](../project-status.zh.md)。

> 核查日期：2026-08-05
> 当前工作区：`<workspace>/rv-npu-jimu.v10`
> 上游仓库：`git@gitee.com:RSPwFPGAs/rv-npu.git`

## 1. 核查范围与结论摘要

本报告比较以下四类版本：

1. Gitee 上游 `master`：当前产品主线，提交 `cc05ffc7098a946d6368db1f14901498b702bccd`。
2. Gitee 旧闭环实验分支：重点比较最新的 `origin/explore/closed-loop-fw-optimization-x86`，提交 `08b66c2`。
3. 本地 `rv-npu-jimu.v11`：无独立 Git 历史的开发快照。
4. 本地 `rv-npu-jimu.v10`：当前持续开发的 JIMU 闭环工作区，包括大量尚未整理提交的改动。

主要结论如下：

- 当前 Gitee `master` 已经不是旧闭环分支的简单后续版本，而是一次面向 Tile/DSL、并发执行单元、硬件 MHA、DMA 和 Portal IR 的架构升级。
- 旧闭环分支的演进关系是：

  ```text
  explore/closed-loop-fw-optimization
      -> explore/closed-loop-fw-optimization-adderboard-x86
          -> explore/closed-loop-fw-optimization-x86
  ```

  其中 `-x86` 是旧闭环体系最完整的上游历史版本，但它不是未来新分支的理想基线。
- 当前 v10 在旧闭环基础上补齐了可选择验证维度、初始候选选择、独立验证门、指标门、结构化 DAG、DAG 差异门、Skill 版本管理和 Agent Skill 注入等能力，功能明显超过 v11 与上游旧闭环分支。
- 当前 v10 的 `firmware/bert/bert_layer.c` 已恢复为正确、未优化的基准版本；它与 v11 固件、v11 基线、v10 基线以及上游旧闭环分支中的基线逐字节一致。
- 该旧基准固件不能直接覆盖到当前 Gitee `master`。主线固件已增加硬件 MHA、MHA-v2、DMA、链式启动、在线 Softmax 等新路径，两者属于不同架构阶段。
- 当前 v10 工作树严重不适合直接整体提交：存在大量历史结果、构建产物、删除记录、未跟踪文件和混合功能改动。后续应从干净的上游 `master` 新建分支，按功能分批移植，而不是合并整个 v10 工作树。
- v10 的核心闭环测试当前为 `13 passed`；平台辅助测试为 `44 passed, 1 failed`，唯一失败是 Skill 测试仍断言 `2.1.0`，而实际 Skill 已升级为 `2.2.0`。这是提交前应修复的版本同步缺陷。
- 上游 `master` 的完整测试在当前环境中因缺少 `amaranth` 而在收集阶段中止，不能据此认定上游代码失败。

## 2. 仓库与版本状态

### 2.1 上游克隆状态

认证后的完整克隆位于：

```text
<workspace>/rv-npu.gitee-repo
```

状态：

- 远端：`git@gitee.com:RSPwFPGAs/rv-npu.git`
- 分支：`master`
- HEAD：`cc05ffc7098a946d6368db1f14901498b702bccd`
- 提交时间：2026-08-05 07:56 +0800
- 工作树：干净
- `git pull --ff-only`：Already up to date
- `git fsck --no-dangling`：通过

失败的克隆目录 `<workspace>/rv-npu.gitee` 仅包含一个无 HEAD、无提交的 `.git` 元数据目录，已删除。真实克隆 `rv-npu.gitee-repo` 和 ZIP 快照均未受影响。

### 2.2 版本总览

| 版本 | 形式 | 主要定位 | 适合作为新分支基线 |
|---|---|---|---|
| Gitee `master` | 正式 Git 主线 | 新 Tile/DSL/硬件执行架构 | 是 |
| `explore/closed-loop-fw-optimization-x86` | 上游实验分支 | 旧架构下最完整的固件 Agent 闭环 | 仅作迁移参考 |
| 本地 v11 | 文件系统快照 | 较早的正确性基线与闭环原型 | 否 |
| 本地 v10 | 当前开发工作区 | 最新 JIMU、结构化 DAG、门禁和 Skill 管理 | 作为功能来源，不作为分支基线 |
| ZIP `rv-npu-master` | 下载快照 | 用于交叉核对主线文件 | 否 |

当前 v10 的根提交 `c08cda6b...` 不存在于 Gitee 上游对象库中，因此 v10 和当前上游不能依靠共同祖先做常规合并。比较时必须区分：

- Git 历史比较：用于上游分支之间；
- 文件树比较：用于 v10/v11 和上游之间；
- 功能语义比较：用于判断哪些能力可以移植。

## 3. 目录结构比较

以下文件数量用于描述规模，不等于代码行数。v10/v11 的统计排除了 `.git`、虚拟环境、缓存、构建输出和运行结果；若是否排除备份文件或二进制文件的口径不同，总数会有少量变化。

### 3.1 Gitee `master`

主线共约 239 个受跟踪文件，主要分布：

| 一级目录 | 文件数 | 主要内容 |
|---|---:|---|
| `hdl/` | 63 | MHA、MHA-v2、DMA、MVU、MFU、存储、互联和顶层 RTL |
| `docs/` | 54 | 架构、历史、性能、MHA/MVU/NoC 演进文档 |
| `tests/` | 32 | 集成、NoC、MVU、MMM、DSL/Portal IR 测试 |
| `dsl/` | 27 | 前端 IR、Lowering、后端绑定与 Tile 相关 IR |
| `tools/` | 19 | 性能、trace、检查与开发工具 |
| `emulator/` | 17 | 新 ISA/执行单元的软件模型 |
| `firmware/` | 11 | 新架构 BERT 固件及运行时支持 |
| `sim/` | 4 | 仿真入口和支持代码 |

结构重点：

```text
rv-npu (master)
├── dsl/
│   ├── backend/
│   ├── lowers/
│   ├── portal_ir/
│   ├── tilelang_ir/
│   └── tle_ir/
├── firmware/
│   ├── bert/
│   └── lib/
├── emulator/
│   └── kernels/
├── hdl/
│   ├── dma/
│   ├── mfu/
│   ├── mha/
│   │   ├── tests/
│   │   └── v2/
│   ├── mvu/
│   ├── intrf/
│   ├── mem/
│   └── top/
├── tests/
│   ├── integration/
│   ├── dispatcher_noc/
│   ├── mmm/
│   └── mvu/
├── tools/
├── sim/
└── docs/
```

特征是“硬件、DSL、固件、仿真、测试”一体化。主线没有 `jimu-dse/`，说明当前 JIMU 闭环尚未并入产品主线。

### 3.2 上游旧闭环 `explore/...-x86`

该分支约 133 个受跟踪文件：

| 一级目录 | 文件数 | 主要内容 |
|---|---:|---|
| `adderboard/` | 44 | 旧训练、布局、黄金模型和验证工具 |
| `firmware/` | 30 | 旧 ISA 固件、BERT 层和示例 |
| `jimu-dse/` | 16 | 早期闭环脚本、目标、基线和 Skill 文档 |
| `emulator/` | 10 | 旧 NPU 模拟器与事件跟踪 |
| `kernels/` | 9 | 核函数与生成支持 |
| `tests/` | 5 | 旧固件集成测试 |
| `.opencode/` | 3 | 三个早期 Agent Skill |

```text
rv-npu (closed-loop-x86)
├── .opencode/skills/
│   ├── dag-analyze/
│   ├── dim-optimize/
│   └── vrf-cache/
├── jimu-dse/
│   ├── baseline/
│   ├── docs/skills/
│   ├── goals/
│   └── scripts/
├── emulator/
├── firmware/
│   ├── bert/
│   ├── examples/
│   └── lib/
├── adderboard/
├── kernels/
└── tests/integration/
```

该分支在其历史中主动移除了 HDL/RTL，代表的是“功能模拟器 + 固件闭环”阶段，不包含当前主线的完整硬件体系。

### 3.3 当前 v10

过滤结果与构建产物后，当前 v10 约有 200 个源码和文档文件，主要分布：

| 一级目录 | 约文件数 | 相比旧闭环新增重点 |
|---|---:|---|
| `jimu-dse/` | 64 | 独立门禁、结构化 DAG、Skill 锁定、工作负载、详细文档 |
| `firmware/` | 34 | 正确性基线、旧固件库与示例 |
| `adderboard/` | 33 | 旧功能模型与验证辅助 |
| `.opencode/` | 16 | 从规范 Skill 同步生成的 Agent Skill |
| `tests/` | 15 | DAG、门禁、Skill、闭环快照和端到端测试 |
| `emulator/` | 11 | 事件 DAG、微操作 DAG、新增结构化 DAG |
| `docs/` | 10 | 项目和优化说明 |
| `kernels/` | 9 | 旧核函数支持 |

```text
rv-npu-jimu.v10
├── jimu-dse/
│   ├── baseline/
│   ├── docs/
│   │   └── skills/
│   │       ├── isa/
│   │       └── versions/
│   ├── goals/
│   ├── scripts/
│   │   ├── npu_closed_loop.sh
│   │   ├── validation_gate.py
│   │   ├── metric_gate.py
│   │   ├── dag_diff_gate.py
│   │   └── skillctl.py
│   └── workloads/
├── .opencode/skills/
│   ├── common-constraints/
│   ├── dag-analyze/
│   ├── dim-optimize/
│   ├── inc-folding/
│   ├── self-verify/
│   └── vrf-cache/
├── emulator/
│   ├── npu_event_trace.py
│   ├── npu_micro_op_dag.py
│   └── npu_dag_structured.py
├── firmware/
├── tests/
├── adderboard/
└── kernels/
```

与上游旧闭环相比，v10 最大的结构变化是把闭环中的复杂判断拆成可单测的 Python 工具，并把 Skill 从零散提示词提升为带规范源、版本、哈希和同步机制的受控输入。

### 3.4 v11

v11 过滤后约 111 个源码文件，结构基本对应较早的闭环原型：

- `firmware/` 约 30 个文件；
- `adderboard/` 约 25 个文件；
- `jimu-dse/` 约 14 个文件；
- `emulator/` 约 10 个文件；
- `tests/` 仅保留少量集成测试。

v11 和当前 v10 有约 111 个共同源码路径，其中大部分未改变。v10 在 v11 之上增加了约 90 个与平台化、门禁、Skill、DAG 结构化和测试有关的文件。v11 更适合用于确认旧固件正确性来源，不适合作为继续开发的平台基线。

## 4. 关键功能比较

| 能力 | Gitee master | 旧闭环 x86 | v11 | 当前 v10 |
|---|---|---|---|---|
| BERT 固件 E2E | 是，多后端/多执行路径 | 是，旧 SW 路径 | 是 | 是 |
| dim2/dim4 测试 | 覆盖但体系已扩展 | 固定测试矩阵 | 基本测试 | 可由闭环参数选择 `2/4/all` |
| 固件 Agent 闭环 | 无 | 有 | 有，较早 | 有，功能最完整 |
| OpenCode | 无 | 固定 Skill | 固定 Skill | 显式传入全部有效 Skill |
| Pi Agent | 无 | 只传部分 Skill | 早期组合 | 将全部 Skill 合成单一输入包 |
| 初始候选选择 | 无 | 主要接续 `candidate_best` | 有限 | 基线、源码文件、运行目录均可选 |
| 正确性金标准 | 主线自身模型 | 旧固件基线 | 旧正确基线 | 金标准固定，优化起点可独立选择 |
| 指令回归门 | 主线性能体系 | Shell 内嵌 | 简单 | 独立 `metric_gate.py`，可开关/设阈值 |
| 结构化 DAG | 主线有 trace 方向但无 JIMU 工具 | 文本/图片 DAG | 文本 DAG | JSON/文本摘要/候选 ID/宏候选 |
| DAG 差异门 | 无 | 无独立工具 | 无 | 独立 `dag_diff_gate.py` |
| Skill 版本管理 | 无 | 无 | 无 | lock、版本快照、SHA256、自动同步 |
| 公共安全约束 | 无 Agent | 零散 | 零散 | 禁止 stash/reset/checkout、修改测试和模拟器 |
| HDL | 完整 | 已移除 | 无 | 无 |
| DSL/Portal IR | 完整 | 无 | 无 | 无 |
| MHA/MHA-v2/DMA | 完整 | 无 | 无 | 无 |

## 5. `bert_layer.c` 的版本关系

### 5.1 哈希与规模

| 来源 | SHA256 | 行数 | 含义 |
|---|---|---:|---|
| Gitee `master` 当前固件 | `bbbf9247d8d16f7796bf78231174926e93063d4f80c31d57ec63a4fd73ea19e9` | 1286 | 新架构、多执行路径固件 |
| 旧闭环 x86 当前固件 | `7993029093c6234900b152f170e6be74b40202c47536bb8cfce5d271a73f7c6b` | 514 | 旧闭环优化结果 |
| 旧闭环 x86 基线 | `0957b9cb6a5dff4cf9f1b3dbf55ca882d94a428debde3d9da1aa4a7a2dacb664` | 580 | 旧架构未优化正确基线 |
| v11 固件及基线 | `0957b9...b664` | 580 | 与旧闭环基线一致 |
| v10 当前固件及基线 | `0957b9...b664` | 580 | 与旧闭环基线一致 |
| v10 Git HEAD 中的旧固件 | `598f442bbfd6d9d26b95e56c56a98614b64bef6650df5250cedad7192e1a0345` | 490 | 之前提交过的优化形式 |

因此，当前 v10 的事实状态是：

```text
金标准源码 = v10 baseline = v10 当前 firmware
             = v11 baseline = v11 firmware
             = 上游旧闭环分支 baseline
```

这满足“每次优化以正确的未优化版本作为金标准，同时允许从其他候选继续搜索”的设计要求。继续优化时，正确性比较对象不应随 `--start-from` 改变。

### 5.2 为什么不能把该基线直接提交覆盖 master

Gitee `master` 的 1286 行固件已包含旧基线没有的关键能力，例如：

- chain start/wait 与新的调度方式；
- MVM VRF 路径和 MRF 预加载；
- K/V 转置与 head-outer attention；
- online softmax；
- 硬件 MHA 与 MHA-v2；
- 多种 memory-centric K/V 路径；
- interleaved SMC 等新实现。

所以旧基线只能继续用于验证旧 SW 后端和迁移 JIMU 方法，不能作为当前主线的固件替换件。面向主线的优化目标应逐步从“让 Agent 直接重写整份 C 文件”提升为“让 Agent 在主线已有 lowering/binding 规则中选择、组合和调参”。

## 6. Agent 闭环与门禁比较

### 6.1 旧闭环 x86

旧脚本约 898 行，主要能力包括：

- `dram-optimization`、`compute-optimization`、`combined-optimization` 三类目标；
- `pi` 与 `opencode` 两种 Agent；
- 从 `candidate_best.c` 恢复；
- 使用 seq1 生成可视化 DAG；
- 采集 p2/p6 指标；
- dim 改变检查、指令数与 `MV_MUL` 回归检查；
- VRF 缓存、维度优化和 DAG 分析 Skill。

主要局限：

- 大量逻辑堆在 Shell 中，难以独立测试；
- OpenCode/Pi 获得的 Skill 集合不一致；
- 无 Skill 版本、哈希和可回滚机制；
- DAG 以人读文本/图片为主，Agent 很难稳定引用具体机会；
- seq1 DAG 无法表达跨 token 复用；
- 运行目录和不同 dim/seq 的图文件可能相互覆盖。

### 6.2 当前 v10

当前 `npu_closed_loop.sh` 约 1330 行，新增的主要控制接口包括：

- `JIMU_MAX_ITER`：最大优化轮数；
- `JIMU_THRESHOLD`：达到目标后提前停止的阈值；
- `JIMU_VALIDATION_DIM=2|4|all` / `--validation-dim`：选择闭环验证维度；
- `JIMU_INSTR_GATE=on|off`：显式控制指令回归门；
- `JIMU_INSTR_REGRESSION_LIMIT`：门开启时允许的指令数相对回退比例；
- `JIMU_DAG_EVIDENCE_GATE`：控制 DAG 证据门；
- `JIMU_AGENT_TIMEOUT`、`JIMU_AGENT_RETRIES`：Agent 超时和重试；
- `--start-from baseline|<source.c>|<run-dir>`：选择优化起点。

闭环现在明确区分：

```text
正确性金标准：固定的 canonical baseline
优化搜索起点：baseline、已有源码或历史 candidate_best
最终接受条件：正确性 + 指标门 + DAG 证据门（按配置）
```

独立工具的职责：

| 工具 | 职责 |
|---|---|
| `validation_gate.py` | 统一执行与解析正确性验证 |
| `metric_gate.py` | 比较指令/DRAM 等指标，处理允许回退比例 |
| `dag_diff_gate.py` | 比较基线和候选 DAG，验证优化证据是否成立 |
| `skillctl.py` | 校验、锁定、同步和组合 Skill |
| `npu_dag_structured.py` | 将 trace/DAG 转为结构化候选和摘要 |

## 7. DAG 分析能力与主线适配问题

### 7.1 当前 v10 如何分析 DAG

当前流程可以概括为：

```text
固件执行 trace
  -> 指令/微操作事件
  -> 依赖图与生命周期
  -> 识别候选模式
  -> 输出结构化 JSON、可读摘要和稳定候选 ID
  -> Agent 按候选 ID 修改源码
  -> 候选 DAG 与基线 DAG 做差异门验证
```

结构化分析目前重点寻找：

- DRAM 重复读取及可能的 VRF 缓存机会；
- 中间结果反复写回/读回；
- 可折叠或可融合的连续操作；
- 值生命周期和 VRF 容量冲突；
- 宏级候选及预估收益；
- 优化前后关键节点、边、流量和指令变化。

这比直接把数千行 DAG 文本交给 Agent 更可靠，因为 Agent 能引用稳定 ID、明确修改目标和预期指标。

### 7.2 seq1 的局限

seq 表示一次 BERT 层输入中的 token/位置数量。例如：

- `seq1` 只有一个 token，图小、生成快，适合观察单 token 内部数据流；
- `seq2` 开始出现 token 之间的 K/V、权重或中间数据复用；
- `seq6` 更接近当前回归测试的长序列路径，也更能暴露重复 DRAM 流量和容量压力。

当前 v10 已修复不同 dim/seq 图构建目录相互覆盖的问题，但优化候选的主结构化 DAG 仍偏重 seq1，而正确性和性能通常在 seq2/seq6 上评估。这会造成：

- Agent 看不到跨 token 复用机会；
- seq1 上看似有效的局部优化，在 seq6 上收益变小或回退；
- DAG 证据和最终指标使用了不同规模，解释力不足。

建议保留 seq1 作为“局部可读图”，同时增加 seq2/seq6 聚合证据，并区分：

- 单 token 固定成本；
- 随 seq 线性增长的成本；
- 跨 token 可摊销成本；
- 只有长序列才出现的 VRF 容量与调度冲突。

### 7.3 迁移到 master 时必须重做的部分

当前 v10 的 event trace 和 DAG 语义针对旧 ISA。主线新增了 DMA 指令、硬件 MHA/MHA-v2、chain/scoreboard 和多个并发执行单元，不能只扩充 opcode 名称后直接复用。

主线适配至少需要：

1. 将 DMA、MHA、MHA-v2、chain start/wait 纳入事件模型；
2. 用 VRF bank、地址范围和真实读写集合建立 RAW/WAR/WAW 依赖；
3. 根据 dispatcher 时间线表示并发，而不是按 trace 顺序强制串行；
4. 将 VRF 容量、Tile 维度等常数从硬编码改为主线配置或 `TileSpec`；
5. 输出执行单元占用、重叠率、scoreboard 等待和关键路径；
6. 把 DSE 目标接到 `dsl/backend/bridge.py` 的 lowering/binding 选择。

主线 `docs/tile-centric-history/gap-analysis.md` 也提出需要显式任务/指令 DAG，但更强调 trace-time reconstruction：DAG 用于测量并行重叠和验证编译器/scoreboard hazard，而不是代替主线静态调度器。这与 v10 的结构化 DAG 基础兼容，但语义必须升级。

## 8. 主线 DSL/后端与未来优化入口

当前 `master` 的关键优化接口集中在 `dsl/backend/bridge.py`，包括：

- `OpToChainTable`：高层操作到执行 chain 的选择；
- `ChainToCallTable`：chain 到具体调用实现的绑定；
- `attention_sw`：软件 attention 路径；
- `attention_v2`：MHA-v2 路径。

这意味着面向主线的 Agent 策略应分为两层：

1. **安全的规则级 DSE**：在已验证的 lowering、执行后端、Tile、DMA/MHA 策略间选择，优先进入主线。
2. **受控的源码生成/修改**：仅用于旧 SW 固件路径或实验性候选，必须通过完整门禁，不能直接替换主线所有实现。

这种改造还能避免 Agent 为降低单一指标而删除功能、绕过硬件路径或针对某个 dim/seq 过拟合。

## 9. 测试体系与本次实测

### 9.1 测试规模

| 版本 | 测试特征 |
|---|---|
| Gitee master | 约 26 个 Python `test_*.py`，另有 HDL 测试；E2E 覆盖多后端和多轮验证 |
| 旧闭环 x86 | 约 5 个测试文件；E2E 重点覆盖 6 组 dim/hidden/seq |
| v11 | 测试很少，主要依赖单个 BERT E2E 文件 |
| 当前 v10 | 约 9 个主要测试文件；增加门禁、Skill、结构化 DAG 和闭环快照测试 |

旧体系的六组 E2E 参数为：

- dim2、hidden4、seq2/seq6；
- dim4、hidden8、seq2/seq6；
- dim4、hidden4、seq2/seq6。

主线 E2E 已扩展到约 18 组配置，覆盖：

- SW 路径；
- MHA HW/MC；
- MHA-v2；
- DMA；
- Portal IR；
- TTIR、TileLang 和 TileLang primitive。

主线验证轮次还包括 FP16 golden、软件模拟器、HDL sequential/batch、bypass、ideal 以及可选 Verilator，显著超过旧闭环仅围绕软件固件模拟的验证边界。

### 9.2 v10 本次实测结果

平台辅助测试：

```text
44 passed, 1 failed
```

唯一失败：

```text
tests/test_skillctl.py::test_vrf_cache_v2_contains_staged_capacity_and_metric_contracts
```

原因不是 Skill 内容丢失，而是测试仍要求 `vrf-cache` 版本为 `2.1.0`，实际规范与锁文件已升级为 `2.2.0`。这是典型的版本元数据同步缺陷，应在提交前统一测试、lock 和版本快照的预期。

固件正确性及 E2E：

```text
python3 -m pytest \
  tests/test_npu_vrf_validation.py \
  tests/integration/test_bert_e2e.py \
  -q -rs -p no:cacheprovider

13 passed in 3.70s
```

这说明当前恢复后的未优化 `bert_layer.c` 在现有 dim2/dim4、seq2/seq6 回归矩阵中通过。

### 9.3 master 本次测试限制

按 README 运行主线测试时，在收集阶段出现 20 个错误，统一原因是：

```text
ModuleNotFoundError: No module named 'amaranth'
```

这是当前 Python 环境缺少主线 HDL 依赖，不是已证明的代码回归。正式分支开发前应建立独立环境并安装主线声明的 `amaranth`、`pytest`、`numpy`、`pyelftools` 后重新运行。

另外，主线 CI workflow 中仍存在对若干当前树中不存在的旧测试路径的引用，例如旧 `tests/test_kernels.py`、`hdl/tests/` 或 `tests/test_bert_encoder.py`。提交前应以当前 README 和实际测试树为准复核 CI 配置。

## 10. 当前 v10 工作树风险

当前 v10 的 Git 状态混合了长期开发和运行结果：

- 34 个受跟踪文件被修改；
- 282 个受跟踪文件被删除，主要是历史运行结果；
- 约 3396 个未跟踪文件；
- 过滤构建/结果后仍有约 28 个已跟踪源码改动和约 55 个未跟踪源码/文档候选；
- 仅源码差异也约有 3186 行新增、1018 行删除。

原始 Git tree diff 的文件数和行数会被 `_build`、`_out`、`results/`、备份文件、ELF/二进制等严重放大，因此不能作为实际功能补丁规模。

直接从该工作树提交的风险：

- 把运行结果或本机构建产物带入仓库；
- 将多个独立功能压成一个无法审查的大提交；
- 用旧架构固件覆盖主线新架构；
- 将自动生成的 `.opencode/skills` 与规范源重复维护；
- 难以确定失败来自移植、依赖环境还是原有脏状态。

## 11. 建议的分支与提交策略

### 11.1 分支基线

建议从干净的 Gitee `master@cc05ffc` 新建功能分支。不要以下列内容作为新分支起点：

- 当前脏 v10 工作树；
- 无历史的 v11 快照；
- 已落后于新架构的 `explore/...-x86`；
- ZIP 快照。

旧闭环分支和 v10 仅作为“选择性移植的功能来源”。

### 11.2 推荐 PR 拆分

建议按以下顺序移植，每个 PR 独立可测试、可回滚：

#### PR-A：主线 trace/DAG 基础

- 移植通用事件和 DAG 数据模型；
- 适配主线 DMA、MHA、MHA-v2 和 chain/scoreboard；
- 不引入 Agent，不修改主线固件算法；
- 增加最小 trace-to-DAG 单测。

#### PR-B：结构化、多尺度 DAG

- 移植 `npu_dag_structured.py` 的候选、生命周期、摘要和稳定 ID；
- 增加 seq1/seq2/seq6 聚合；
- 增加 RAW/WAR/WAW、并行单元和关键路径信息；
- VRF/Tile 容量读取主线配置。

#### PR-C：验证、指标和 DAG 门禁

- 移植三个独立 gate；
- 让 gate 调用主线现有测试矩阵；
- 明确必需配置与可选 HDL/Verilator 配置；
- 不允许候选修改测试、模拟器或 golden。

#### PR-D：Skill 规范与版本管理

- 以 `jimu-dse/docs/skills/isa/*.md` 为唯一规范源；
- 引入版本快照、lock 和 SHA256 manifest；
- OpenCode 显式接收全部 Skill；
- Pi 使用合成文件；
- 决定 `.opencode/skills` 是提交生成物还是 CI 临时产物，避免双源漂移。

#### PR-E：主线 Agent/DSE 闭环

- 将 Agent 决策优先绑定到 `OpToChainTable`、`ChainToCallTable` 和 attention 后端选择；
- 保留旧 C 源码修改作为可选实验后端；
- 支持固定金标准和可选择优化起点；
- 所有候选经过 PR-C 门禁后才可成为 `candidate_best`。

### 11.3 首次提交前的必做事项

1. 修复 Skill `2.1.0`/`2.2.0` 测试不一致；
2. 在干净主线环境安装依赖并跑通现有测试；
3. 更新或确认 `.gitignore`，排除 `results/`、构建目录、缓存、ELF、日志和临时备份；
4. 明确 Skill 规范源和生成目录的提交政策；
5. 为每个移植功能保留来源说明，方便追溯旧实验分支和本地 v10；
6. 不使用 `git stash/reset/checkout` 操作当前工作树，也不通过修改测试或模拟器让候选通过。

## 12. 推荐的近期技术优先级

综合收益、风险和主线匹配度，建议顺序为：

1. **先修基础可信度**：Skill 版本测试、干净环境、主线测试、产物隔离。
2. **再迁移 trace/DAG，不先迁移 Agent**：先证明主线数据可观测且依赖关系正确。
3. **补多尺度 DAG**：seq1 保持可读，seq2/seq6 提供真实复用和容量证据。
4. **将 DAG 对齐主线并发硬件**：以真实执行单元、scoreboard 等待和关键路径为优化依据。
5. **最后接 Agent**：让 Agent 优先选择已经验证的 lowering/binding 规则，减少整文件自由改写。

该顺序能保留 v10 已验证的平台化成果，同时避免把旧架构假设带入主线。

## 13. 最终判断

当前 v10 不是“落后的旧仓库副本”，而是一个建立在旧固件架构上的、功能更完整的 Agent 优化实验平台；Gitee `master` 则是硬件和编译执行架构更先进、但尚未集成 JIMU 闭环的平台主线。

两者最合理的结合方式不是把任一方整体覆盖到另一方，而是：

```text
以 master 为代码与架构基线
  + 移植 v10 的结构化观测、门禁和 Skill 治理
  + 将 Agent 的决策面改造成主线 lowering/binding DSE
  + 保留旧固件闭环作为兼容/实验后端
```

这样既能延续当前已完成的 DAG-PR、PRJ4/PRJ5 和基线治理工作，也能让后续分支具备清晰历史、完整测试边界和可审查的功能增量。
