# JIMU NPU RTL 并行时序模拟器说明

> 本文可作为项目解题报告中“新增 NPU RTL 模拟器与智能体优化闭环”部分的主体材料。说明以当前仓库中的 SystemVerilog、Verilator 回放适配器、时序配置和闭环脚本为准。

## 1. 建设目的与定位

项目原有的 Python/C NPU 模拟器能够执行 RISC-V 固件发出的 NPU 指令，并给出 BERT 等工作负载的数值结果，但它更适合作为**功能正确性模型**：命令之间的重叠、有限队列、结构冲突和真实的等待关系无法只靠“逐条指令延迟相加”准确表达。

为此，本项目增加了一个由 Verilator 驱动的、支持并行资源调度的 NPU RTL 时序模拟器。其核心价值是：对同一份动态 NPU 命令轨迹，在时钟周期粒度上模拟有限 ROB、依赖检查、多个控制器并行工作、共享 DRAM 总线、片上 SRAM bank 冲突、流水线启动间隔和栅栏，从而判断固件的数据流变换究竟缩短了最终完成时间，还是仅仅减少了某一类操作。

需要明确其边界：

- Python/C 功能模拟器负责计算数据并作为数值正确性的判定标准；
- RTL 模拟器负责命令、控制和资源调度时序；
- 当前 RTL 不包含 FP16 算术数据通路，也不是完整 SoC 的位精确仿真；
- 所有候选必须在同一版本化时序配置下比较，模拟周期不能直接解释为硅后实测频率或真实芯片延迟。

这种“功能模型给结果、RTL 模型给周期”的协同仿真方式，既保留了软件模拟的可用性，又使软件流水、预取、双缓冲和片上驻留等并行优化有可度量的硬件反馈。

## 2. 总体工作流程

```text
RISC-V 固件 ELF
      │
      ▼
MiniRV64 ISS + NPU 功能/时序设备
      │  记录动态 NPU 命令、DRAM 区间、寄存器资源、张量与源码来源
      ▼
Python RTL 适配器 emulator/npu_rtl_sim.py
      │  生成 unit、latency、II、RAW/WAR/WAW 掩码、bank 掩码、DRAM/fence 标志
      ▼
Verilator C++ 回放器 sim/jimu_rtl_harness.cpp
      │
      ▼
SystemVerilog 时序核 rtl/jimu_npu_timing_core.sv
      │  有限 ROB + 五类控制器 + scoreboard + DRAM/bank/栅栏约束
      ▼
逐事件 schedule + RTL 原始计数器 + 派生并行指标 + 可选 VCD
      │
      ├── 人工分析、可视化和报告
      └── closed_loop.py 智能体闭环评分与候选晋升
```

RTL 采用“离线轨迹回放”方式。固件先在功能模拟器上运行并产生完整动态命令流，再将命令编码后送入 RTL。这样可以复用现有 ISS、设备模型、张量语义和黄金结果，同时把调度决策交给可综合的 SystemVerilog 核。

## 3. 模拟器实现的主要功能

### 3.1 有限 ROB 与跨控制器乱序发射

RTL 内部默认配置 16 项 ROB。命令按程序顺序进入 ROB，前端每周期最多接收一条命令，调度器每周期最多发射一条满足条件的命令。

模拟器将命令划分到五类相互独立的控制器：

| 控制器 | 主要命令 | 可见资源 |
|---|---|---|
| Load | DRAM 读入 | load、共享 `dram_bus`、目标 SRAM bank |
| Store | DRAM 写回 | store、共享 `dram_bus`、源 SRAM bank |
| MVU | `MV_MUL` 等矩阵向量计算 | mvu、MRF/VRF bank |
| Vector | 向量运算、激活、Softmax、LayerNorm 等 | vector、VRF/SRF bank |
| Control | 标量和配置类操作 | control |

同一控制器中的未发射命令保持顺序；不同控制器之间允许越过一个暂时阻塞的命令。因此，独立的 Load、MVU、Vector 和 Control 操作能够同时处于执行状态。例如下一块权重的 DRAM 预取可以和当前块的 MVU 计算重叠，但前提是不存在数据依赖、bank 端口冲突、ROB 窗口限制或栅栏。

### 3.2 语义依赖 scoreboard

适配器把每条命令的 `uses` 和 `defs` 编码为默认 128 位的语义资源掩码，RTL 对所有更老且尚未完成的命令检查：

- RAW：后续读必须等待前序写完成；
- WAR：后续写不能覆盖前序尚未完成的读；
- WAW：对同一物理位置的写必须保持正确顺序。

VRF、MRF、SRF、配置寄存器和按粒度切分的 DRAM 地址区间被视为物理资源。`pipe` 与 `vpipe_a` 则转换成类似 SSA 的弹性版本化 token，避免所有流水线临时值共用一个名字而产生虚假的全局 WAW 依赖。

资源位只有在两个语义对象不可能同时存在于同一 ROB 窗口后才复用。若 128 位仍不足，适配器会保守哈希，并在 `resource_encoding.conservative_hash_collisions` 中报告碰撞；有效优化结果应检查该值，不能把碰撞造成的额外序列化误认为真实硬件瓶颈。

### 3.3 DRAM、片上 bank 与流水线约束

- Load 和 Store 共用一条 DRAM 总线，DRAM 传输在整个建模时长内互斥；
- DRAM 总线可以与无依赖的 MVU 或 Vector 计算并行；
- 本地 SRAM 按 bank 建模，每个 bank 同时允许一条读流和一条写流；相同方向访问同一 bank 会形成结构冲突；
- 各控制器具有可配置的启动间隔 II。上一条命令未达到下一次允许发射的周期时，新命令产生 unit stall；
- 当前保守配置把 `INST_ISSUE` 当作完整 chain fence，栅栏必须位于退休队首，年轻命令也不能越过更老栅栏。

### 3.4 可审计的延迟配置

每条命令都携带 `latency` 和 `initiation_interval`，而不是把固定延迟写死在 RTL 中。参数来自 `jimu-dse/timing/jimu-rtl-dim4.yaml` 等版本化配置，事件中还会保存 `latency_source` 和 `memory_tier` 以便追溯。

以 dim4 配置为例：

| 操作 | 延迟/II 示例 | 来源 |
|---|---:|---|
| MVU `MV_MUL` | 8 / 10 周期 | HDL 控制器行发射、加法树和完成返回的一阶契约 |
| 向量加减乘、GELU/EXP | 3 / 3 周期 | MFU 命令包络 |
| Softmax/LayerNorm | 6 / 1 周期 | 单组 SLU 流水契约 |
| `M_RD@18` | 17 / 17 周期 | `1 + native_dim²` 的 VecToMat 排空 |
| dim4 片上向量传输 | 2 周期 | 1 周期建立 + 8 B/cycle |
| dim4 最小 DRAM 传输 | 14 周期 | 12 周期建立 + 16 B 最小突发/8 B/cycle |

外部 DRAM 传输采用：

```text
dram_cycles = direction_setup
            + ceil(max(payload_bytes, minimum_transfer_bytes)
                   / dram_bytes_per_cycle)
```

片上传输采用：

```text
on_chip_cycles = direction_setup
               + ceil(payload_bytes / on_chip_bytes_per_cycle)
```

这些参数来自现有 HDL 抽象所推导的第一版控制器契约与显式存储层次假设，状态为 `hdl-derived-first-pass-uncalibrated`。因此报告实验应同时记录配置名和 SHA-256，而不能脱离配置版本比较周期。

### 3.5 调试和证据产物

一次 RTL 分析会产生：

- `rtl-timing-schedule.json`：逐命令入队、开始、结束、资源、依赖、阻塞原因、张量和源码信息；
- `timing-schedule.json`：供图挖掘和闭环读取的同结构通用别名；
- `rtl-commands.txt`：实际送入 RTL 的命令包；
- `rtl-harness-schedule.csv`：C++ harness 记录的原始周期与计数；
- `cross-layer-graph.json/.txt`：张量—命令—依赖—时序的跨层证据；
- `run-summary.json`：功能等价性和汇总性能；
- `rtl-wave.vcd`：可选的逐信号波形，可用 GTKWave 检查。

## 4. 性能计数方式

本项目把性能数据分为三层：RTL 每周期原始计数、Python 从事件区间派生的并行指标、闭环用于候选晋升的归一化分数。三层不能混用。

### 4.1 RTL 原始计数器

| 输出指标 | RTL 中的计数条件 | 正确解读 |
|---|---|---|
| `rtl_counter_cycles` | 复位结束后每个仿真时钟加 1，直到输入耗尽且 ROB 完全空闲 | 完全空闲时间，包含末条命令完成后的顺序退休尾部 |
| `rtl_counter_active_cycles` | `rob_count != 0` | ROB 中存在未退休命令的周期，包括已完成但等待顺序退休的项 |
| `rtl_counter_memory_compute_overlap_cycles` | `dram_busy && compute_busy` | DRAM 与 MVU/Vector 至少各有一条活动命令的真实交集周期 |
| `rtl_counter_frontend_full_stall_cycles` | `cmd_valid && !cmd_ready` | 有待输入命令但 ROB 已满的前端背压周期 |
| `rtl_counter_dependency_stall_cycles` | 最老未发射命令被 RAW/WAR/WAW 阻塞 | 数据/名字依赖压力 |
| `rtl_counter_unit_stall_cycles` | 当前周期早于目标控制器的 `next_issue` | 控制器 II 压力 |
| `rtl_counter_dram_stall_cycles` | 待发命令需要 DRAM 且总线忙 | 共享 DRAM 结构冲突压力 |
| `rtl_counter_bank_stall_cycles` | 待读/写 bank 与活动同向端口冲突 | 片上 SRAM 端口压力 |
| `rtl_counter_barrier_stall_cycles` | 栅栏不在队首或被更老栅栏阻挡 | chain fence/控制顺序压力 |
| `rtl_counter_dispatches` | 一条命令被调度 | 总发射数 |
| `rtl_counter_completions` | 活动命令剩余周期降为 0 | 总完成数，同周期可完成多条 |
| `rtl_counter_max_inflight` | 活动但未完成命令数的历史最大值 | RTL 观察到的最大执行中并发度 |
| `rtl_counter_unit_N_busy_cycles` | 对应控制器的 busy bit 为 1 | 该控制器至少有一条活动命令的周期并集 |

阻塞计数采用**单命令、单原因、每周期一次**的归因方法：只观察当前最老的未发射命令；若它同时存在多个阻塞条件，按照 barrier/order、dependency、unit II、DRAM、bank 的优先级只记录一个原因。C++ harness 同时把这一周期记到对应事件的 `rtl_stall_cycles_by_reason`。

这意味着 stall 计数是诊断“压力出现在哪里”的指标，而不是可以相加的延迟分解。同一个调度变化可能同时改变多类计数；被计数的最老命令虽然等待，更年轻且属于其他控制器的独立命令仍可能在同周期发射。因此不能写成：

```text
总周期 = dependency_stall + dram_stall + bank_stall + ...
```

也不能把多个事件的 queue wait 全部相加，直接当作可节省的墙钟周期。

### 4.2 从事件时间区间派生的并行指标

记每条命令执行区间为 `[start_i, finish_i)`，定义：

```text
T = max(finish_i)                         # 最后一条命令完成时间
S = sum(finish_i - start_i)               # 所有命令时长的串行和
U = union_length(all active intervals)    # 至少一条命令活动的周期并集

gross_overlap_cycles = S - U
scheduler_idle_hole_cycles = T - U
net_parallelism_savings_cycles = S - T
                               = gross_overlap_cycles
                               - scheduler_idle_hole_cycles
```

主要指标的含义如下：

- `rtl_predicted_npu_cycles` / `rtl_completion_makespan_cycles`：均为 `T`，即闭环的主评分指标；
- `rtl_idle_cycles`：来自 `rtl_counter_cycles`，通常比 makespan 多一个很短的顺序退休尾部；
- `rtl_retirement_tail_cycles = rtl_idle_cycles - T`；
- `serial_command_cycles = S`：假设所有命令完全串行时的工作量参考；
- `gross_overlap_cycles = S - U`：重叠执行带来的总重叠工作量；存在三条及以上并发命令时，它不等同于简单的“发生并行的时钟数”；
- `scheduler_idle_hole_cycles = T - U`：从第 0 周期到最后完成之间没有任何命令执行的空洞；
- `net_parallelism_savings_cycles = S - T`：相对完全串行的净节省；旧名 `overlap_saved_cycles` 只是兼容别名；
- `memory_compute_overlap_cycles`：DRAM 活动区间与 MVU/Vector 活动区间的交集长度；
- `memory_compute_overlap_ratio = memory_compute_overlap_cycles / dram_busy_cycles`；
- `max_concurrent_ops`：时间线上同时活动命令数的最大值；
- `load/mvu/vector/..._utilization = 对应控制器忙周期并集 / T`；
- `dram_bus_utilization = DRAM 忙周期并集 / T`。

报告性能时应首先比较 makespan，再用串行工作量、重叠、空洞、利用率和 stall 解释原因。例如减少 DRAM 命令可能让 `gross_overlap_cycles` 变小，因为可重叠的总工作本身减少了；只要 makespan 同时大幅降低，这仍是有效优化。

### 4.3 数据量计数的口径

项目中同时存在三种字节口径：

- `dram_elements`：逻辑传输元素数，与容器和总线数据类型无关；
- `functional_container_bytes`：功能模拟器 NumPy `float32` 容器字节数，即元素数乘 4；旧字段 `total_bytes` 是它的兼容别名；
- `rtl_payload_bytes` / `logical_dram_payload_bytes`：按 RTL 配置的数据类型计算，默认 FP16，即元素数乘 2；
- `modeled_dram_transaction_bytes`：进一步考虑最小突发后的建模总线事务字节数。

因此不能把 `total_bytes` 描述成 RTL 总线流量，也不能跨不同 element size 的 profile 直接比较字节指标。

### 4.4 闭环评分公式

`rtl-cycle-optimization` 目标只给 `rtl_predicted_npu_cycles` 权重 1.0，并按最小化方向评分。设本次运行固定基线为 `B`、候选周期为 `C`：

```text
score = (B - C) / abs(B)
```

即分数就是相对运行基线的周期改善比例。候选必须先通过全部门禁，然后分数还需至少比当前最佳分数高 `min_score_delta`（当前为 0.001）才会晋升。注意评分始终归一化到运行开始时的基线，而不是上一轮候选。

## 5. 与 Agent 闭环脚本的结合方式

### 5.1 配置入口

RTL 优化闭环由 `jimu-dse/goals/rtl-cycle-optimization/goal.yaml` 声明，关键配置为：

- 允许修改的目标：`firmware/bert/bert_layer.c`；
- 固定 RTL profile：`jimu-dse/timing/jimu-rtl-dim4.yaml`；
- 评分指标：最小化 `rtl_predicted_npu_cycles`；
- Agent 上下文：固件、RTL profile、基线分析和三项领域技能；
- 门禁：修改范围、固件构建、完整性能 probe、BERT 数值正确性；
- 运行产物：提示词、候选、diff、probe、图、Agent stdout/stderr 和最终报告。

闭环驱动支持 `opencode` 和 `pi` 两类 Agent 后端；当前目标默认使用 OpenCode。RTL、profile、模拟器、测试、指标实现和运行产物都不在 Agent 的允许修改范围内，从制度上避免通过篡改评分器“获得优化”。

### 5.2 一轮优化如何执行

一轮完整流程如下：

1. 将工作文件恢复为当前已验证的最佳候选；
2. 构建固件并进行一次 pre-agent probe；
3. 功能/时序设备执行 ELF，采集动态命令、张量和源码信息；
4. Verilator RTL 回放命令并生成时序 schedule；
5. 从 schedule 中提取主分数、基线差值、资源利用率、并行重叠、关键因果链、主要阻塞事件和依赖资源，形成有长度上限的 Agent 提示；
6. Agent 读取证据与技能，只实现一个主要优化假设；
7. 驱动器恢复任何越权文件，只保留允许范围内的候选修改；
8. 驱动器重新构建、重新 probe，并执行全部正确性门禁；
9. 门禁全部通过后按固定基线计算分数；若优于当前最佳则晋升，否则恢复当前最佳候选；
10. 保存本轮候选、diff、原始计数、schedule、Agent 日志和晋升决定，供下一轮使用。

下一轮提示还会包含最近候选相对上一个已验证版本的实测指标增量，帮助 Agent 避免重复已经失败的资源迁移。报告和 prompt 中只放有界的瓶颈摘要，完整事件数组保留在 `timing-schedule.json` 中供按需审计。

闭环结束时，驱动器会把用户开始运行前的固件恢复到工作树；最佳结果单独保存在运行目录的 `candidate_best.c`，不会悄悄覆盖用户原文件。

### 5.3 Agent 应如何使用性能证据

Agent 的正确工作顺序是：

1. 先查看 `critical_path_top_events` 和 `critical_path_top_blockers`，定位接近最后完成链的长事件；
2. 对候选区域打开完整 `timing-schedule.json`，查看 `dependency_predecessors`、`dependency_reasons`、`dependency_resources`、bank、源码和张量身份；
3. 提出一个有因果依据的变换，如权重片上驻留、双缓冲预取、VRF 重命名/换 bank、消除非可观察中间量的 DRAM 落地、位置级软件流水；
4. 同时证明数值语义、容量、地址别名、bank 端口、活跃区间和可观察输出均安全；
5. 预测瓶颈迁移，并以 makespan 的实际下降作为最终判断。

“关键路径”字段当前是从最晚完成事件沿已知依赖和资源前驱回溯得到的一条事后因果链，并非严格的零松弛关键路径分析。因此它适合缩小排查范围，但不能替代对完整依赖图的确认。

### 5.4 推荐运行命令

在 Linux/WSL 的项目根目录执行：

```bash
# 安装 Python 依赖；系统还需要 Verilator、CMake、C/C++ 编译器和 RISC-V 交叉编译器
python3 -m pip install -r requirements.txt -r requirements-timing.txt

# 检查目标配置和 Agent 将收到的实际提示词
python3 jimu-dse/scripts/closed_loop.py validate-config --goal rtl-cycle-optimization
python3 jimu-dse/scripts/closed_loop.py render-prompt --goal rtl-cycle-optimization

# 先跑一轮，审计结果后再增加轮数
python3 jimu-dse/scripts/closed_loop.py run \
  --goal rtl-cycle-optimization \
  --agent opencode \
  --max-iterations 1
```

也可以使用兼容 shell 包装器：

```bash
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal rtl-cycle-optimization --agent opencode
```

一次闭环位于 `jimu-dse/results/run-*`。至少应检查：

- `resolved-config.yaml`：运行时实际解析配置；
- `baseline-probe.json` 与 `baseline/timing-schedule.json`：固定基线；
- `prompt-N.txt`：Agent 收到的证据和约束；
- `agent-N.stdout.jsonl`、`agent-N.stderr.log`：完整 Agent 行为；
- `candidate-N.c`、`diff-N.patch`：候选修改；
- `probe-N.json`、`iteration-N.json`：门禁、原始指标、分数和晋升原因；
- `iteration-N/timing-schedule.json`：候选完整 RTL 调度；
- `candidate_best.c`、`run-summary.json`、`report.md`：最终最佳版本和汇总。

### 5.5 不启动 Agent 的独立分析

先构建内核和固件，再直接分析当前固件：

```bash
make kernels firmware
python3 scripts/analyze_firmware.py \
  --manifest jimu-dse/workloads/bert-dim4-seq6.yaml \
  --rtl-profile jimu-dse/timing/jimu-rtl-dim4.yaml \
  -o _out/bert-rtl --no-render
```

若已经有 `trace-events.json`，可以只回放 RTL：

```bash
python3 scripts/simulate_rtl.py \
  --events _out/firmware-analysis/trace-events.json \
  --manifest jimu-dse/workloads/bert-dim4-seq6.yaml \
  --profile jimu-dse/timing/jimu-rtl-dim4.yaml \
  -o _out/firmware-analysis/rtl-timing-schedule.json
```

加入 `--rtl-wave`（统一分析入口）或 `--wave 文件名.vcd`（独立回放入口）可生成 VCD。

## 6. 已归档实验如何说明模拟器价值

### 6.1 dim4、seq6 十轮闭环

归档运行 `run-20260822-161925-7452` 的基线与第 7 轮最佳候选均通过功能门禁：

| 指标 | 基线 | 最佳 | 说明 |
|---|---:|---:|---|
| RTL makespan | 7854 | 4481 | 降低 42.95% |
| 串行命令周期和 | 10216 | 6028 | 总工作量显著减少 |
| DRAM 元素数 | 2256 | 888 | 重复搬运和中间落地减少 |
| gross overlap | 2809 | 1907 | 可重叠工作总量随命令减少而下降 |
| 调度空洞 | 447 | 360 | 无活动区间缩短 |
| 存储/计算重叠 | 2286 | 1363 | 绝对重叠减少，但 makespan 更短 |
| 最大并发命令数 | 4 | 3 | 峰值并发数并不等于性能 |

第 10 轮进一步把 DRAM 元素数降到 600、串行命令周期降到 5212，但 makespan 从 4481 回升至 4555，因此没有晋升。其原因是有效并行节省从 1547 降至 657，调度空洞从 360 增至 585。这个反例证明：只优化访存量或指令总时长会选错候选，必须用并行 RTL 的最后完成时间评分。

### 6.2 dim16、hidden16、seq16 十轮闭环

归档运行 `run-20260822-200156-21529` 在大配置上把 RTL makespan 从 51924 降到 11069，归一化改善为 78.68%，第 10 轮为最佳候选。该结果说明同一接口可以扩展到更大工作负载，但它仍是 `jimu-rtl-dim16.yaml` 固定时序假设下的候选对比，不能当作芯片实测加速比。

## 7. 验证、可视化与报告复现

RTL lint 和针对性测试：

```bash
make rtl-lint
make rtl-test
```

测试覆盖：SSA pipeline token、独立 DRAM/计算重叠、真实依赖阻塞、SRAM bank 读端口冲突、MRF ping-pong 预取、消除 DRAM 中间落地和 VRF bank 轮换。

项目还提供答辩用并行图生成脚本，它只读取已保存结果，不会重新仿真或修改闭环：

```bash
python scripts/visualize_rtl_parallelism.py --latest 3
```

输出位于 `_out/rtl-parallelism`，包括近期运行周期对比、baseline/best 控制器时间轴和关键因果链轨道。图中每一行对应 Load、Store、MVU、Vector 或 Control，淡蓝背景表示至少两个资源并行，适合直观说明并行从何处产生。

## 8. 当前局限与后续改进

1. 当前是命令/控制 RTL，不含完整 FP16 数据通路，数值正确性仍依赖功能模拟器。
2. 轨迹离线回放，RTL FIFO 反压尚不会反向改变 RISC-V CPU 的轮询指令数；锁步 Python 时序设备用于补充这部分行为。
3. SRAM bank 掩码由资源和地址元数据推导，命令在整个建模时长内保守占用端口。
4. 外部 DRAM、MVU、Vector 等延迟仍是一阶未校准参数，应通过组件级 microbenchmark、真实 HDL/RTL 仿真或 FPGA 测量继续标定。
5. `INST_ISSUE` 的完整栅栏语义较保守，可能低估合法的跨 chain 并行。
6. 事后因果链不等同于形式化关键路径；后续可引入零松弛分析和更精细的资源前驱。
7. 下一阶段可把 RTL MMIO/FIFO 接口直接连入 ISS，加入 AXI 类请求/响应、显式 MRF ping-pong 地址、DPI 功能核和面积/功耗/综合结果，使闭环从固件数据流优化扩展为软硬件联合设计空间搜索。

## 9. 结论

本项目新增的 RTL 时序模拟器把“并行”从抽象描述变成了可复现、可计数、可回溯的调度事实。其核心不在于给每条指令再附加一个固定延迟，而在于同时建模有限观察窗口、真实数据依赖、多控制器并行、共享资源竞争和栅栏，并用最终完成 makespan 统一评价固件变换。

与 Agent 闭环结合后，智能体负责根据时序和跨层证据提出受限的固件修改，固定模拟器负责产生客观周期，独立功能模型和测试负责拦截错误，闭环驱动负责晋升或回退。这使开放式代码生成转化为一个边界明确、结果可审计、能够断点续跑的工程优化过程。

## 10. 主要代码与文档索引

| 内容 | 仓库路径 |
|---|---|
| SystemVerilog 时序核 | `rtl/jimu_npu_timing_core.sv` |
| Verilator C++ harness | `sim/jimu_rtl_harness.cpp` |
| Python trace 编码与 schedule 生成 | `emulator/npu_rtl_sim.py` |
| 统一固件分析入口 | `scripts/analyze_firmware.py` |
| 已有轨迹 RTL 回放入口 | `scripts/simulate_rtl.py` |
| 智能体闭环驱动 | `jimu-dse/scripts/closed_loop.py` |
| RTL 优化目标 | `jimu-dse/goals/rtl-cycle-optimization/goal.yaml` |
| dim4 时序配置 | `jimu-dse/timing/jimu-rtl-dim4.yaml` |
| RTL 数据流技能 | `jimu-dse/docs/skills/isa/rtl-dataflow.md` |
| RTL 设计说明 | `docs/rtl-timing-simulator.md` |
| 时序参数来源与边界 | `docs/hdl-derived-timing-parameters.md` |
| 并行可视化说明 | `docs/rtl-parallelism-visualization.md` |
| RTL 单元测试 | `tests/unit/test_npu_rtl_sim.py`、`tests/unit/test_npu_rtl_optimization_space.py` |
