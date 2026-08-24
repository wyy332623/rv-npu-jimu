# NPU 闭环 FW-HW 协同：设计与技能

> 本文解释稳定的架构概念。当前已实现能力、schema 和验证结论见[项目状态](project-status.zh.md)，具体 DAG 证据格式见[结构化 DAG](structured-dag.zh.md)。

## 摘要

本文描述 NPU 固件的自动优化闭环。系统将固件优化视为一个受约束的搜索问题：候选固件必须保持数值正确，同时改善 DRAM 带宽、片上寄存器文件利用率或计算量。

核心流程为：

```text
PROBE → ANALYZE → AGENT → VALIDATE → DEPLOY → LOOP
```

每轮流程都会构建固件、运行 NPU 模拟器、统计 DRAM 流量、生成微操作 DAG 和 DRAM cluster，然后由 Agent 根据技能库生成候选补丁，重新构建并验证。

## 闭环阶段

| 阶段 | 工作 |
|------|------|
| Probe | 根据当前工作负载（如 `bert`、`adder`）构建固件，运行模拟器，记录 DRAM 流量、指令跟踪和 DAG |
| Analyze | 比较 DRAM 比例，识别 DAG 中的保存-加载对，并计算各簇的算术强度 |
| Agent | Agent（`pi` 或 `opencode`）读取 DAG、DRAM cluster 和目标技能，生成目标固件的修改 |
| Validate | 重新构建并运行模拟器，通过工作负载对应的测试（如 `pytest`）判断通过或失败 |
| Deploy | 保存候选固件和审计所需的运行结果及 DAG |
| Loop | 若指标改善则继续下一轮，否则收敛并结束 |

## 状态模型

每执行一条指令，模拟器状态可能改变：

- DRAM：输入、权重、中间 tensor 和输出；
- MRF：当前驻留的权重 tile；
- VRF：MVM 输入、累加器、MFU cache 和临时值；
- SRF：Softmax 的最大值和总和；
- pipeline/vpipe_a：当前链内的隐式向量值；
- 标量寄存器：mask、tile 几何尺寸、迭代次数和精度模式。

Agent 必须理解这些状态的定义-使用关系，才能判断某次 DRAM 保存是否可以由 VRF cache 或 pipeline 传递替代。

## 优化技能

### VRF Cache

按 L1→L2 分阶段减少 DRAM：L1 消除中间 Tensor 的保存—加载往返；L2 在完整验证矩阵的生命周期和 bank/row 分配证明下缓存循环不变量或序列输入。L3 权重驻留不属于当前已证明范围。

### Dim Optimize

将多 tile 投影重构为单 tile 投影，使 `NATIVE_DIM` 匹配 `hidden_size`，减少 `MV_MUL` 和 `M_RD_DRAM` 次数。

### INC Folding

将相同地址的 `V_WR_DRAM` + `V_RD_DRAM` 对折叠为 INC 变体。INC 指令必须使用 LO 格式，不能使用 SI 格式。

### DAG Analyze

优先读取结构化 JSON/JSONL：从微操作 `defs/uses` 建立依赖，合并真实 seq2/seq6，检查 dim2/dim4、hidden4/hidden8 的验证矩阵，并生成 primitive、宏和跨配置 VRF 分配证明。TXT/DOT/SVG 只用于人工取证。

## 约束和补偿

优化必须保持：

1. 固件可用 RISC-V GCC 编译；
2. NPU ISA 和 MMIO 协议不变；
3. 最终输出通过 NumPy golden reference 和端到端测试；
4. 共享 VRF、MRF、SRF 和 DRAM 地址没有 RAW/WAR/WAW 冲突；
5. 不能把需要跨链的 pipeline 值误认为会跨 `INST_ISSUE` 保留。

如果近似计算或 FP16 截断造成误差，必须明确记录误差预算和补偿方法，不能用降低测试标准代替验证。

## 度量

| 指标 | 含义 |
|------|------|
| `total_bytes` | DRAM 读写总字节数，主要用于 VRF cache 优化 |
| `mv_mul_count` | `MV_MUL` 与 `MV_MUL_INC` 次数，主要用于计算效率优化 |
| `mat_rd_ops` | 权重 tile 加载次数 |
| `instr_count` | 固件发出的 NPU 指令总数 |
| `max_diff` | 与 golden reference 的最大数值差异 |

优化候选必须同时满足正确性门槛和目标指标改善；只减少指令但破坏输出的候选不能部署。

## 结果和可复现性

每轮结果保存到 `jimu-dse/results/run-*/`，包括：

- `candidate_*.c` 和 `candidate_best.c`；
- `diff_*.patch`；
- `p*_probe.json` 和 `val_*.json`；
- `micro_op_dag`、`dram_clusters` 以及 Agent 输入输出。

流程不依赖 Git，基线使用文件复制管理，因此可以在导出目录中复现。重新运行同一配置时，应使用相同的 `NATIVE_DIM`、hidden size、seq_len、DRAM 布局宏和 Agent 参数。

运行目录是本地证据，默认不提交到源码分支。需要长期保留的结论应记录配置、日期和 SHA256，并整理到 `jimu-dse/docs/reports/`。
