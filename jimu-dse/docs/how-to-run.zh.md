# NPU 固件闭环优化——运行指南

> **目的**：通过 DAG 自动引导 rv-npu 固件优化。流程会探测 DRAM 流量、生成微操作 DAG、调用 AI Agent 识别 VRF cache 和保存-加载对消除机会，验证正确性，并重复执行直到收敛。
>
> **不依赖 Git**：基线管理使用文件复制，因此适用于导出目录、浅克隆或没有 `.git` 的目录。

## 快速开始

### AdderBoard 目标

AdderBoard 使用独立的闭环 runner，因此不会改变原有 BERT CLI 和 baseline
行为：

```bash
# 仅测量的一轮闭环；候选仍须通过正确性门禁
bash jimu-dse/scripts/adderboard_closed_loop.sh --model 130p --agent none
bash jimu-dse/scripts/adderboard_closed_loop.sh --model 140p --agent none

# 让已安装的 Agent 只修改所选 AdderBoard 固件
bash jimu-dse/scripts/adderboard_closed_loop.sh --model 130p --agent opencode
bash jimu-dse/scripts/adderboard_closed_loop.sh --model 140p --agent opencode

# 对每个候选运行完整验证套件
bash jimu-dse/scripts/adderboard_closed_loop.sh --model 140p \
  --agent opencode --validation full --agent-timeout 900
```

目标配置位于 `jimu-dse/goals/adderboard/`。每次运行都会创建独立的
`jimu-dse/results/run-*/adderboard-<model>/` 目录，并保存 Agent 提示词、
候选固件、diff、构建和验证日志、metrics、probe 数据，以及指令 DAG、
微操作 DAG、算子图和 DRAM cluster 图。

无论运行成功、候选失败还是进程收到中断信号，runner 都会把所选固件
恢复为启动前的内容。候选只有先通过构建和正确性验证，才能按主要性能
指标排序并更新 `candidate_best.c`。

Agent 输出会实时显示在终端，并同步写入 `agent_output.log`。超过
`--agent-timeout` 指定的秒数后，runner 会终止 Agent 及其子进程、拒绝
候选并恢复固件。OpenCode 使用当前版本支持的 `--auto` 参数，prompt
通过附件传入，不再作为超长命令行参数传递。

可使用下面的命令测试拒绝和恢复路径：

```bash
bash jimu-dse/scripts/adderboard_closed_loop.sh \
  --model 130p --agent none --iterations 1 --inject-build-failure
```

该选项会在本轮结果目录中生成一个故意无法编译的候选。runner 应拒绝
该候选，保留 build/validation 日志和 diff，不更新 `candidate_best.c`，
并恢复工作区。

### 1. 安装依赖

```bash
sudo apt install -y build-essential cmake python3 python3-pip python3-venv gcc-riscv64-unknown-elf
python -m venv venv_jimu
source venv_jimu/bin/activate
pip install numpy pyelftools pytest
```

### 2. 构建产物

```bash
make kernels
CC=riscv64-unknown-elf-gcc \
  NATIVE_DIM=2 SEQ_LEN=2 \
  _HIDDEN_SIZE=4 _PROJ_BASE=12 _MAT_SIZE=16 _STRIDE=20 _NUM_TILES=2 \
  _LN1_GAMMA=132 _LN1_BETA=140 _LN2_GAMMA=148 _LN2_BETA=156 \
  _SCRATCH=1280 NUM_HEAD=2 \
  make -C firmware BUILD_DIR=build_dim2 all
```

### 3. 运行闭环

```bash
# G1：dim=2 的 DRAM 优化（VRF cache）
make opencode
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization --agent opencode

# G2：dim=4 的计算效率优化（单 tile 投影）
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode

# G3：组合优化（dim=4 + VRF cache）
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode
```

使用默认 pi Agent：

```bash
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh
```

指定模型：

```bash
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh \
    --goal combined --agent opencode --model opencode/deepseek-v4-flash-free
```

### 4. 查看结果

```bash
ls jimu-dse/results/run-*/candidate_best.c
ls jimu-dse/results/run-*/dag_iter1/
cat jimu-dse/results/run-*/diff_1.patch
```

### 5. 使用不同目标和工作负载运行

闭环流程与具体工作负载无关。默认目标为 BERT，也可以通过 `--workload adder` 选择 Adderboard。

```bash
# G1：在默认 BERT 工作负载上进行 DRAM 优化
bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization

# G2：在 BERT 上进行计算效率优化
bash jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode

# G3：在 BERT 上执行组合优化
bash jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode

# M0/G1：在 Adderboard 上执行 G1 VRF 缓存优化
bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization --workload adder --agent opencode
```

## 优化目标

三个标准目标分别定义在 `jimu-dse/goals/<name>/goal.sh` 中；通过 `--workload` 可以将目标应用于不同工作负载。

| 目标 | 名称 | Dim（BERT） | Dim（Adder） | 技能 | 主要指标 |
|------|------|------------|-------------|------|----------|
| **G1** | `dram-optimization` | 2 | 4 | `vrf-cache` | `total_bytes`（DRAM） |
| **G2** | `compute-optimization` | 4 | 4 | `dim-optimize` | `test_pass` |
| **G3** | `combined` | 4 | 4 | `dim-optimize` + `vrf-cache` | `test_pass` |

### G1：DRAM 优化

在固定维度下使用 VRF 缓存消除保存-加载往返。对 BERT，目标包括 K、V、Q、Z、SO、LN 和 GELU 中间结果；对 Adder，重点是 context 和 score 缓存。

### G2：计算效率

将固件从多 tile 投影重构为单 tile 投影，使 `NATIVE_DIM` 与 `hidden_size` 匹配（特指 BERT）。每次投影的 `MV_MUL` 从 4 次降为 1 次，seq=6 时的权重 tile 加载（`M_RD_DRAM`）从 144 次降为 36 次。

### G3：组合优化

同时应用两种变换：先使用 `dim-optimize` 重构为单 tile，再使用 `vrf-cache` 消除剩余 DRAM 保存-加载往返。

## 可选依赖

```bash
sudo apt install graphviz       # 将 DAG .dot 渲染为 SVG
pip install amaranth            # 完整 HDL 验证
```

固件目标为 RV64IM，使用 `riscv64-unknown-elf-gcc` 编译。每次 probe 配置时，闭环流程会自动重新构建正确 DRAM 布局的 ELF。
