# NPU 固件闭环优化——运行指南

> 文档入口见 [`README.zh.md`](README.zh.md)，当前验证状态和功能边界见 [`project-status.zh.md`](project-status.zh.md)。

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

## 优化目标

| 目标 | 名称 | Dim | Hidden | 技能 | 主要指标 |
|------|------|-----|--------|------|----------|
| **G1** | `dram-optimization` | 2 | 4 | `vrf-cache` | `total_bytes` |
| **G2** | `compute-optimization` | 4 | 4 | `dim-optimize` | `mv_mul_count` |
| **G3** | `combined` | 4 | 4 | `dim-optimize` + `vrf-cache` | `mv_mul_count` |

### G1：DRAM 优化

固定 dim=2，以 DRAM 字节数为主指标。当前 DAG-PR6 会生成真实 seq2/seq6 DAG，并按验证范围补齐 dim2/dim4、hidden4/hidden8 的证明矩阵。Agent 必须先完成 eligible L1，再使用已经通过跨配置 VRF 分配证明的 L2 宏；L3 权重驻留当前禁止。

DAG 证据门禁默认开启。Agent 必须声明一个 `macro_candidates.json` 中当前 eligible、allocation-proven、cross-config-proven 的宏；没有 eligible 宏时才允许声明 primitive candidate。闭环核对精确 Tensor/address 的结构变化，以及 seq2/seq6 实测流量方向。
诊断时可以关闭拒绝，但仍会保留 `dag_diff_N.json` 和 `dag_diff_N.md`：

```bash
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --agent opencode \
  --dag-evidence-gate off

# 等价环境变量
JIMU_DAG_EVIDENCE_GATE=off bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --agent opencode
```

### G2：计算效率

将固件从多 tile 投影重构为单 tile 投影，使 `NATIVE_DIM` 与 `hidden_size` 匹配。每次投影的 `MV_MUL` 从 4 次降为 1 次，seq=6 时的权重 tile 加载（`M_RD_DRAM`）从 144 次降为 36 次。

### G3：组合优化

在 dim=4 下同时应用两种变换：先使用 `dim-optimize` 重构为单 tile，再使用 `vrf-cache` 消除剩余 DRAM 保存-加载往返。

## 可选依赖

```bash
sudo apt install graphviz       # 将 DAG .dot 渲染为 SVG
pip install amaranth            # 完整 HDL 验证
```

固件目标为 RV64IM，使用 `riscv64-unknown-elf-gcc` 编译。每次 probe 配置时，闭环流程会自动重新构建正确 DRAM 布局的 ELF。

## 标准基线与续跑起点

`jimu-dse/baseline/bert_layer.c` 是已知正确、未优化的标准固件。新运行默认从
该文件开始，优化结果不得回写覆盖标准基线。

如果要接在以前的结果后继续优化，使用 `--start-from`：

```bash
# 从某次运行的 candidate_best.c 继续
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization \
  --start-from jimu-dse/results/run-<timestamp>

# 从指定迭代继续
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization \
  --start-from jimu-dse/results/run-<timestamp>/candidate_4.c
```

每次运行都会把所选起点复制为 `optimization_baseline.c`。本次运行的全部指标和
diff 都以该快照为基准；运行结束后，工作区中的固件恢复为未优化标准基线。
`run_manifest.json` 会同时记录标准基线和续跑起点的路径、类型及 SHA256。

`--resume <run-dir>` 保留为兼容写法，等价于选择该目录中的
`candidate_best.c`。

## 正确性范围与常用开关

```bash
# 默认：dim2 + dim4 六组 BERT 配置
--validation-dim all

# 调试单个 DIM；不能作为跨 DIM 发布验证
--validation-dim dim2
--validation-dim dim4

# 使用目标本身的 DIM
--validation-dim goal

# 显式控制 seq6 指令回退门禁
--instruction-gate on
--instruction-regression-limit 0.10
--instruction-regression-limit off

# DAG 门禁仅在诊断时关闭
--dag-evidence-gate off
```

`JIMU_VALIDATION_DIM`、`JIMU_INSTR_GATE`、`JIMU_INSTR_REGRESSION_LIMIT` 和 `JIMU_DAG_EVIDENCE_GATE` 提供对应的环境变量写法。完整参数以脚本 `--help` 为准：

```bash
bash jimu-dse/scripts/npu_closed_loop.sh --help
```

## 验证源码或某次候选

候选验证应使用临时替换和退出恢复，不能覆盖规范基线：

```bash
candidate=jimu-dse/results/run-<timestamp>/candidate_best.c
backup=$(mktemp --suffix=.bert_layer.c)
cp firmware/bert/bert_layer.c "$backup"
trap 'cp "$backup" firmware/bert/bert_layer.c; rm -f "$backup"' EXIT INT TERM
cp "$candidate" firmware/bert/bert_layer.c

make kernels
python3 -m pytest tests --ignore=tests/integration -q
python3 -m pytest tests/integration/test_bert_e2e.py -q -rs
```

六组 E2E 全部通过才代表 `--validation-dim all` 的完整软件验证。若未安装 Amaranth，HDL Round 2/3 仍可能跳过，应单独说明。

## 清理工作区

安全清理脚本默认只预览，保留 `venv/` 和全部 `results/run-*`：

```bash
# 查看将删除的生成物
bash jimu-dse/scripts/clean_workspace.sh

# 删除构建目录、ELF/对象、缓存和本地备份
bash jimu-dse/scripts/clean_workspace.sh --apply

# 额外删除一个明确的临时运行；不能传 results 根目录
bash jimu-dse/scripts/clean_workspace.sh --apply \
  --run-dir jimu-dse/results/run-<timestamp>
```

清理后运行测试前需要重新执行 `make kernels`。重要运行结论应先提升到 `jimu-dse/docs/reports/`，不要把 `candidate_best.c` 当作规范基线。
