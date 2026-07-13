> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU 闭环固件优化——如何运行

> ** 目标**:rv-npu的自动DAG制导固件优化。
> 循环探测器可以进行 DRAM 流量,生成微op DAG 图表,引用
> 识别优化机会的AI代理( VRF 缓存、 保存载入)
> 双消除),验证正确性,并重复到趋同.
>
> ** 无git依赖**: 所有基线管理都使用文件副本.
> 用于导出代码,浅克隆,或任何没有.git的文件系统.

---

## 快速启动( 从导出文件夹)

这些步骤将您从新导出到完成优化运行。
预期时间:~3分钟(第一次运行包括内核+固件构建).

### 1. 安装依赖关系

```bash
# System packages
sudo apt install -y build-essential cmake python3 python3-pip python3-venv \
    gcc-riscv64-unknown-elf

# Python packages
python -m venv venv_jimu
source venv_jimu/bin/activate
pip install numpy pyelftools pytest
```

### 2. 构建艺术

```bash
# Build C kernel library (libnpukernels.so)
make kernels

# Build RISC-V firmware for the probe
CC=riscv64-unknown-elf-gcc \
  NATIVE_DIM=2 SEQ_LEN=2 \
  _HIDDEN_SIZE=4 _PROJ_BASE=12 _MAT_SIZE=16 _STRIDE=20 _NUM_TILES=2 \
  _LN1_GAMMA=132 _LN1_BETA=140 _LN2_GAMMA=148 _LN2_BETA=156 \
  _SCRATCH=1280 NUM_HEAD=2 \
  make -C firmware BUILD_DIR=build_dim2 all
```

### 3. 运行闭环

** 选择一个进球并运行:**

```bash
# G1: DRAM optimization at dim=2 (VRF cache)
make opencode
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization --agent opencode

# G2: Compute efficiency at dim=4 (single-tile projections)
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode

# G3: Combined (dim=4 + VRF cache)
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode
```

** 带pi( 默认代理) : **
```bash
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh
```

** 采用自定义模式:**
```bash
JIMU_MAX_ITER=3 bash jimu-dse/scripts/npu_closed_loop.sh \
    --goal combined --agent opencode --model opencode/deepseek-v4-flash-free
```

预期输出 (pi):
```
[START] Starting from baseline — /* NPU — BERT Encoder Layer Firmware (All Features)
--- Iteration 1 ---
[PROBE] seq=2...  [BUILD] seq=2 OK
[PROBE] seq=6...  [BUILD] seq=6 OK
[PROBE] Generating micro-op DAG for agent analysis...
  seq=2: 2144B  seq=6: 6240B
  Baseline for this run set to 6240B
[AGENT] Invoking pi (timeout: 600s)...
...
[VALIDATE] DRAM: 5856B (saved 384B vs run-start 6240B)
...
===== Done =====
Baseline:   6240B
Best:       5856B
Improvement: 384B (6.2%)
To resume:  ./jimu-dse/scripts/npu_closed_loop.sh --resume .../run-...
```

### 4. 检查结果

```bash
# View the optimized candidate
ls jimu-dse/results/run-*/candidate_best.c

# View the DAG graphs for audit
ls jimu-dse/results/run-*/dag_iter1/

# View diff against baseline
cat jimu-dse/results/run-*/diff_1.patch
```

### 5. 以不同的目标运行

```bash
# G1: DRAM optimization at dim=2 (VRF cache)
bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization

# G2: Compute efficiency at dim=4 (single-tile projections)
bash jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode

# G3: Combined (dim=4 + VRF cache)
bash jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode
```

---

## 优化目标

管道支持三个优化目标,

|目标|名称|阴暗|隐藏|技能|小学|
|------|------|-----|--------|-------|----------------|
|** G1** 国家|津巴布韦| 2 | 4 |津巴布韦|ZPROT000XZ (中文(简体) ).|
|** G2 **|津巴布韦| 4 | 4 |津巴布韦|津巴布韦|
|** G3** 国家|津巴布韦| 4 | 4 |ZPROT0001Z + ZPROT0001Z + ZPROT0001Z + ZPROT00001Z|津巴布韦|

### G1: DRAM 优化

通过应用 VRF 缓存来消除,在固定的dim=2 时减少 DRAM 字节
用于K,V,Q,Z,SO,LN,GELU中间体的保存载荷圆盘.

### G2:计算效率

将固件从多瓦调整为单瓦预测
正在增加 Native DIM 以匹配隐藏的 大小 。 减少 MV  MUL 操作
每个投影从 4 到 1, 和重瓦负载( M  RD  DRAM) 从
144至36 obs at excess=6。

### G3: 合并

两种变换在暗数=4. 技术是ZPROT000Z
应用先(调整单片结构),然后是 QQZPROT000XZ
消除剩余的 DRAM 保存的圆路 。

---

## 先决条件

### 系统依赖( apt)

```bash
sudo apt install -y \
    build-essential cmake \
    python3 python3-pip python3-venv \
    gcc-riscv64-unknown-elf \
    graphviz           # optional, for DOT → SVG rendering of DAG graphs
```

### Python 附属物( pip)

```bash
# Required — core pipeline
pip install numpy pyelftools pytest

# Optional — full 4-round validation with Amaranth HDL simulation
pip install amaranth

# Optional — for rendering DAG .dot files to SVG
# sudo apt install graphviz   (see above)
```

### RISC-V 交叉编译器

固件目标RV64IM,

```bash
# Verify:
riscv64-unknown-elf-gcc --version
# Expected: gcc 10.x or later, target: riscv64-unknown-elf
```

在ZPROT000Z:ZPROT0001Z上,

### AI 代理人

循环支持两个AI代理. 只需要安装一个。

#### 备选案文1:pi(违约)

```bash
npm install -g @earendil-works/pi-coding-agent
pi --version
# Expected: 0.79.x or later
```

#### 备选案文2:OpenCode(备选案文)

```bash
# Install OpenCode CLI
npm install -g @openai-code/cli

# Configure agent skills and permissions (run once):
make opencode

# This creates:
#   .opencode/skills/dag-analyze/SKILL.md
#   .opencode/skills/vrf-cache/SKILL.md
#   opencode.json (permissions for file write, read, pytest)
```

这些技能来自ZZPROT000Z-单源
真实的。 如果技能更新, 请再次运行 QQZPROT000QZ 。

如果没有安装代理, 循环复制未修改的固件
作为候选者 并运行 作为唯一的测量管道。

---

## 快速启动

### 1. 构建 C 核心库

```bash
make kernels
```

这构建了模拟器使用的 QQZPROT000MXZ 。

### 2. 运行新优化循环

```bash
bash jimu-dse/scripts/npu_closed_loop.sh
```

运行5次迭代(可通过 QQZPROT000XZ 配置) :

|阶段|发生什么事|
|-------|-------------|
|** 标准**|兹普罗特0001兹|
|页:1|构建固件, 运行仿真器, 测量 DRAM 流量|
|** 以后各段=6**|下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至下至|
|** 加拿大**|报告 DRAM 比率 QZPROT000ZZ 和 分组分解|
|** 委员会**|如果改进 < 15%/运行启动基线,则提前停止|
|页:1|具有DAG + DRAM + 技能的 Invokes pi, pi补丁 bert layer.c|
|** 估价**|与候选人重建固件, 重新测量 DRAM|
|(单位:千美元)|生成用于审计的DAG图表优化后|
|** 就业**|保存候选人, 使用优化代码从迭代 2 重复|

### 3. 从上次运行中恢复

```bash
bash jimu-dse/scripts/npu_closed_loop.sh --resume jimu-dse/results/run-20260622-142411/
```

运行输出打印一个可随时使用的恢复命令在结尾.

### 4. 运行验证测试套件

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

---

## 运行输出结构

每次运行都会创建时间标记的目录 :

```
jimu-dse/results/run-<YYYYMMDD>-<HHMMSS>-<PID>/
├── candidate_best.c       ← Best optimized firmware (copy for --resume)
├── candidate_1.c          ← Iteration 1 candidate
├── candidate_2.c          ← Iteration 2 candidate
├── dag_agent/             ← DAG graphs for the agent prompt
│   ├── micro_op_dag.dot
│   └── micro_op_dag.txt
├── dag_iter1/             ← DAG graphs for iteration 1 (audit trail)
│   ├── micro_op_dag.dot
│   ├── micro_op_dag.txt
│   ├── instr_dag.dot
│   ├── op_graph.dot
│   ├── sym_graph.dot
│   └── sym_graph_instantiated.dot
├── dag_iter2/             ← DAG graphs for iteration 2
├── prompt_1.txt           ← Agent prompt for iteration 1
├── prompt_2.txt           ← Agent prompt for iteration 2
├── diff_1.patch           ← Diff against baseline (iteration 1)
├── diff_2.patch           ← Diff against baseline (iteration 2)
├── p2_probe.json          ← DRAM probe result for seq=2
├── p6_probe.json          ← DRAM probe result for seq=6
└── val_1.json             ← Validation result for iteration 1
```

将 DAG 视图为 SVG( 需要图解):

```bash
dot -Tsvg jimu-dse/results/run-*/dag_iter1/micro_op_dag.dot -o /tmp/dag.svg
```

---

## 配置

|信封变量|默认|说明|
|-------------|---------|-------------|
|津巴布韦| 5 |每个运行的最大优化迭代|
|津巴布韦| 0.15 |汇合阈值(违反基线)。 改进时停止 < 15%|
|津巴布韦| 600 |每个代理引用( pi 或 opencode) 的超时秒数|
|津巴布韦|津巴布韦|QQZPROT000XZ格式的 OpenCode模型|
|津巴布韦|津巴布韦|RISC-V 交叉编译器|

### CLI 旗帜

|旗帜|默认|说明|
|------|---------|-------------|
|津巴布韦|津巴布韦|优化目标:ZPROT000XZ,ZPROT00001Z,ZPROT00002XZ|
|津巴布韦|毕|要使用的代理 : QZPROT000XZ 或 ZPROT00001Z|
|津巴布韦|津巴布韦|OpenCode 模型 (overrides env var) (英语).|
|津巴布韦| — |从上一个运行目录中恢复|

实例:

```bash
# G1: DRAM optimization at dim=2
JIMU_MAX_ITER=10 JIMU_THRESHOLD=0.05 bash jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization

# G2: Compute efficiency at dim=4 with OpenCode
make opencode
JIMU_MAX_ITER=5 bash jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode

# G3: Combined with custom model and resume
bash jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode --model opencode/deepseek-v4-flash-free
```

---

## 优化目标

管道测量两种配置以检测DRAM流量模式:

|测试|下列语|它所揭示的|
|------|---------|-----------------|
|平方厘米2| 2 |最小中间保存的基线|
|dim2-seq6 数字| 6 |3x DRAM 缩放显示 VRF 溢出/ 保存负载对|

典型的未优化基线 DRAM :

|度量衡|下级=2|下级=6|
|--------|-------|-------|
|V RD DRAM 行动| ~180 | ~456 |
|V WR DRAM 行动| ~60 | ~156 |
|M RD DRAM 行动|144(不变)|144(不变)|
|总字节| ~2,144 | ~6,240 |

VRF缓存优化后,典型结果:

|重复|以下=6个|节余与基线|
|-----------|-----------|-------------------|
|0(基线)|6 240B (韩语)| — |
|1 (XZPROT000XZ缓存)|5,856B (韩语).| 6.2% |
|2 (XZPROT000XZ缓存)|4 704B (中文(简体) ).| 24.6% |
|3 (LN 抓伤)|3 936个B| 36.9% |
|4(X缓存)|3,744B (中文(简体) ).| 40.0% |

---

## 基线文件

未优化的参考固件位于QQZPROT000XQZ.
这是任何 VRF 缓存之前的 原始固件的 承诺副本
优化。 为更新基准:

```bash
# After a successful optimization, promote the best candidate:
cp jimu-dse/results/run-<timestamp>/candidate_best.c jimu-dse/baseline/bert_layer.c
```

---

## 手动调试

### 每名操作者诊断

```bash
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s --no-header 2>&1 | grep "max_diff"
```

仅显示最后输出比较( Q、 K、 V 中间件为
优化自由且未验证).

### 生成全部 DAG 图形

```bash
python3 jimu-dse/scripts/generate_all_graphs.py
```

(原始内容存档于2019-09-31). Official in QQZPROT000 XZ.

---

## 技能库

|技能|文件|说明|
|-------|------|-------------|
|dag 分析|津巴布韦|读取 DAG 以识别保存的负载对|
|vrf 缓存|津巴布韦|用芯片 VRF 缓存替换 DRAM 圆路|
|自我验证|津巴布韦|固件修改后的自我验证|

---

## 文件布局

```
.
├── firmware/bert/bert_layer.c    ← Target firmware (only file the agent modifies)
├── jimu-dse/
│   ├── baseline/bert_layer.c     ← Unoptimized reference (committed, no git needed)
│   ├── goals/                    ← Optimization goal configurations
│   │   ├── dram-optimization/    ← G1: VRF cache at dim=2
│   │   ├── compute-optimization/ ← G2: Single-tile at dim=4
│   │   └── combined/             ← G3: Both G1 + G2 at dim=4
│   ├── scripts/
│   │   ├── npu_closed_loop.sh    ← Main pipeline driver
│   │   └── visualize_graph.py    ← DAG graph generator
│   ├── docs/
│   │   ├── how-to-run.md         ← This file
│   │   ├── npu-closed-loop-design.md ← Full architecture document
│   │   └── skills/isa/           ← Optimization skills for the agent
│   │       ├── dag-analyze.md    ← Read DAG to find save-load pairs
│   │       ├── vrf-cache.md      ← VRF cache (G1 skill)
│   │       ├── dim-optimize.md   ← Dim efficiency (G2/G3 skill)
│   │       └── self-verify.md    ← Post-optimization verification
│   └── results/                  ← Run outputs (gitignored)
├── emulator/                     ← NPU behavioral model (do NOT modify)
├── iss/                          ← RISC-V ISS (MiniRV64)
├── tests/integration/            ← BERT E2E validation tests
└── _build/                       ← C kernel library (make kernels)
```

---

## 解决问题

|问题|可能的原因是|修补|
|---------|-------------|-----|
|津巴布韦|未安装交叉编译器|津巴布韦|
|津巴布韦|错误的工作目录|从 repo 根中运行|
|津巴布韦|未建立核心库|津巴布韦|
|津巴布韦|固件构建失败|请检查access-date=中的日期值 (帮助) ZPROT000X env var, 手动运行 ZPROT0001Z|
|津巴布韦|未安装代理|津巴布韦|
|pi 输出摘要而不是代码|提示丢失的写指令|循环提示包括写入指令; 检查 proogle  .txt|
|输出时的 ZPROT000Z|优化引入的数字错误|还原候选人, 请检查 diff %. patch 不正确的更改|
|津巴布韦|未安装 Graphviz|XZPROT000XZ( 可选, 用于 SVG 渲染)|
