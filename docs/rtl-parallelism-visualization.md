# RTL 并行工作情况可视化

`scripts/visualize_rtl_parallelism.py` 是面向项目报告和答辩 PPT 的独立制图工具。它读取 RTL 优化闭环已经保存的 `timing-schedule.json` 与 `run-summary.json`，不会启动编译、仿真或新的优化，也不会修改闭环产物和工作树代码。

## 快速使用

在项目根目录手动执行：

```bash
python scripts/visualize_rtl_parallelism.py
```

默认读取 `jimu-dse/results` 中最近 3 次状态为 `completed` 的 Verilator RTL 闭环，并输出到 `_out/rtl-parallelism`。

常用命令：

```bash
# 最近 5 次已完成的 RTL 闭环
python scripts/visualize_rtl_parallelism.py --latest 5

# 明确选择一个或多个运行；参数可重复
python scripts/visualize_rtl_parallelism.py \
  --run run-20260822-200156-21529 \
  --run run-20260822-161925-7452 \
  -o _out/ppt-rtl

# 只生成 baseline 与 best 的并行工作对照图
python scripts/visualize_rtl_parallelism.py --view comparison

# 自动发现时也纳入被中断或尚未完成的运行
python scripts/visualize_rtl_parallelism.py --include-incomplete

# 将未晋升或负优化的候选轮次与 baseline 对比
python scripts/visualize_rtl_parallelism.py \
  --run run-20260817-162615-1077 \
  --candidate 1 \
  -o _out/rtl-parallelism-negative-case

# 只生成负优化归因页和差异聚焦图
python scripts/visualize_rtl_parallelism.py \
  --run run-20260817-162615-1077 \
  --candidate 1 \
  --view diff \
  -o _out/rtl-parallelism-negative-case

# 生成每组修改指令前后各 24 cycles 的局部时间轴
python scripts/visualize_rtl_parallelism.py \
  --run run-20260817-162615-1077 \
  --candidate 1 \
  --view windows \
  --window-context-cycles 24 \
  -o _out/rtl-parallelism-negative-case

# Attention 读取窗口以宏观 Attention 起点 t=0 对齐
python scripts/visualize_rtl_parallelism.py \
  --run run-20260817-162615-1077 \
  --candidate 1 \
  --view windows \
  --window-alignment attention-start \
  -o _out/rtl-parallelism-attention-aligned
```

在 PowerShell 中，多行命令可改为一行，或者使用反引号替代上例中的反斜杠续行。

## 输出内容

所有图片均为 1920×1080、16:9、白底 SVG，可直接插入 PowerPoint，放大后仍保持清晰：

```text
_out/rtl-parallelism/
├── recent-rtl-runs-summary.svg
├── visualization-manifest.json
└── run-YYYYMMDD-HHMMSS-PID/
    ├── baseline-vs-best.svg
    ├── baseline-detail.svg
    └── best-detail.svg
```

- `recent-rtl-runs-summary.svg`：最近几次闭环的 baseline/best 绝对周期横向比较。
- `baseline-vs-best.svg`：baseline 与最优结果分别展开到完整横轴，让短事件和并行区间在一页 PPT 中保持可读；上方周期数和提升率用于比较绝对性能。
- `baseline-detail.svg`、`best-detail.svg`：分别按自身周期范围展开，适合观察各硬件单元的并行工作细节。
- `visualization-manifest.json`：记录选中的闭环、周期数、最优迭代以及原始调度文件路径，便于报告追溯。

使用 `--candidate N` 时，工具不会依赖 `best_iteration`，而是读取指定的 `iteration-N/timing-schedule.json`，并额外生成：

- `baseline-vs-candidate.svg`：分别展开 baseline 和候选的资源并行时间轴。
- `baseline-vs-candidate-diff.svg`：按原始指令序号对齐两侧调度，仅显示操作、目标单元、持续周期、存储或资源映射发生变化的命令，并汇总这些变化导致的并行周期差异。开始/结束周期的整体平移不会被误算成代码差异。
- `candidate-detail.svg`：候选调度的详细时间轴。
- `parallelism-regression.svg`：当候选周期高于 baseline 时生成，把串行工作量、并行隐藏周期和最终 makespan 放在同一个可加和关系中，适合解释“访存减少但并行性下降导致负优化”的案例。

`--view windows` 会把相邻的修改指令合并成局部窗口，在 `changed-windows/` 中为每个窗口生成一张 16:9 SVG。默认取 baseline 与 candidate 修改点上下文周期范围的并集作为共享绝对时间轴，因此上下图的同一 cycle 刻度和事件位置严格对齐。使用 `--window-alignment attention-start` 后，Attention 读取窗口会定位包围该组指令的 Q 投影配置起点，以红色虚线标注 `Attention start (t=0)`，再用相对周期对齐上下图；面板右上角仍保留各自的绝对周期范围用于追溯。淡红色竖区和红色外框标出修改事件。事件色块参考 `dram_clusters.svg` 的信息层级：第一行以粗体显示语义任务名，第二行显示“操作码＋周期数”；空间稍小时依次退化为“任务名＋周期数”、短操作码＋周期数或单行短标签，过窄时不显示文字且不会压缩字号。`changed-windows-manifest.json` 记录每个窗口对应的原始指令范围、对齐方式、Attention 起点及共享相对周期范围。

## 图的阅读方式

图中每一横行对应一个 RTL 资源：Load、Store、MVU、Vector 和 Control。彩色区间表示该资源正在工作；淡蓝色竖向背景表示至少两个资源同时工作。`Parallel units` 行用浅灰、蓝色和深蓝色分别表示 1、2、3 个及以上资源并行。

关键事件会在对应资源行顶部显示为醒目的红色标记，并额外汇总到独立的 `Critical path` 红色轨道；轨道标签同时显示关键事件数量，因此既能看出关键事件落在哪类资源上，也能连续观察整条关键路径。

对照图中的 baseline 和 best 分别使用自己的周期横轴并铺满可用宽度，重点是观察各自内部的并行关系。绝对周期差异由页面上方的 makespan、提升率和近期闭环汇总图表达，避免最优版本因共用较长的 baseline 横轴而被压缩在左侧。

## 数据范围与限制

- 仅支持 `backend=verilator-rtl` 且保存了 `events` 的调度产物。
- 有 `best_iteration` 时读取对应迭代；没有晋升候选或对应文件缺失时，用 baseline 作为 best 并明确标注 `无晋升候选`。
- 自动发现默认忽略中断运行；使用 `--include-incomplete` 可纳入这类结果。显式 `--run` 不受此过滤限制。
- 时间轴来自闭环的 RTL 时序模型，并以像素列聚合大量事件，适合报告展示与优化对比；它不是逐信号的波形查看器，也不替代 VCD/GTKWave 调试。
