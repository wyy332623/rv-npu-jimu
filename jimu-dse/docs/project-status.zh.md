# JIMU-DSE 当前项目状态

> 更新时间：2026-08-19。本文描述当前 v10 工作树的闭环能力；单次实验结果仍应以对应运行目录中的 manifest、DAG 和验证报告为准。

## 结论

项目已经从“让 Agent 直接阅读文本 DAG 并尝试修改固件”演进为受独立门禁约束的优化平台：基线、续跑起点、skill、验证范围和 DAG 证据都有机器可读记录。G1 DRAM 优化现已完成 L1 中间张量、L2 循环不变量/序列输入缓存、六个投影的分阶段 L3 权重驻留，以及单位向量片上精确合成。dim2-hidden4 的 seq2/seq6 实测达到信息传输硬下界 **608 B / 736 B**。

## 已验证基线

| 项目 | 当前值 |
|---|---|
| 规范基线 | `jimu-dse/baseline/bert_layer.c` |
| 当前固件 | `firmware/bert/bert_layer.c`，清理前与规范基线内容一致 |
| 达到硬下界的候选 | `jimu-dse/results/run-20260817-143300-22983/candidate_best.c` |
| 基线 SHA256 | `0957b9cb6a5dff4cf9f1b3dbf55ca882d94a428debde3d9da1aa4a7a2dacb664` |
| 非集成测试 | 2026-08-11：`56 passed` |
| BERT E2E/VRF | 2026-08-19：完整组合 `13 passed` |
| Skill 校验 | `verified 6 skills` |

BERT E2E 的六配置是：

- `dim2-hidden4-head2-seq2/seq6`；
- `dim4-hidden4-head2-seq2/seq6`；
- `dim4-hidden8-head2-seq2/seq6`。

构建目录是可再生成产物，工作区清理后需要先运行 `make kernels`，测试和闭环会按配置重新生成固件 ELF。

## 当前闭环能力

### 基线与续跑

- 新运行默认从未优化规范基线开始。
- `--start-from baseline` 可显式选择规范基线。
- `--start-from <run-dir|candidate.c>` 可从某次候选继续。
- 每次运行独立保存 `optimization_baseline.c`，不会把候选写回规范基线。
- `run_manifest.json` 记录规范基线、优化起点、Git 状态、参数、验证范围和 SHA256。

### 验证与门禁

- `--validation-dim dim2|dim4|all|goal` 控制正确性矩阵；默认 `all`。
- 指令数回退门禁和阈值可显式开关。
- DAG 证据门禁默认开启，可诊断性关闭，但关闭时仍生成报告。
- 公共约束禁止 Agent 修改测试、模拟器、ISS，或使用 `git stash/reset/checkout` 隐藏状态。
- 候选只有通过独立正确性、主指标、指令回退和 DAG 一致性检查后才可能被接受。

### Skill 平台

`jimu-dse/docs/skills/isa/*.md` 是唯一真源；`skillctl.py` 负责语义版本归档、同步、回滚和 SHA256 校验。OpenCode 显式接收全部有效 skill，PI 接收合并后的 `skills_bundle.md`。

| Skill | 版本 |
|---|---:|
| `common-constraints` | 1.0.0 |
| `dag-analyze` | 1.14.0 |
| `dim-optimize` | 1.0.0 |
| `inc-folding` | 1.0.0 |
| `self-verify` | 2.0.0 |
| `vrf-cache` | 2.10.0 |

每次运行的 `skills_manifest.json` 和 `run_manifest.json` 都记录实际传入的名称、版本和 SHA256。

## DAG-PR1～PR6 状态

| 阶段 | 已实现能力 |
|---|---|
| PR1 | 从事件 Trace 生成稳定微操作、def-use 边、Tensor、Phase 和生命周期 JSON/JSONL |
| PR2 | 生成聚类、重复模式、流量/FLOPs/算术强度摘要 |
| PR3 | 提取同地址 DRAM 往返候选，记录重定义、重叠生命周期和收益估算 |
| PR4 | 冻结迭代前 DAG，比较 before/after，并把声明与实测变化接入接受门禁 |
| PR5 | 合并真实 seq2/seq6 DAG，形成跨序列候选族、循环不变量和宏候选 |
| PR6 | 对完整验证矩阵执行确定性 VRF first-fit 分配，生成跨配置证明并实施 L1→L2 分阶段门禁 |

PR6 当前能对以下 L2 宏给出完整验证矩阵证明：

- `macro-dram-l2-loop-invariants`；
- `macro-dram-l2-sequence-input`。

六配置中最大的已证明 L2 分配占用出现在 `dim4-hidden8-seq6`，仍处于 MFU initial VRF 的容量约束内。门禁不仅识别独立 `DRAM_LOAD/STORE` 节点，也会统计融合算子 `uses/defs` 中的 DRAM 资源。

L3 已拆成 K/V、Q、SELF_OUTPUT、FFN_INTERMEDIATE、FFN_OUTPUT 五个确定性合同；每一段都证明循环交换、MRF 驻留、bank-13 部分和/双状态区以及 FP16 `tc` 累加顺序，并分别通过精确 Tensor/地址 DAG 差分门禁。最后的 `macro-dram-l2-unit-vector-synthesis` 使用 FP16 `0x0000/0x3c00` 和 lane 写掩码在片上构造单位向量，使最终候选达到 608 B / 736 B；其 `next_macro_contract.json` 为 `blocked-no-eligible-scope`，表示当前计分模型下已无剩余可接受 DRAM 宏。

## 当前边界与风险

- HDL Round 2/3 只有安装 Amaranth 时才执行；没有安装不等于 RTL 已验证。
- 当前分配证明覆盖精确地址和已记录生命周期，不宣称解决任意部分重叠地址。
- `results/run-*` 是本地证据，体积大且依赖环境，默认不提交；重要结论应提升为带日期的专题报告。
- 历史文档可能引用旧 schema、seq1 DAG 或旧上游目录，已经移动到 `archive/`。
- 工作树可能保留此前源码修改和历史 results 的 Git 删除记录；安全清理脚本不会替用户恢复或丢弃这些变更。

## 下一步建议

1. 用当前规范基线运行短轮次 G1，要求 Agent 先完成剩余 L1，再进入已证明 L2。
2. 对接受候选比较 seq2/seq6 的实测字节数、指令数和六配置 E2E，而不是只看 Agent 自述。
3. 为 L3 单独建立调度与数值顺序证明，证明完成前保持门禁阻止。
4. 在准备提交分支时，把源码、测试、skill、文档和历史结果清理拆成可审查的提交。

运行命令和产物解释见[运行指南](how-to-run.zh.md)与[结构化 DAG 文档](structured-dag.zh.md)。
