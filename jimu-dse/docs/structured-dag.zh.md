# 结构化 DAG 与跨配置证明

结构化 DAG 是事件 Trace 的只读分析层：它把指令折叠成微操作，建立 def-use 依赖，恢复 Tensor/Phase，并生成可由独立门禁验证的候选和 VRF 分配证明。它不会改变模拟器或固件执行语义。

当前实现包含 DAG-PR1～PR6。单配置 schema 为 `jimu-npu-micro-op-dag 1.1.0`，多序列 schema 为 `jimu-npu-multiseq-dag 2.0.0`，跨配置分配证明 schema 为 `jimu-npu-cross-config-allocation-proof 1.0.0`。

## 证据层次

```text
事件 Trace
  └─ 单配置 DAG：微操作、边、Tensor、Phase、生命周期、primitive 候选
       ├─ seq2 + seq6：跨序列候选族和 L1 宏
       └─ dim/hidden/seq 验证矩阵：L2 VRF 分配证明与 L3 阻止理由
            └─ before/after DAG + 实测指标：独立接受门禁
```

Agent 的文字解释不是证据。JSON/JSONL 中的原始 `defs`、`uses`、地址、事件序号、ELF 哈希和门禁结果才是权威输入。

## 单配置文件

| 文件 | 用途 |
|---|---|
| `run_metadata.json` | Schema、固件配置、ELF SHA256、数量和标注方法 |
| `micro_ops.jsonl` | 稳定微操作 ID、kind、Phase、defs/uses 和 Tensor 标注 |
| `edges.jsonl` | RAW/WAR/WAW 等生产者—消费者依赖 |
| `tensors.json` | DRAM Tensor 切片及生产者、消费者 |
| `phases.json` | Phase 层次、流量、FLOPs 和算术强度 |
| `patterns.json` | 按语义类型压缩的重复 Phase |
| `lifetimes.json` | 当前 VRF/MRF 等资源的定义—使用生命周期 |
| `candidates.json` | primitive 同地址 DRAM 往返、证明和拒绝原因 |
| `candidate_summary.md` | primitive 候选的紧凑索引 |
| `summary.md` | 单配置概览 |

`micro_op_dag.txt`、`dram_clusters.txt` 和 DOT/SVG 仍可用于人工取证，但 Agent 不应从图形位置推断依赖。

## 多序列与跨配置文件

闭环在 `dag_agent/` 或冻结的 `dag_before_iterN/` 根目录生成：

| 文件 | 用途 |
|---|---|
| `multiseq_metadata.json` | seq 输入、证明配置、ELF 哈希和矩阵完整性 |
| `loop_invariants.json` | seq2/seq6 复用族、收益投影和实现就绪状态 |
| `candidate_evidence.jsonl` | primitive 与宏的紧凑证据索引 |
| `multiseq_summary.md` | 面向 Agent 的跨序列摘要 |
| `macro_candidates.json` | L1/L2/L3 宏、资格、级别和预期资源范围 |
| `macro_candidate_summary.md` | 宏优先级和阻止原因 |
| `allocation_proof.json` | 每个必需配置的精确 bank/row、生命周期、冲突检查和摘要哈希 |
| `allocation_summary.md` | 分配证明的紧凑入口；选定宏后再打开完整 JSON |

当前 BERT `--validation-dim all` 要求六个配置全部出现：

- `dim2-h4-head2-seq2/seq6`；
- `dim4-h4-head2-seq2/seq6`；
- `dim4-h8-head2-seq2/seq6`。

缺少任一 `required_config_id` 时，`validation_matrix_complete=false`，宏不得声明为跨配置已证明。

## 生成

生成一个具体 DAG：

```bash
python3 jimu-dse/scripts/visualize_graph.py \
  --phase micro --dim 2 --hidden 4 --seq-len 6 --num-head 2 \
  --no-render -o /tmp/jimu-dag-seq6
```

`visualize_graph.py` 的构建目录包含 dim、hidden 和 seq，避免不同配置复用错误 ELF。闭环会自动生成 seq2/seq6 和验证范围要求的额外证明 DAG，因此正常运行无需手工拼接：

```bash
JIMU_MAX_ITER=1 bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --agent opencode --validation-dim all
```

## 候选级别

| 级别 | 当前含义 | 状态 |
|---|---|---|
| L1 | 消除中间 Tensor 的 DRAM 保存—加载往返 | 可从宏或 primitive 选择 |
| L2 | 缓存循环不变量或序列输入 | 仅完整跨配置分配证明后可选 |
| L3 | 权重驻留、循环换序和片上部分和 | 当前阻止 |

L2 分配器对每个 DRAM 资源使用精确地址的首次到末次读取区间，检查只读性，并在 `MFU_INITIAL_VRF` 中按 `NATIVE_DIM` 对齐执行确定性 first-fit。它与所有已存在 VRF 生命周期做区间冲突检查，而不是只看当前空闲行。

L3 仍缺少循环交换、MRF clobber、逐位置部分和以及 FP16 运算顺序证明。`macro-dram-l3-weight-stationary` 的收益只能作为上界，不能由 Agent 实现。

## Agent 声明和门禁

源码必须且只能声明一个选择：

```c
// JIMU_DAG_MACRO: macro-dram-l1-attention
```

如果没有 eligible 宏，才允许 primitive 回退：

```c
// JIMU_DAG_CANDIDATE: candidate-dram-0007
```

门禁执行以下检查：

1. 声明在冻结的 before-DAG 中唯一且 eligible；
2. 宏同时满足 `allocation_proven=true`、`cross_config_proven=true` 和 `validation_matrix_complete=true`；
3. 必须先完成所有 eligible L1，才能选择 L2；L3 当前拒绝；
4. 每个宏成员的精确 Tensor/address DRAM 操作都减少，宏范围外不能偷偷减少；
5. DRAM 统计读取所有节点的 `uses/defs`，包括融合 `VV_BINOP` 等节点中的 bias/参数读取；
6. seq2 不回退、seq6 严格改善，并通过正确性和指令回退门禁。

每轮 Agent 输入复制为不可变 `dag_before_iterN/`。候选 probe 使用 metric-only，不能覆盖该证据。比较结果写入 `dag_diff_N.json` 和 `dag_diff_N.md`。

## 限制

- Tensor、position、tile 和 Phase 是基于 DRAM 布局的语义恢复；无法可靠恢复的 head 保持 `null`。
- 当前证明以精确地址为主，不宣称覆盖任意部分地址区间重叠。
- first-fit 证明说明当前已观测验证矩阵可分配，不等于形式化证明所有未来配置。
- `--dag-evidence-gate off` 只用于诊断；报告仍生成，但错误不会阻止候选。

当前实现边界和测试结果见[项目状态](project-status.zh.md)。
