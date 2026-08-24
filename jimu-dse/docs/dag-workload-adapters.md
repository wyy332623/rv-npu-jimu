# DAG 多模型适配架构

## 目标

DAG 的 ISA 事件、micro-op、RAW 依赖和流量统计与模型无关；张量地址、执行阶段、缓存复用级别以及固件输入准备与模型有关。本架构将二者分开，禁止未知模型退回到 BERT 地址解释。

```text
固件 ELF + Workload Runtime
            │
            ├── 构建参数与输入/权重装载
            ├── 混合执行阶段边界
            ▼
       EventTracer（通用）
            ▼
   micro-op / RAW DAG（通用）
            ▼
   DAG Workload Adapter
            ├── DRAM 地址 → Tensor/position/tile/role
            ├── Tensor → L2/L3 复用策略
            └── 工作负载范围与目标源码
            ▼
   结构化 DAG、跨长度分析与保守证明
```

## 代码边界

- `jimu-dse/scripts/dag_workload_runtime.py`：固件构建、输入准备、固件运行和阶段边界。
- `emulator/npu_dag_adapter.py`：地址布局与张量语义，不执行固件。
- `emulator/npu_dag_structured.py`：只消费通用 micro-op 和 adapter 注释。
- `jimu-dse/scripts/npu_workload_probe.py`：统一生成指标 JSON，替代闭环中写死 BERT 的内嵌 Python。
- `jimu-dse/scripts/visualize_graph.py --workload NAME`：统一入口。

## 当前支持

| Workload | 原始 DAG | 结构化 Tensor | 阶段边界 | 通用 L1/L2 | 专用 L3 |
|---|---|---|---|---|---|
| `bert` | 支持 | 支持 | DRAM cluster 推断 | 支持 | 支持现有专用证明 |
| `adder_140p` | 支持 | 支持 | 两次固件执行的精确边界 | 支持 | 阻塞，等待 Adder MRF/调度证明 |

Adder-140p 是混合模型。DAG 覆盖两个 NPU 固件阶段，不包含 host/ISS 完成的 embedding、RMSNorm1、Q/KV、RoPE、RMSNorm2、Gate/Up、最终 RMSNorm 和 LM head。该边界写入 `run_metadata.json`，不能把输出解释成完整模型 DAG。

## 安全规则

1. 未注册 workload 直接报错，不允许按 BERT 解释。
2. Adapter 声明的 DRAM 区间必须互不重叠。
3. 原始事件地址和依赖始终是权威证据；Tensor 名称是 adapter 注释。
4. 通用层只自动证明精确 producer/consumer 的 L1，以及只读值和 VRF 生命周期明确的 L2。
5. L3 涉及 MRF 驻留、clobber、循环交换和浮点顺序。没有 workload 专用证明时必须输出 `blocked-workload-schedule-proof-required`。
6. DAG 允许在缺少 PyTorch 时使用明确标记的合成数值生成控制流。正确性门禁仍必须加载真实模型权重，不能使用合成数值。

## 新模型接入步骤

1. 在 `dag_workload_runtime.py` 实现 `WorkloadRuntime`：
   - `build(config)` 返回匹配配置的 ELF；
   - `run(config, elf)` 返回 NPU、EventTracer、TraceRecorder；
   - 混合执行模型必须返回半开区间 `[event_start, event_end)` 的阶段列表。
2. 在 `npu_dag_adapter.py` 实现 `DagWorkloadAdapter`：
   - 为每个 DRAM 区间声明 Tensor 名、role、position/tile stride；
   - 只为确实可跨循环复用的只读值声明 L2/L3 策略；
   - 设置真实 `target_file` 和 DAG 的 `execution_scope`。
3. 在两个 registry 中注册相同的 workload 名称。
4. 新建 `jimu-dse/workloads/NAME.sh`，声明源码、Make target、golden 测试与精确通过数量。
5. 选择至少两个不同且有语义的长度；禁止使用重复键，例如 `5 5`。
6. 添加测试，至少覆盖：区间边界、未知地址、阶段边界、跨长度对齐、错误 workload 拒绝和 L3 fail-closed。

## 使用示例

生成 BERT DAG：

```bash
python3 jimu-dse/scripts/visualize_graph.py \
  --phase micro --workload bert \
  --dim 2 --hidden 4 --seq-len 6 --num-head 2 \
  --output /tmp/dag-bert --no-render
```

生成 Adder-140p 的 NPU 子图：

```bash
python3 jimu-dse/scripts/visualize_graph.py \
  --phase micro --workload adder_140p \
  --dim 4 --hidden 4 --seq-len 24 --num-head 1 \
  --output /tmp/dag-adder-24 --no-render
```

闭环入口使用 manifest 名 `adder`：

```bash
bash jimu-dse/scripts/npu_closed_loop.sh \
  --workload adder --goal dram-optimization \
  --validation-dim all --start-from baseline
```

当前 Adder 专用 L3 证明尚未实现时，闭环会在没有安全 contract 后停止，而不是向 agent 发送 BERT 的错误优化建议。
