# NPU 链式执行示例

本目录演示使用链式 dispatch API（`npu_issue_chain` + `npu_wait_chain`）编写单链和多链固件。

## 背景

**指令链**是一组连续的 NPU 指令，通过 `OP_INST_ISSUE`（opcode 45）原子提交。链内：

- 指令写入 FIFO 时不进行逐指令停顿；
- pipeline register 在指令之间传递数据；
- `V_RD` 将向量加载到 pipeline，并把旧 pipeline 保存到 `vpipe_a`；
- `V_WR` 和 `V_WR_DRAM` 会保存结果，同时保留 pipeline 值；
- `INST_ISSUE` 清除 pipeline，值不会跨链保留。

## 隐式和显式操作数

| 操作数 | 来源 | 使用者 |
|--------|------|--------|
| MRF（显式） | `M_RD_DRAM` 或 `M_RD` 加载 | 仅 `MV_MUL` |
| pipeline（隐式） | 最近一次 `V_RD` 或 `V_RD_DRAM` | `MV_MUL`、`VV_*`、激活函数、`V_WR`、`V_WR_DRAM` |
| `vpipe_a`（隐式） | `V_RD` 保存旧 pipeline | `VV_ADD`、`VV_MUL` 等二元操作 |
| SRF（显式） | `OP_S_RECIP`、`OP_S_SQRT` 等写入 | `MEM_SPU_BROADCAST` |

`MV_MUL` 同时使用一个显式操作数 MRF 和一个隐式操作数 pipeline。指令字本身不编码这两个来源，链内的指令顺序保证它们处于正确状态。

## CHAIN_STATUS

| Bit | 单元 | 指令 |
|-----|------|------|
| 0 | VMM | `V_RD`、`V_WR`、DRAM 向量传输、`VV_*`、激活函数、`V_FUNC` |
| 1 | MMM | `M_RD_DRAM`、`M_RD`、`M_WR_DRAM` |
| 2 | MVU | `MV_MUL` |

pipeline 不是功能单元，而是所有指令共享的数据通路，不单独计 busy 状态。

## 示例文件

### `01_single_chain.c`

在一条链中完成一次 MVM：

```text
M_RD_DRAM → M_WR → V_RD_DRAM → MV_MUL → V_WR
```

随后第二条链读取 VRF 结果并写入 DRAM。

### `02_multi_chain.c`

- 链 1：MVM，`W × X → MULTIPLY_VRF`；
- 链 2：bias 加法，`W×X + bias → DRAM`；
- 还包含一个将 SiLU、MVM 和残差加法组合为单链的 FFN 示例。

独立链在真实 RTL 中可以进行 SMC（Simultaneous Multi-Chaining）并发，但当前 Python 模拟器仍按顺序执行。

### `03_softmax_chain.c`

演示通过隐式 pipeline 将以下操作连接起来：

```text
Q × K.T → score → Softmax → attention × V → context
```

Softmax 直接读取 `MV_MUL` 的 pipeline 输出，并将结果写回 pipeline，因此 score 和 Softmax 之间不需要 DRAM 保存-加载往返。

## DAG 生成

`chain_dag.py` 会生成事件级 DAG 和折叠后的微操作 DAG：

```bash
PYTHONPATH=. python3 firmware/examples/chain_dag.py --output /tmp/chain_dag/
dot -Tpng /tmp/chain_dag/chain_example_events.dot -o chain_events.png
dot -Tpng /tmp/chain_dag/chain_example_microops.dot -o chain_microops.png
```

事件级 DAG 会显示 `MV_MUL` 的两条输入边：一条通过 MRF 连接显式矩阵操作数，另一条通过 pipeline 连接隐式向量操作数。

## 构建

```bash
cd firmware
make TARGET=examples/01_single_chain BUILD_DIR=build_examples
make TARGET=examples/02_multi_chain BUILD_DIR=build_examples
make TARGET=examples/03_softmax_chain BUILD_DIR=build_examples
```

## API

| 函数 | 用途 |
|------|------|
| `npu_send_inst(inst)` | 推入一条指令，不进行 FIFO 停顿 |
| `npu_issue_chain()` | 发送 `OP_INST_ISSUE`，提交当前链 |
| `npu_wait_chain()` | 轮询 `NPU_CHAIN_STATUS`，等待所有单元空闲 |
| `npu_wait_done()` | 旧版接口，推荐使用链式 API |

