---
name: vrf-cache
description: 将 K/V tensor 数据从 DRAM 往返重定向到片上 VRF cache
license: MIT
---

# VRF Cache 技能

## 问题

固件通常会计算 K、V、Q、Z、SO 或 LayerNorm 输出等中间结果，通过存储宏/函数（如 `save_row_tiles()`、`SEND_LO(OP_V_WR_DRAM, ...)`）写入 DRAM，随后下游操作再通过 `V_RD_DRAM` 从 DRAM 读取。当目标 VRF 容量足够时，这种往返是不必要的。

## VRF 容量

| VRF bank | Mem ID | 大小（元素） | 用途 |
|----------|--------|--------------|------|
| `MFU_INITIAL_VRF` | 6 | 4096 | 通用中间结果缓存（如 GELU） |
| `ADDSUB_VRF_0` | 7 | 1024 | tile 行 0 累加器 |
| `ADDSUB_VRF_1` | 8 | 4096 | tile 行 1 累加器和 X cache |
| `ADDSUB_VRF_2` | 9 | 64 | 第二个 tile 行（多 tile） |
| `MVM_INITIAL_VRF` | 5 | 20480 | MVM 输入向量 |
| `MULTIPLY_VRF` | 1 | 64 | 临时 MVM 结果 |

通常选择 `MFU_INITIAL_VRF`（mem 6）作为 scratchpad，因为它具有较大的 4096 元素容量。

## 变换

请先分析正在编辑的具体代码库。以下模式适用于**任何 NPU 工作负载**，不只适用于 BERT；如果当前固件中不存在 `mvm_tiled_q` 或 `SAVE_K_BASE`，不要盲目搜索它们。

### 通用步骤 1：在 C 代码中定位保存-加载对

使用 `dag-analyze` 技能的输出确定哪些 DRAM 地址对应保存-加载往返，然后找到这些地址在 C 代码中的写入和读取位置。

- **保存模式**：查找 `SEND_LO(OP_V_WR_DRAM, <addr>)` 或执行该操作的封装函数；
- **加载模式**：查找 `SEND_LO(OP_V_RD_DRAM, <addr>)` 或执行该操作的封装函数。

### 通用步骤 2：将保存重定向到 VRF 缓存

不要写入 DRAM，而应将数据移动到指定的 VRF bank（通常为 6，即 `MEM_MFU_INITIAL_VRF`）。必须为每个保存的数据分配唯一的 `cache_offset`，避免相互覆盖：

```c
// 不再写入 DRAM：
// SEND_LO(OP_V_WR_DRAM, addr);

// 假设数据当前位于 MEM_ADDSUB_VRF_0 等源 VRF 中：
uint32_t cache_offset = <根据循环索引计算唯一偏移>;
// 如果数据尚未处于可读状态，先激活它：
// SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_SI(OP_V_WR, 6, cache_offset);  // 写入 MFU_INITIAL_VRF[offset]
```

### 通用步骤 3：将加载重定向到 VRF 缓存

用 VRF 读取替换后续 DRAM 加载，并使用与保存时完全相同的 `cache_offset`：

```c
// 不再从 DRAM 加载：
// SEND_LO(OP_V_RD_DRAM, addr);
uint32_t cache_offset = <计算完全相同的唯一偏移>;
SEND_SI(OP_V_RD, 6, cache_offset);
```

## 不要修改的内容

- 不要修改 Q 投影。Q 在 attention 中按位置计算，应保留在 `ADDSUB_VRF` 中并直接消费；
- 不要修改权重加载（`M_RD_DRAM`），权重必须来自 DRAM；
- 不要修改模拟器，只修改当前工作负载对应的固件文件（例如 `firmware/bert/bert_layer.c`、`adderboard/firmware/adder_140p.c`）；
- 不要改变数值计算。同一个 `W×x+b` 仍然会执行，变化的只是输出路由。

## 验证

修改固件后，应按照项目的测试说明执行验证。可以使用 `self-verify` 技能，或查阅项目说明以获取准确的验证和收敛检查命令。
