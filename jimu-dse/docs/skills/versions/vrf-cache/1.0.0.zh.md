---
name: vrf-cache
description: 将 K/V tensor 数据从 DRAM 往返重定向到片上 VRF cache
license: MIT
---

# VRF Cache 技能

## 问题

固件会为每个位置计算 `K=V=Wx+b`，通过 `save_row_tiles()` 写入 DRAM，然后在 attention 代码中通过 `V_RD_DRAM` 重新读取。当目标 VRF 容量足够时，这次往返是不必要的。

## VRF 容量

| VRF bank | Mem ID | 大小（元素） | 用途 |
|----------|--------|--------------|------|
| `MFU_INITIAL_VRF` | 6 | 4096 | GELU 临时结果 |
| `ADDSUB_VRF_0` | 7 | 1024 | tile 行 0 累加器 |
| `ADDSUB_VRF_1` | 8 | 4096 | tile 行 1 累加器和 X cache |
| `ADDSUB_VRF_2` | 9 | 64 | 第二个 tile 行（多 tile） |
| `MVM_INITIAL_VRF` | 5 | 20480 | MVM 输入向量 |
| `MULTIPLY_VRF` | 1 | 64 | 临时 MVM 结果 |

对于 `dim=2`、`hidden=4`、`seq_len=6`：每个位置的 K、V、Q 都是 4 个元素，6 个位置的 K+V 共 48 个元素；`MFU_INITIAL_VRF` 容量为 4096 个元素，仅使用约 **0.6%**。

## 变换

### 步骤 1：在 `mvm_tiled_q()` 计算 K/V 后缓存

`mvm_tiled_q()` 将 K 放入 `ADDSUB_VRF_0/1` 后，原本调用 `save_row_tiles()` 写入 DRAM。改为使用 VREG_MOVE 将数据复制到 mem 6，并按位置计算偏移：

```c
uint32_t cache_offset = pos * num_tiles * NATIVE_DIM;
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_SI(OP_V_WR, 6, cache_offset);
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
SEND_SI(OP_V_WR, 6, cache_offset + NATIVE_DIM);
```

跳过 `save_row_tiles()`，使数据保留在片上。

### 步骤 2：attention 中从 VRF 读取 K/V

K.T tile 原本使用：

```c
SEND_LO(OP_V_RD_DRAM, SAVE_K_BASE + p * num_tiles * 8 + tr * 8);
```

改为：

```c
uint32_t cache_offset = p * num_tiles * NATIVE_DIM + tr * NATIVE_DIM;
SEND_SI(OP_V_RD, 6, cache_offset);
```

V（包括 V.T 构造和 V.T 重新转置）使用相同的 cache 方案。

### 步骤 3：处理多个 tile 行

当 `num_tiles=2` 时，每个位置有 `tr=0` 和 `tr=1` 两个 tile 行，二者都必须缓存：

```c
uint32_t cache_offset = pos * num_tiles * NATIVE_DIM;
// tr=0 存放在 cache_offset
// tr=1 存放在 cache_offset + NATIVE_DIM
```

## 不要修改的内容

- 不要修改 Q 投影。Q 在 attention 中按位置计算，应保留在 `ADDSUB_VRF` 中并直接消费；
- 不要修改权重加载（`M_RD_DRAM`），权重必须来自 DRAM；
- 不要修改模拟器，只修改 `firmware/bert/bert_layer.c`；
- 不要改变数值计算。同一个 `W×x+b` 仍然会执行，变化的只是输出路由。

## 验证

```bash
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s --no-header 2>&1 | grep "max_diff"
```

所有值都必须小于 0.05。随后检查 DRAM 流量：

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k seq6 -s 2>&1 | grep "DRAM"
```
