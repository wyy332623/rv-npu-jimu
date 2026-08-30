---
name: vrf-cache
description: 将 K/V tensor 数据从 DRAM 往返重定向到片上 VRF cache
license: MIT
---

# VRF Cache 技能

## 问题

固件为每个位置计算 `K=V=Wx+b`，先通过 `save_row_tiles()` 写入 DRAM，再由 attention 通过 `V_RD_DRAM` 读回。如果目标 VRF 容量足够，这次 DRAM 往返是不必要的。

## 变换

1. 在 `mvm_tiled_q()` 计算 K/V 后，将 `ADDSUB_VRF_0/1` 中的数据通过 `VREG_MOVE` 复制到 `MFU_INITIAL_VRF`（mem 6），并使用按位置计算的偏移；
2. 跳过 `save_row_tiles()`，让数据留在片上；
3. 在 `dot_product_attention()` 中将 K/V 的 `V_RD_DRAM` 替换为 mem 6 的 `V_RD`。

```c
uint32_t cache_offset = pos * num_tiles * NATIVE_DIM;
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_SI(OP_V_WR, 6, cache_offset);
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
SEND_SI(OP_V_WR, 6, cache_offset + NATIVE_DIM);
```

读取时：

```c
uint32_t cache_offset = p * num_tiles * NATIVE_DIM + tr * NATIVE_DIM;
SEND_SI(OP_V_RD, 6, cache_offset);
```

当 `num_tiles=2` 时，`tr=0` 存放在 `cache_offset`，`tr=1` 存放在 `cache_offset + NATIVE_DIM`。K 和 V 都必须缓存。

## 不要修改的内容

- 不要修改 Q 投影；Q 在 attention 中按位置计算，并直接从 `ADDSUB_VRF` 消费；
- 不要修改权重加载（`M_RD_DRAM`），权重必须来自 DRAM；
- 不要修改模拟器，只修改 `firmware/bert/bert_layer.c`；
- 不要改变数值计算，只改变输出路由。

## 验证

```bash
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s --no-header 2>&1 | grep "max_diff"
python3 -m pytest tests/integration/test_bert_e2e.py -k seq6 -s 2>&1 | grep "DRAM"
```

所有 `max_diff` 都必须小于 0.05，同时应确认 DRAM 流量下降。
