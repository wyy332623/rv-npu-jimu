---
name: dag-analyze
description: 读取 DAG 输出，识别用于 VRF 缓存的 DRAM 保存-加载对
license: MIT
---

# DAG 分析技能

## 输入

流程会在 `_out/` 生成：

| 文件 | 内容 |
|------|------|
| `micro_op_dag.txt` | 所有微操作、定义/使用关系和数据依赖边 |
| `dram_clusters.txt` | 包含 FLOPs、字节数和算术强度（AI）的 DRAM 流量簇 |

## 识别保存-加载对

保存-加载对是指一个 `DRAM_STORE` 后面紧跟着从**相同 DRAM 地址**读取的 `DRAM_LOAD`，且期间没有对该地址的写操作。

```text
33 DRAM_STORE  uses=VRF[7][0] defs=DRAM[0x300]
150 DRAM_LOAD   uses=DRAM[0x300] defs=VRF[18][0]
    <- [33 DRAM_STORE] via DRAM[0x300]
```

该边表示节点 150 读取了节点 33 写入的内容，即数据先写入 DRAM，随后又被读回。

## 定位修改位置

1. 根据 `DRAM_STORE` 的事件索引定位固件代码；
2. 根据 `DRAM_LOAD` 的事件索引定位对应的读取代码；
3. 两者通常位于 `firmware/bert/bert_layer.c`；
4. 地址可帮助识别 tensor：`0x200=Q`、`0x300=K`、`0x400=V`。

## 解读流量簇

- `LoadB`：DRAM → NPU 的流量；
- `StoreB`：NPU → DRAM 的流量；
- `FLOPs`：执行的浮点运算量；
- `AI = FLOPs / (LoadB + StoreB)`：算术强度，越高越好。

AI 低于 1.0 的簇受内存带宽限制，是优先优化目标。

## 输出

输出符合条件的保存-加载对，例如：

```text
Save-load pairs:
  DRAM_STORE[0x300] at node 33 → DRAM_LOAD[0x300] at node 150 (K[0])
  DRAM_STORE[0x308] at node 34 → DRAM_LOAD[0x308] at node 177 (K[0] tr1)
  DRAM_STORE[0x400] at node 89 → DRAM_LOAD[0x400] at node 161 (V[0])
```

