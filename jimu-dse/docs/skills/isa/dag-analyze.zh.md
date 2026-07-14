---
name: dag-analyze
description: 读取 DAG 输出，识别用于 VRF 缓存的 DRAM 保存-加载对
license: MIT
---

# DAG 分析技能

## 输入

流程会在 `_out/` 生成以下文件：

| 文件 | 内容 |
|------|------|
| `micro_op_dag.txt` | 所有微操作、定义/使用关系和边 |
| `dram_clusters.txt` | 包含 FLOPs、字节数和算术强度（AI）的 DRAM 流量簇 |

## 如何读取 DAG

### 识别保存-加载对

保存-加载对是指：一个 `DRAM_STORE` 后面紧接着一个从**相同 DRAM 地址**读取的 `DRAM_LOAD`，且两者之间没有对该地址的写操作。

在 `micro_op_dag.txt` 中：

```text
 33 DRAM_STORE       DRAM_STORE           [71-72]      uses=VRF[7][0] defs=DRAM[0x300]
    <- [30 VV_ADD] via VRF[7][0]
    <- [32 VREG_MOVE] via ('pipe',)
 34 DRAM_STORE       DRAM_STORE           [73-74]      uses=VRF[8][0] defs=DRAM[0x308]
...
150 DRAM_LOAD        DRAM_LOAD            [340-341]    uses=DRAM[0x300] defs=VRF[18][0]
    <- [33 DRAM_STORE] via DRAM[0x300]
```

节点 150 上的边 `<- [33 DRAM_STORE] via DRAM[0x300]` 表明节点 150 读取了节点 33 写入的内容。这就是一个 **DRAM 保存-加载对**：数据先写入 DRAM，随后又从 DRAM 读回。

### 定位需要修改的函数

1. 找到 `DRAM_STORE`，根据事件索引（如 `[71-72]`）定位对应的固件代码；
2. 找到 `DRAM_LOAD`，根据事件索引定位对应的固件代码；
3. 根据当前工作负载定位对应的固件文件；
4. DRAM 地址的含义取决于固件的 scratch space 布局，应查看定义映射关系的 C 宏。

### 解读流量簇

```text
  LoadB  StoreB  FLOPs   AI
   40       8      48    1.0  K Proj: loads WEIGHT+X, saves K
   32       8      60    1.5  Attn Score: loads K+Q+V, saves prob
```

- `LoadB`：DRAM → NPU 的流量；
- `StoreB`：NPU → DRAM 的流量；
- `FLOPs`：执行的浮点运算量；
- `AI = FLOPs / (LoadB + StoreB)`：算术强度，越高越好。

算术强度低于 1.0 的簇受内存带宽限制，是优先优化目标。

## 输出

输出符合条件的保存-加载对列表：

```text
Save-load pairs:
  DRAM_STORE[0x300] at node 33 → DRAM_LOAD[0x300] at node 150 (K[0])
  DRAM_STORE[0x308] at node 34 → DRAM_LOAD[0x308] at node 177 (K[0] tr1)
  DRAM_STORE[0x400] at node 89 → DRAM_LOAD[0x400] at node 161 (V[0])
  ...
```
