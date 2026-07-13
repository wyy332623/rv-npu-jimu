> 本文件由自动翻译生成，仅供参考；以英文原文为准。

---
名称: dag- 分析
说明: 读取 DAG 输出以识别 VRF 缓存的 DRAM 保存负载对
许可证:麻省理工学院
---

# DAG 分析技能

## 输入

管道在 QQZPROT000 QZ 生成这些文件:

|文件|内容|
|------|---------|
|津巴布韦|所有微型操作都带有ZPROT000Z和边缘|
|津巴布韦|带有 FLOP 、 字节、 AI 的 DRAM 流组|

## 如何阅读 DAG

### 识别保存损失的对等

保存的一对是QQZPROT000XXZ,然后是QZPROT0001Z,其地址为**same DRAM**,没有给该地址写插件。

在ZPROT000Z区:
```
 33 DRAM_STORE       DRAM_STORE           [71-72]      uses=VRF[7][0] defs=DRAM[0x300]
    <- [30 VV_ADD] via VRF[7][0]
    <- [32 VREG_MOVE] via ('pipe',)
 34 DRAM_STORE       DRAM_STORE           [73-74]      uses=VRF[8][0] defs=DRAM[0x308]
...
150 DRAM_LOAD        DRAM_LOAD            [340-341]    uses=DRAM[0x300] defs=VRF[18][0]
    <- [33 DRAM_STORE] via DRAM[0x300]
```

150号节点的边缘QQZPROT000XQZ显示,150号节点读了33号节点所写的. 这是一个**DRAM保存的负载配对**——数据到DRAM就马上回来.

### 确定要修改的函数

1. 查找 DRAM STORE — 请检查事件索引 (XZPROT000 XZ) 并找到相应的固件代码
2. 查找 DRAM LOAD — 检查事件索引并找到相应的固件代码
3. 两人都在ZPROT000Z区
4. DRAM地址告诉你哪一个是:0x200=Q,0x300=K,0x400=V

### 解释集群视图

在ZPROT000Z区:
```
  LoadB  StoreB  FLOPs   AI
   40       8      48    1.0  K Proj: loads WEIGHT+X, saves K
   32       8      60    1.5  Attn Score: loads K+Q+V, saves prob
```

- ZPROT000Z是DRAM ~ NPU流量
- ZPROT000Z是NPU ~ DRAM流量
- ZPROT000Z 正在计算
- ZPROT00000Z ——较高更好。

低AI集群( < 1.0)具有内存约束性,是首要的优化目标.

## 产出

编制合格保存对的列表:
```
Save-load pairs:
  DRAM_STORE[0x300] at node 33 → DRAM_LOAD[0x300] at node 150 (K[0])
  DRAM_STORE[0x308] at node 34 → DRAM_LOAD[0x308] at node 177 (K[0] tr1)
  DRAM_STORE[0x400] at node 89 → DRAM_LOAD[0x400] at node 161 (V[0])
  ...
```
