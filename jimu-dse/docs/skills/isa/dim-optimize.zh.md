> 本文件由自动翻译生成，仅供参考；以英文原文为准。

---
名称: 淡化优化
说明: 将固件结构从多瓦调整为单瓦预测
---

# 低调的技巧

## 问题

当XZPROT000XZ时,每次投影都需要XZPROT0001ZZ.
瓷砖马特穆尔。 对于dim=2,隐藏=4:XQZPROT000XQZ,每个预测需要4 ZPROT0001XQZ.
说明。 目标是调整固件结构,使**NATION DIM匹配
隐藏 大小**,使每张投影单张XQZPROT000+XZPROT0001+Z对.

## 目标配置

```
NATIVE_DIM = 4, hidden_size = 4, num_head = 2
MAT_SIZE = NATIVE_DIM × NATIVE_DIM = 16
head_size = hidden_size / num_head = 2
num_tiles = 1  (single tile!)
heads_per_tile = NATIVE_DIM / head_size = 2
```

## 转换步骤

### 1. 简化投影功能

QQZPROT000XXZ函数具有XZPROT0001ZZ的环绕,比2×2=4次.
对于单瓦,这个折叠为直线序列:

** 在(dim=2, num tiles=2, MAT SIZE=4)之前:**
```c
for (tc = 0; tc < num_tiles; tc++) {
    SEND_LO(OP_V_RD_DRAM, vec_chunk_addr + tc * NATIVE_DIM);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
    for (tr = 0; tr < num_tiles; tr++) {
        SEND_LO(OP_M_RD_DRAM, mat_dram_base + (tr * num_tiles + tc) * MAT_SIZE);
        SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_MV_MUL, 0, 0);
        // ... accumulate via VV_ADD ...
    }
}
// Bias add per tile-row
for (tr = 0; tr < num_tiles; tr++) {
    SEND_LO(OP_V_RD_DRAM, bias_dram_base + tr * NATIVE_DIM);
}
// Move tr=1 accumulator to VRF_1
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_2, 0);
SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
```

** 在(dim=4、num tiles=1、MAT SIZE=16)之后:**
```c
// Single load of full input vector
SEND_LO(OP_V_RD_DRAM, input_vec_addr);
SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

// Single load of full weight matrix (4×4 = 16 elements)
SEND_LO(OP_M_RD_DRAM, mat_dram_base);
SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

// Single MV_MUL — the MRF holds the full 4×4 matrix
SEND_SI(OP_MV_MUL, 0, 0);
// Pipeline now has 4 elements = full hidden_size result

// No accumulation needed — single tile

// Bias add (single tile-row, full bias)
SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);  // save result
SEND_SI(OP_V_RD, MEM_FILL, 0);          // or load bias from DRAM
SEND_LO(OP_V_RD_DRAM, bias_dram_base);
SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, 0);
SEND_SI(OP_VV_ADD, 0, 0);
SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);  // store to a single tile-row VRF
```

### 2. 统一ZPROT0001Z和ZPROT00001Z

由于两者现在都进行相同的单瓦操作,所以将它们合并为一个功能.

### 3. 更新 ZPROT000Z 的注意

在凹陷=4时,头 大小=2,一瓦一行包含**2头**装在一个矢量内:

```
VRF[6] vector: [h0_q0, h0_q1, h1_q0, h1_q1]
                 head 0     head 1
Mask 0x03 → head 0: [1, 1, 0, 0]
Mask 0x0C → head 1: [0, 0, 1, 1]
```

QZPROT000XXZ 函数需要:
- 外部环绕 QQZPROT000XXZ( nums  tiles=1)
- 内部环绕QQZPROT000XZ(h=0,h=1).
- 每个头部使用一个遮罩从暗向量中选择其头 大小=2切片
- 用 QZPROT000 XZ 写回上下文

### 4. 简化 ZPROT000Z和 ZPROT0001Z

在 num tiles=1 时,这些操作在单瓦的一排上:
```c
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_LO(OP_V_WR_DRAM, dram_base);
```

### 5. 更新 QZPROT000XZ

在 num tiles=1 时,层纹在单瓦的一排上运行:
- 单装ZPROT000XZ
- 单VRF XQ ZPROT000XQZ
- 不需要瓦片-1备份逻辑

## 自我核查

修改固件后,用:

```bash
# Test single-tile config (dim=4, hidden=4, num_tiles=1)
python3 -m pytest tests/integration/test_bert_e2e.py -k "dim4-h4" -v

# All 6 test cases (if multi-tile paths were preserved):
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

测试套件包括专用的Dim4-h4测试箱用于单瓦验证.
如果多瓦路径被移除, Dim2-h4 和 dim4-h8 测试案例将
自动跳过。

## 成本模型

|度量衡|dim=2(基线)|dim=4(优化)|
|--------|------------------|-------------------|
|每个投影的MV MUL| 4 | **1** |
|M RD DRAM 每个投影|4 块|** 1 块**|
|M RD DRAM共计(seq=6)|144项行动|**36项行动**|
|每个投影VV ADD|2 (瓷砖堆积)| **0** |
|VRF  ADDSUB 用户|VRF 0, VRF 1, VRF 2|** 只限VRF 0**|
|每块砖头注意| 1 |2 (擦伤)|
