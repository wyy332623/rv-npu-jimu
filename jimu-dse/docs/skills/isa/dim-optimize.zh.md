---
name: dim-optimize
description: 将固件从多 tile 投影重构为单 tile 投影
---

# Dim-Optimize 技能

## 问题

当 `NATIVE_DIM < hidden_size` 时，每次投影都需要 `num_tiles × num_tiles` 的分块矩阵乘。以 `dim=2、hidden=4` 为例，`num_tiles=2`，每次投影需要 4 条 `MV_MUL` 指令。目标是让 **`NATIVE_DIM` 与 `hidden_size` 匹配**，使每次投影只需要一对 `M_RD_DRAM + MV_MUL`。

## 目标配置

```text
NATIVE_DIM = 4, hidden_size = 4, num_head = 2
MAT_SIZE = NATIVE_DIM × NATIVE_DIM = 16
head_size = hidden_size / num_head = 2
num_tiles = 1（单 tile）
heads_per_tile = NATIVE_DIM / head_size = 2
```

## 变换步骤

### 1. 简化投影函数

`mvm_tiled_q()` 中原有的 `tc`、`tr` 双重循环会执行 2×2=4 次。单 tile 配置下，将其改为直线流程：

```c
// 一次加载完整输入向量
SEND_LO(OP_V_RD_DRAM, input_vec_addr);
SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

// 一次加载完整权重矩阵（4×4 = 16 个元素）
SEND_LO(OP_M_RD_DRAM, mat_dram_base);
SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

// 一次 MV_MUL；MRF 中保存完整的 4×4 矩阵
SEND_SI(OP_MV_MUL, 0, 0);
// pipeline 现在包含完整 hidden_size 的 4 个元素

// 单 tile 不需要累加
```

随后执行单个 tile 行的 bias 加法，并将结果写入一个 tile-row VRF。无需保留多 tile 累加器和 tile-1 的备份逻辑。

### 2. 合并 `mvm_tiled_q()` 与 `mvm_tiled_vrf()`

两者在单 tile 配置下执行相同的操作，因此可以合并为一个函数。

### 3. 更新 attention 的 `heads_per_tile=2`

在 `dim=4、head_size=2` 时，一个 tile 行将两个 head 打包在同一个向量中：

```text
VRF[6] vector: [h0_q0, h0_q1, h1_q0, h1_q1]
                 head 0     head 1
Mask 0x03 → head 0: [1, 1, 0, 0]
Mask 0x0C → head 1: [0, 0, 1, 1]
```

`dot_product_attention()` 需要：

- 只遍历 `tr=0`；
- 内层遍历 `h=0..heads_per_tile-1`（即 h=0、h=1）；
- 每个 head 使用 mask 从 dim=4 向量中选出 2 个元素；
- 使用 `REG_WRITE_VECTOR_MASK` 将 context 写回对应切片。

### 4. 简化 `save_row_tiles()` 和 `load_and_add_row_tiles()`

当 `num_tiles=1` 时，它们只操作一个 tile 行：

```c
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_LO(OP_V_WR_DRAM, dram_base);
```

### 5. 更新 `apply_layernorm()`

单 tile 下 LayerNorm 只需要：

- 加载一次 gamma/beta；
- 保存/恢复一个 VRF；
- 不再需要 tile-1 备份逻辑。

## 自验证

```bash
# 单 tile 配置（dim=4，hidden=4，num_tiles=1）
python3 -m pytest tests/integration/test_bert_e2e.py -k "dim4-h4" -v

# 全部 6 种配置（保留多 tile 路径时）
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

测试套件包含专门的 dim4-h4 单 tile 用例。若删除多 tile 路径，dim2-h4 和 dim4-h8 用例会自动跳过。

## 成本模型

| 指标 | dim=2（基线） | dim=4（优化后） |
|------|---------------|----------------|
| 每次投影的 MV_MUL | 4 | **1** |
| 每次投影的 M_RD_DRAM | 4 个 tile | **1 个 tile** |
| seq=6 的 M_RD_DRAM 总数 | 144 次 | **36 次** |
| 每次投影的 VV_ADD | 2（tile 累加） | **0** |
| VRF_ADDSUB 使用情况 | VRF_0、VRF_1、VRF_2 | **仅 VRF_0** |
| 每个 tile 行的 attention head 数 | 1 | 2（使用 mask） |
