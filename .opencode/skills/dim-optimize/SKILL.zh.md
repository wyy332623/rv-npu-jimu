---
name: dim-optimize
description: 将固件从多 tile 投影重构为单 tile 投影
---

# Dim-Optimize 技能

## 问题

当 `NATIVE_DIM < hidden_size` 时，每次投影需要 `num_tiles × num_tiles` 的分块矩阵乘。例如 `dim=2、hidden=4` 时，`num_tiles=2`，每次投影需要 4 条 `MV_MUL`。目标是让 `NATIVE_DIM` 与 `hidden_size` 匹配，使每次投影只需要一对 `M_RD_DRAM + MV_MUL`。

## 目标配置

```text
NATIVE_DIM = 4, hidden_size = 4, num_head = 2
MAT_SIZE = 16
head_size = 2
num_tiles = 1
heads_per_tile = 2
```

## 变换步骤

1. 将 `mvm_tiled_q()` 的 `tc/tr` 双重循环改为直线流程：一次加载完整输入向量、一次加载 4×4 权重矩阵、执行一次 `MV_MUL`，并取消 tile 累加；
2. 合并单 tile 下功能相同的 `mvm_tiled_q()` 和 `mvm_tiled_vrf()`；
3. 更新 attention，使一个 dim=4 向量携带两个 head：mask `0x03` 选择 head 0，mask `0x0C` 选择 head 1；
4. 简化 `save_row_tiles()` 和 `load_and_add_row_tiles()`，它们只处理一个 tile 行；
5. 简化 `apply_layernorm()`，只保留一次 gamma/beta 加载和一个 VRF 保存/恢复路径。

```text
VRF[6]: [h0_q0, h0_q1, h1_q0, h1_q1]
Mask 0x03 → [1, 1, 0, 0]
Mask 0x0C → [0, 0, 1, 1]
```

## 自验证

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k "dim4-h4" -v
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

如果删除了多 tile 路径，dim2-h4 和 dim4-h8 测试会自动跳过。

## 成本模型

| 指标 | dim=2（基线） | dim=4（优化后） |
|------|---------------|----------------|
| 每次投影的 MV_MUL | 4 | **1** |
| 每次投影的 M_RD_DRAM | 4 个 tile | **1 个 tile** |
| seq=6 的 M_RD_DRAM 总数 | 144 次 | **36 次** |
| 每次投影的 VV_ADD | 2 | **0** |
| VRF_ADDSUB 使用情况 | VRF_0、VRF_1、VRF_2 | **仅 VRF_0** |
| 每个 tile 行的 attention head 数 | 1 | 2（使用 mask） |
