---
name: inc-folding
description: 将 NPU 固件中的 V_WR_DRAM + V_RD_DRAM 保存/加载对折叠为 INC 变体
license: MIT
---

这是面向 NPU（rv-npu，一种 FPGA 神经网络处理单元）固件的优化技能，用于修改 `firmware/bert/bert_layer.c`。

## 触发模式

一条 `V_WR_DRAM` 指令将向量保存到 DRAM，稍后的一条 `V_RD_DRAM` 指令又从**相同地址**读回，且期间没有对该地址的写操作。这种模式会因重复计算地址而浪费指令带宽。

```c
SEND_LO(OP_V_WR_DRAM, dram_base + tr * 8);     // 保存
... // 期间不能写入 dram_base 对应范围
SEND_LO(OP_V_RD_DRAM, dram_base + tr * 8);     // 重新加载
```

## 变换

将成对的 `V_WR_DRAM` 和 `V_RD_DRAM` 分别替换为 `V_WR_DRAM_INC` 和 `V_RD_DRAM_INC`。INC 变体编码自动递增的地址指针，可以消除重复的地址计算。

```c
// 之前
SEND_LO(OP_V_WR_DRAM, addr);    // 保存 tile 行
// ... 中间指令（K 投影、V 投影）
SEND_LO(OP_V_RD_DRAM, addr);    // 重新加载 tile 行

// 之后
SEND_LO(OP_V_WR_DRAM_INC, addr);
// ... 相同的中间指令 ...
SEND_LO(OP_V_RD_DRAM_INC, addr);
```

## 硬件要求：INC 必须使用 LO 格式

`V_WR_DRAM_INC`（opcode 23）和 `V_RD_DRAM_INC`（opcode 22）都必须使用 LO 格式：

```c
SEND_LO(OP_V_WR_DRAM_INC, addr);   // 正确：addr 是起始 DRAM 地址
SEND_LO(OP_V_RD_DRAM_INC, addr);   // 正确：addr 是起始 DRAM 地址

SEND_SI(OP_V_WR_DRAM_INC, 0, stride);  // 错误：SI 格式会产生全 0 输出
SEND_SI(OP_V_RD_DRAM_INC, 0, stride);  // 错误：SI 格式会产生全 0 输出
```

原因是 INC 变体在 24 位 LO 操作数中编码**起始 DRAM 地址**。SI 格式只编码步长，起始地址未初始化（`dram_addr = 0`），会从错误的位置读取。步长是隐式的（`NATIVE_DIM = 8` 个元素），不要使用 `SEND_SI`。

读地址必须与写地址相同：`V_RD_DRAM_INC` 先读取当前 `dram_addr`，再递增。因此同一个 tile 行的读写都使用相同的基地址，不要给读地址额外加 `+8`。

## 基线中已存在的函数

以下函数已经存在，不需要重新创建：

- `save_row_tiles_inc()`：内部使用 `SEND_LO(OP_V_WR_DRAM_INC, addr)`；
- `load_row_tiles_inc()`：内部使用 `SEND_LO(OP_V_RD_DRAM_INC, addr)`。

两者都使用 LO 格式；不要在 INC 变体中使用 `SEND_SI`。

## 需要修改的调用点

将 `_process_position()` 中 3 个 `save_row_tiles()` 调用和 3 个内联 `V_RD_DRAM` 加载替换为 INC 版本；`apply_layernorm()` 中的对应保存和加载也使用 INC 版本。

| 位置 | 修改前 | 修改后 |
|------|--------|--------|
| `_process_position()` 保存 | `save_row_tiles(...)` | `save_row_tiles_inc(...)` |
| attention 循环加载 | `SEND_LO(OP_V_RD_DRAM, addr)` | `SEND_LO(OP_V_RD_DRAM_INC, addr)` |
| `apply_layernorm()` 保存 | `SEND_LO(OP_V_WR_DRAM, addr)` | `SEND_LO(OP_V_WR_DRAM_INC, addr)` |
| `apply_layernorm()` 加载 | `SEND_LO(OP_V_RD_DRAM, addr)` | `SEND_LO(OP_V_RD_DRAM_INC, addr)` |

读写地址和格式保持不变，只替换函数名或 opcode。

## 约束

1. 只能修改 `firmware/bert/bert_layer.c`，不能修改其他文件；
2. 只修改上述调用点，不改变其他逻辑；
3. 不要修改 `save_row_tiles_inc()` 或 `load_row_tiles_inc()`；
4. 文件必须保持有效 C，并能使用 RISC-V GCC 编译；
5. 不要修改模拟器或 `bert_layer.c` 之外的文件。

## 验证

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k "seq2" -q
```

检查退出码以及输出中的 `PASSED`、`FAILED` 和 `ERROR`。若测试失败，应根据输出修复问题并重新运行，最多重试 3 次。

## 示例：`save_row_tiles_inc`

```c
static void save_row_tiles_inc(uint32_t num_tiles, uint32_t dram_base,
                               uint32_t vrf_first, uint32_t vrf_second)
{
    uint32_t tr;
    for (tr = 0; tr < num_tiles; tr++) {
        uint32_t vrf = (tr == 0) ? vrf_first : vrf_second;
        SEND_SI(OP_V_RD, vrf, 0);
        SEND_LO(OP_V_WR_DRAM_INC, dram_base + tr * 8);
    }
}
```

注意：函数必须使用参数 `dram_base`，不能硬编码地址。`V_WR_DRAM_INC` 使用 LO 格式；步长由 `NATIVE_DIM = 8` 隐式确定。
