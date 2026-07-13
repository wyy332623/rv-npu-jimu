> 本文件由自动翻译生成，仅供参考；以英文原文为准。

---
名称: 缩写
描述 : 在 NPU 固件上将 V WR DRAM+V RD DRAM {}ZPROT00000}} 成对的 INC 变体
许可证:麻省理工学院
---

您正在优化 NPU (rv- npu) 的固件, 一个 FPGA 神经处理单元 。
您的任务就是将**inc colding** 技能应用到 \ZPROT000XZ 。

## 触发模式

V WR DRAM 指令将一个向量保存到 DRAM , 之后是 V RD DRAM 指令
从 ** 同一地址装回, 并且没有给地址写任何插文。
冗余地址计算中的这种模式废物指示带宽。

检测 :
```
SEND_LO(OP_V_WR_DRAM, dram_base + tr * 8);     // save
... (no write to dram_base range) ...
SEND_LO(OP_V_RD_DRAM, dram_base + tr * 8);     // reload
```

## 转变

将 V WR DRAM 替换为 V WR DRAM INC 和配对的 V RD DRAM 替换为
V RD DRAM INC (英语). INC变体编码了自动递增的地址指针,
删除冗余地址计算。

在此之前:
```c
SEND_LO(OP_V_WR_DRAM, addr);    // save tile row
// ... intervening instructions (K projection, V projection)
SEND_LO(OP_V_RD_DRAM, addr);    // reload tile row
```

之后:
```c
// save_row_tiles_inc: uses V_WR_DRAM_INC internally
SEND_LO(OP_V_WR_DRAM_INC, addr);
// ... same intervening instructions ...
// V_RD_DRAM_INC reads from the SAME address. The INC auto-increment
// happens AFTER the read, so both write and read use the same addr.
SEND_LO(OP_V_RD_DRAM_INC, addr);
```

>>> HARDWARE 要求: INC变体必须使用LO格式 QQ
>>> SEND SI将输出全零Q输出——这是一个硬件事实,而不是样式选择QQ

V WR DRAM INC(opcode 23)和V RD DRAM INC(opcode 22) REQUIRE **LO格式**:

```c
SEND_LO(OP_V_WR_DRAM_INC, addr);   // CORRECT — addr encodes the base DRAM address
SEND_LO(OP_V_RD_DRAM_INC, addr);   // CORRECT — addr encodes the base DRAM address

SEND_SI(OP_V_WR_DRAM_INC, 0, stride);  // WRONG — SI format produces Q=[0,0,0,0]
SEND_SI(OP_V_RD_DRAM_INC, 0, stride);  // WRONG — SI format produces Q=[0,0,0,0]
```

** 要求联络处格式的原因**: INC 变体编码 SAPING DRAM 地址
在24位 LO 操作。 SI 格式只编码一个脚步,留下起始
地址未初始化(dram addr = 0),它从错误的内存位置读取。

脚步是隐含的(NATION DIM=8个元素). 不要使用 SEND SI。
不要试图使用 SEND SI —— 使用这种硬件是错误的。

读取的地址必须是作为写字地址的SAME.
V RD DRAM INC 读自 Currente dram addr,然后递增.
V WR DRAM INC 和 V RD DRAM INC 都用于同一瓦片行
使用相同的基址。 不要在读地址中添加+8。

## 函数已写入

基线固件中已经存在以下功能. 您无需创建它们 :

- 使用ZPROT0001Z。
- 使用ZPROT0001Z。

两者都使用LO格式(SEND LO). INC变体REQUIRE LO格式.
不要使用带有INC变体的SEND SI——它们需要24位地址.

## 你需要改变什么

替换 3 个保存  row  tiles () 调用和 3 V  RD  DRAM 内置负载
  process position () 及其 INC 等价物。

###  process position () — 保存调用

|行线|在此之前|之后|
|------|--------|-------|
| 221 |津巴布韦|津巴布韦|
| 223 |津巴布韦|津巴布韦|
| 225 |津巴布韦|津巴布韦|
| 276 |津巴布韦|津巴布韦|

###  process position () — 内置负载呼叫

这些都是在注意力循环。 将 SEND LO(OP V RD DRAM,...)替换为
SEND LO(OP V RD DRAM INC,.) — SAME地址,SAME格式,只是不同的opcode.

|行线|在此之前|之后|
|------|--------|-------|
| 248 |津巴布韦|津巴布韦|
| 250 |津巴布韦|津巴布韦|
| 262 |津巴布韦|津巴布韦|

### 应用  layernorm ()

|行线|在此之前|之后|
|------|--------|-------|
| 184 |津巴布韦|津巴布韦|
| 195 |津巴布韦|津巴布韦|

## 制约因素

1. 只修改 QQZPROT000XQZ. 不修改其他文件。
2. 只修改上表中列出的8个呼叫站点(替换函数)
   名称或操作码)。 不要改变任何其他逻辑。
3. 不要修改 QZPROT000XZ 或 ZPROT0001Z —— 它们
   已经正确。 将 ZPROT000XZ 改为 ZPROT0001Z
   函数将打破固件。
4. 文件必须保持有效C(由GCC为RISC-V编译).

## 核查

写入文件后, 您必须通过运行进行自我验证 :

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k "seq2" -q
```

请检查access-date=中的日期值 (帮助) 退出代码(应为0) 查看输出:
- “PASSED”——通过数字核查
- “ FAILED” — 数字验证失败
- “ 错误” —— 汇编或运行时间错误

不要跳过这一步。 输油管检查出口代码和最大位元。
如果测试失败,读取输出以找到错误,修复,再运行.
Max 3重试.

## 制约因素

1. 只修改 QQZPROT000XQZ. 不要修改任何其他文件。
2. 不要修改模拟器。 不修改 bert layer.c 以外的任何文件。
3. 保留所有现有功能——增加新功能,不要删除.
4. 文件必须保持有效C(由GCC为RISC-V编译).

## 示例:保存 row tiles inc

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

NOTE: The function MUST use `dram_base` (the parameter), not a hardcoded
address. V_WR_DRAM_INC uses LO format: SEND_LO(OP_V_WR_DRAM_INC, addr).
Do NOT use SEND_SI(OP_V_WR_DRAM_INC, 0, stride) — SI format is wrong.
The stride is implicit (NATIVE_DIM = 8).
```
