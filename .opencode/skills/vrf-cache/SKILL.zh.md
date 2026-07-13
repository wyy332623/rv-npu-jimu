> 本文件由自动翻译生成，仅供参考；以英文原文为准。

---
名称: vrf- cache
描述: 将 QQZPROT000XZ 百分位数数据从 DRAM 圆盘转换到 VRF 缓存
许可证:麻省理工学院
---

# VRF 缓存技能

## 问题

固件计算每个位置的K=V=Wx+b,通过QQZPROT00000XQZ保存到DRAM,然后通过QZPROT0001XQZ从DRAM重新装入注意码. 当目标VRF具有能力时,这种绕行是不必要的.

## VRF 能力

|VRF银行|密码|大小( 要素)|用于|
|----------|--------|-----------------|----------|
|MFU INITIAL VRF (中文(简体) ).| 6 | 4096 |GELU 激活(临时)|
|ADDSUB VRF 0 (英语).| 7 | 1024 |0行积分计|
|ADDSUB VRF 1 (中文(简体) ).| 8 | 4096 |第1行积分器 + X 缓存|
|ADDSUB VRF 2 软件| 9 | 64 |第二瓦线(多瓦)|
|MVM INITIAL VRF (英语).| 5 | 20480 |MVM 输入向量|
|百万维基月球| 1 | 64 |临时MVM结果|

对于dim=2,隐藏=4,下游 len=6:
- K / 位置: 4 个元素(2 瓦排 × 2 dim)
- 每个职位五:4个要素
- 每个职位的Q:4个要素
- 所有6个职位的K+V:共计48个要素
- MFU INITIAL VRF容量:4096个元素 → **0.6%使用**

## 转变

### 第1步:在XZPROT0001Z计算后

在QQZPROT000XZ在QZPROT0001XZ生产K后,固件呼叫QZPROT00002Z写给DRAM. 相反:

1. 插入 VREG MOVE 指令,在位置索引偏移处将 QQZPROT000XXZ 复制到 ZPROT0001Z 。
2. 跳过 QQZPROT000Z —— 数据保留在芯片上

VREG 移动模式 :
```c
// Instead of:
save_row_tiles(num_tiles, SAVE_K_BASE + pos * num_tiles * 8,
               MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);

// Use:
uint32_t cache_offset = pos * num_tiles * NATIVE_DIM;
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_SI(OP_V_WR, 6, cache_offset);  // MFU_INITIAL_VRF[offset]
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
SEND_SI(OP_V_WR, 6, cache_offset + NATIVE_DIM);
```

### 步骤2:在注意时从 VRF 装入 XZPROT000XZ

在ZPROT000XXZ中,K.T瓦的建筑用途是:
```c
SEND_LO(OP_V_RD_DRAM, SAVE_K_BASE + p * num_tiles * 8 + tr * 8);
```

改为VRF:
```c
uint32_t cache_offset = p * num_tiles * NATIVE_DIM + tr * NATIVE_DIM;
SEND_SI(OP_V_RD, 6, cache_offset);  // MFU_INITIAL_VRF[offset]
```

对于V(在V.T building和V.T 重译中使用)类似.

### 第3步:处理横跨瓦片行的 XZPROT000ZZ

以 num tiles=2为单位,每个位置有2个瓦片行(tr=0,tr=1). 两者必须缓存:
```c
uint32_t cache_offset = pos * num_tiles * NATIVE_DIM;
// tr=0 stored at cache_offset
// tr=1 stored at cache_offset + NATIVE_DIM
```

## 什么不能改变

- 不修改 Q 投影。 Q在注意期间按职位计算,而不是对所有职位预先计算。 Q应留在ADDSUB VRF并直接消耗.
- 不要修改重载(M RD DRAM). 体重必须来自DRAM.
- 不要修改模拟器。 只修改 QQZPROT000XQZ.
- 不改变数值计算。 执行相同的W×x+b;只有输出路由改变.

## 核查

运行仪器测试以检查所有运算符产生正确的输出 :
```bash
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s --no-header 2>&1 | grep "max_diff"
```

所有值必须 < 0.05。 还核实DRAM的减少:
```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k seq6 -s 2>&1 | grep "DRAM"
```
