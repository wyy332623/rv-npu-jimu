> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# 构建指南

## 先决条件

### 需要的工具

|工具|选项。 版本|安装|
|------|-------------|---------|
|Py| ≥ 3.10 |津巴布韦|
|计算| ≥ 3.20 |津巴布韦|
|海湾合作委员会(本土)| ≥ 10 |——为主机编译 QZPROT0001Z|
|RISC-V 跨海合会| ≥ 10 |XQ ZPROT000XQZ – 编译 RV64IM 光金属的固件|
|数字| ≥ 2.0 |津巴布韦|
|测试| ≥ 7.0 |津巴布韦|
|长凳| ≥ 0.29 |津巴布韦|

### 可选工具

|工具|版本|安装|目的|
|------|---------|---------|---------|

|图维兹| — |津巴布韦|Render DAG.dot 文件到 SVG|
|毕| ≥ 0.79 |津巴布韦|自动优化的AI代理|

### 国际空间站

MiniRV64是一个纯Python RV64IM ISS,包含在XQZPROT000XQZ的回波中.
没有外部依赖关系。

## 构建步骤

### 1. C 核心图书馆

```bash
make kernels
```

将 QQZPROT000XZ 编译为 ZPROT0001Z, 共享
模拟器用于快速矩阵乘法,GELU,软max等的库.

### 2. RISC-V 软件

```bash
make firmware
```

将QQZPROT000XQZ编译为RISC-VELF二进制. 那个
固件是一个在MiniRV64上运行并驱动
NPU通过MMIO写作. 产出:ZPROT000Z。

### 3. 运行测试

```bash
# Integration test (BERT E2E)
python3 -m pytest tests/integration/ -v

# All integration + unit tests
python3 -m pytest tests/ -v
```

测试套件自动检测可选依赖性 :

- ** 无国际空间站** — 试验优雅地跳过解释性信息

## 目录布局

```
_build/
└── kernels/libnpukernels.so      ← C kernel library (compiled)
firmware/
└── build_dim{2,4}/bert.elf       ← RISC-V firmware (compiled per config)
```

使用正确的 DRAM 版式自动重建固件 ELF
每次闭路管道探测到一个配置时,都会进行宏。
