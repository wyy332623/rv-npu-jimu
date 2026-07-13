# 构建指南

## 前置条件

### 必需工具

| 工具 | 最低版本 | 安装方式 |
|------|----------|----------|
| Python | ≥ 3.10 | `apt install python3 python3-pip python3-venv` |
| CMake | ≥ 3.20 | `apt install cmake` |
| GCC（本机） | ≥ 10 | `apt install gcc`，用于为主机构建 `libnpukernels.so` |
| RISC-V 交叉 GCC | ≥ 10 | `apt install gcc-riscv64-unknown-elf`，用于构建 RV64IM 裸机固件 |
| numpy | ≥ 2.0 | `pip install numpy` |
| pytest | ≥ 7.0 | `pip install pytest` |
| pyelftools | ≥ 0.29 | `pip install pyelftools` |

### 可选工具

| 工具 | 版本 | 安装方式 | 用途 |
|------|------|----------|------|
| Graphviz | — | `apt install graphviz` | 将 DAG `.dot` 文件渲染为 SVG |
| pi | ≥ 0.79 | `npm install -g @earendil-works/pi-coding-agent` | 自动优化所用的 AI Agent |

### ISS

仓库中的 `iss/mini_rv64.py` 提供纯 Python 实现的 RV64IM MiniRV64 指令集模拟器，无外部依赖。

## 构建步骤

### 1. C kernel 库

```bash
make kernels
```

该命令将 `kernels/*.c` 编译为 `_build/kernels/libnpukernels.so`。这是模拟器使用的共享库，用于高效执行矩阵乘、GELU、Softmax 等计算。

### 2. RISC-V 固件

```bash
make firmware
```

该命令将 `firmware/bert/bert_layer.c` 编译为 RISC-V ELF 二进制文件。固件是运行在 MiniRV64 上的裸机程序，通过 MMIO 写操作驱动 NPU。输出文件为 `firmware/build/bert.elf`。

### 3. 运行测试

```bash
# 集成测试（BERT 端到端）
python3 -m pytest tests/integration/ -v

# 全部集成测试和单元测试
python3 -m pytest tests/ -v
```

测试套件会自动检测可选依赖：

- **未安装 ISS**：测试会跳过相关项目，并给出说明。

## 目录结构

```text
_build/
└── kernels/libnpukernels.so      ← 已编译的 C kernel 库
firmware/
└── build_dim{2,4}/bert.elf       ← 按配置编译的 RISC-V 固件
```

每次闭环流程探测一种配置时，系统都会使用正确的 DRAM 布局宏自动重新构建固件 ELF。
