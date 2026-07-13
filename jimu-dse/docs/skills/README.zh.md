# NPU 优化技能库

本目录包含固件-硬件闭环协同优化系统的技能定义。每个技能描述一种 Agent 可以应用于 NPU 固件的优化模式。

## 目录结构

```text
skills/
  isa/             # 指令级优化
    inc_folding    # 将 V_WR + V_RD 折叠为 INC 变体
  fusion/          # 算子融合优化
    （计划中）
  tiling/          # 循环分块和 tile 配置
    （计划中）
  compensation/    # 有意近似与误差补偿
    （计划中）
```

## 技能格式

每个技能都是一个 YAML 文件，包含：

- **name**：唯一标识符
- **version**：语义化版本号
- **category**：技能分类
- **trigger**：激活技能的模式
- **preconditions**：应用前必须满足的条件
- **transformation**：变换前后的代码转换
- **cost_model**：预期节省量
- **validation**：正确性验证方法

## 使用方式

```bash
# 将 inc_folding 技能应用到当前固件
pi -p "Apply inc_folding skill to jimu-dse/docs/skills/isa/inc_folding.yaml on firmware/bert/bert_layer.c"
```

## 状态

| 技能 | 版本 | 状态 | DRAM 节省 |
|------|------|------|-----------|
| inc_folding | 1.0.0 | 草稿 | 待定 |
