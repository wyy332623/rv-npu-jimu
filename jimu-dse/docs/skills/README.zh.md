> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU 优化技能库

此目录包含闭路FW- HW 的技能定义
合作优化系统。 每个技能都描述了优化模式
代理可以申请 NPU 固件。

## 目录结构

```
skills/
  isa/             # Instruction-level optimizations
    inc_folding    # Fold V_WR+V_RD into INC variants
  fusion/          # Operator fusion optimizations
    (planned)
  tiling/          # Loop tiling and tile configuration
    (planned)
  compensation/    # Intentional approximation + compensation
    (planned)
```

## 技能格式

每种技能都是YAML文件,其中:

- ** 名称**:唯一标识符
- ** 版本**:Semver
- ** 类别**: 技能分类
- **触发**:激活技能的模式
- ** 先决条件**: 适用前必须具备的条件
- ** 转换**: XQZPROT000XQZ代码转换
- ** 成本 模型**: 预期节余
- ** 验证**: 如何核实正确性

## 使用量

```bash
# Apply a skill to the current firmware
pi -p "Apply inc_folding skill to jimu-dse/docs/skills/isa/inc_folding.yaml on firmware/bert/bert_layer.c"
```

## 状态

|技能|版本|状态|DRAM 保存|
|-------|---------|--------|-------------|
|内存( C)| 1.0.0 |草案|传统发展|
