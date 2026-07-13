# NPU Optimization Skill Library

This directory contains skill definitions for the closed-loop FW-HW
co-optimization system. Each skill describes an optimization pattern
that an agent can apply to the NPU firmware.

## Directory Structure

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

## Skill Format

Each skill is a YAML file with:

- **name**: Unique identifier
- **version**: Semver
- **category**: Skill classification
- **trigger**: Pattern that activates the skill
- **preconditions**: Conditions that must hold before applying
- **transformation**: Before/after code transformation
- **cost_model**: Expected savings
- **validation**: How to verify correctness

## Usage

```bash
# Apply a skill to the current firmware
pi -p "Apply inc_folding skill to jimu-dse/docs/skills/isa/inc_folding.yaml on firmware/bert/bert_layer.c"
```

## Status

| Skill | Version | Status | DRAM Saving |
|-------|---------|--------|-------------|
| inc_folding | 1.0.0 | Draft | TBD |
