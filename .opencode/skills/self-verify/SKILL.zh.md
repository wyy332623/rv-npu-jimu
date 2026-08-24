---
name: self-verify
description: 强制执行固件正确性门禁，禁止通过过滤输出或弱化测试获得成功
license: MIT
---

# Self-Verify 技能

该技能是每次优化的必选技能，排在目标优化技能之后执行，但不能替代闭环中的
独立验收门禁。

## 验收权威

- 运行 prompt 中 `Independent Acceptance Gate` 给出的原始命令。
- 禁止追加 `grep`、`head`、`tail`、`|| true` 或自行修改 `-k` 表达式。
- 成功必须同时满足：pytest 返回码为 0、通过数量符合范围、零 skipped。
- 禁止修改测试、容差、golden 数据、skip/xfail、模拟器、ISS、硬件模型或验证命令。
- 局部测试只能用于定位问题，不能用于接收候选固件。

## BERT 验收矩阵

| 验证范围 | 必须覆盖的配置 | 预期通过数 |
|---|---|---:|
| `dim2` | dim2/hidden4 的 seq2、seq6 | 2 |
| `dim4` | dim4/hidden8 和 dim4/hidden4 的 seq2、seq6 | 4 |
| `all` | 上述全部 DIM2、DIM4 配置 | 6 |

生产候选应使用 `all`。`dim2`、`dim4` 用于定向调试；部分验证通过不能表述为
跨 DIM 正确。

## 必须执行的流程

1. 编译修改后的目标固件。
2. 原样运行 prompt 中的独立验收命令。
3. 检查命令返回码，不能根据过滤后的文本判断。
4. 检查通过数量是否与所选验证范围一致。
5. 检查 pytest 是否报告 skipped。
6. 正确性通过后才能比较性能指标。

如果正确性失败，应按以下顺序定位首个分歧：

1. Q/K/V 投影；
2. attention score 和 softmax；
3. context/self-output；
4. 第一次残差和 LayerNorm；
5. FFN intermediate 和 GELU；
6. FFN output、第二次残差和最终 LayerNorm。

优化后部分张量可能只存在于 VRF。只有 instrumentor 确实捕获到对应张量时，
才能声称该中间结果与 golden 一致。

Agent 结束前必须报告：

```text
validation_scope:
acceptance_command:
pytest_returncode:
passed/expected:
skipped:
metric_before:
metric_after:
result: PASS or FAIL
```

只有闭环的独立门禁可以将候选标记为 accepted。
