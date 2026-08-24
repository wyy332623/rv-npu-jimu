# JIMU-DSE 文档入口

本目录只把“当前如何运行”和“历史上做过什么”分开组织，不把实验报告当作实时规范。

## 建议阅读顺序

1. [当前项目状态](project-status.zh.md)：已实现功能、验证结果、边界和下一步。
2. [运行指南](how-to-run.zh.md)：构建、验证、启动闭环、选择基线和清理工作区。
3. [结构化 DAG](structured-dag.zh.md)：DAG 文件、跨序列/跨配置证明和证据门禁。
4. [闭环设计](npu-closed-loop-design.zh.md)：Probe、Analyze、Agent、Validate、Deploy 的职责。
5. [Skill 管理](skills/README.zh.md)：单一真源、版本、同步、回滚和 SHA256。

英文入口见 [README.md](README.md)。基础 NPU 的 ISA、固件、构建和测试文档位于仓库顶层 [`docs/`](../../docs/)。

## 文档分区

| 分区 | 内容 | 是否描述当前状态 |
|---|---|---|
| 本目录顶层 | 运行、设计、DAG、项目状态 | 是 |
| [`skills/`](skills/README.zh.md) | Agent 的可执行知识和版本记录 | 是，以 `skills/isa/` 为真源 |
| [`reports/`](reports/README.zh.md) | 带日期和假设的专题推导 | 否，需结合当前状态阅读 |
| [`archive/`](archive/README.zh.md) | 旧仓库、旧实现和早期实验报告 | 否，仅用于追溯 |
| [`../results/`](../results/README.md) | 本地运行证据；新 run 默认忽略 | 单次运行有效，不是规范 |

## 事实优先级

文档与实现不一致时，按下面的顺序判断：

1. 当前源码和独立测试；
2. `run_manifest.json`、`skills_manifest.json`、DAG JSON/JSONL 和 SHA256；
3. 本目录的当前状态、运行和设计文档；
4. 专题报告；
5. 归档文档和 Agent 自述。

任何 `candidate_best.c` 都只是某次运行的最佳候选，不会自动成为正确性基线。规范基线固定为 `jimu-dse/baseline/bert_layer.c`。

## 维护约定

- 功能行为变化时，优先更新 `project-status.zh.md` 和对应专题文档。
- CLI 变化时同时更新中英文运行指南。
- DAG schema 或门禁变化时同时更新中英文结构化 DAG 文档。
- Skill 正文只修改 `skills/isa/*.md`，随后运行 `skillctl.py sync`；不要直接编辑 `.opencode/skills/`。
- 历史结论必须写明日期、配置和哈希，然后放入 `reports/` 或 `archive/`。
- `results/run-*`、ELF、对象文件和缓存不提交到源码分支。

## 常用入口

```bash
# 查看当前 skill 版本
python3 jimu-dse/scripts/skillctl.py list

# 预览/执行安全清理；默认保留 venv 和全部历史运行
bash jimu-dse/scripts/clean_workspace.sh
bash jimu-dse/scripts/clean_workspace.sh --apply

# 完整验证
make kernels
python3 -m pytest tests --ignore=tests/integration -q
python3 -m pytest tests/integration/test_bert_e2e.py -q -rs
```
