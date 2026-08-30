# NPU 优化技能库

`jimu-dse/docs/skills/isa/*.md` 是技能及翻译文件的唯一真源。
每个技能必须在 YAML frontmatter 中声明：

```yaml
---
name: vrf-cache
version: 1.0.0
description: ...
---
```

文件名必须与 `name` 一致，`version` 使用语义化版本号。已归档版本的内容
不可原地修改；修改技能正文时必须先提升版本号，否则同步工具会报告版本冲突。
`<name>.zh.md` 翻译文件继承英文主技能的版本，并随主技能一起归档和同步。

## 目录

```text
jimu-dse/docs/skills/
  isa/                         当前生效的技能真源
  versions/<name>/<version>.md 不可变的版本快照
  skills.lock.json             当前版本与 SHA256 锁文件
.opencode/skills/<name>/SKILL.md
                               从真源自动生成的 OpenCode 副本
```

`common-constraints` 会自动放在每次运行的技能列表首位，禁止 Agent：

- 执行 `git stash/reset/checkout/restore/clean`；
- 修改测试、模拟器、ISS、硬件模型或验证命令；
- 修改目标固件以外的文件。

## 常用命令

```bash
# 归档当前版本、生成锁文件并同步所有 OpenCode 技能
python3 jimu-dse/scripts/skillctl.py sync

# 检查真源、版本快照、锁文件和 OpenCode 副本是否一致
python3 jimu-dse/scripts/skillctl.py verify

# 查看当前版本与可回滚版本
python3 jimu-dse/scripts/skillctl.py list

# 回滚指定技能；切换版本前会先归档有效的当前版本
python3 jimu-dse/scripts/skillctl.py rollback vrf-cache 1.0.0
```

也可以执行 `make opencode` 完成同步和校验。闭环脚本启动时会自动执行同步。

## 每次运行的技能记录

运行目录中会生成：

- `skills_manifest.json`：实际注入技能的名称、版本、SHA256、真源和归档路径；
- `skills_bundle.md`：PI 使用的完整合并技能；
- `run_manifest.json`：内嵌同一组技能元数据。

OpenCode 会通过多个 `-f` 参数显式获得所有实际技能；PI 只接受一个
`--skill` 参数，因此使用按运行生成的 `skills_bundle.md`。

每次运行固定使用以下顺序：

```text
common-constraints -> dag-analyze -> 目标优化技能 -> self-verify
```

可使用 `--prepare-only` 只生成 prompt、manifest 和合并技能，不执行 probe、
不调用 Agent，也不修改固件。

`vrf-cache` 2.1 采用分级优化：一次候选只能引入 L1（中间结果缓存）、
L2（循环不变量缓存）或 L3（权重驻留）中的一级。G1 会保存 seq2、seq6
修改前后的 probe JSON，并执行独立双序列指标门禁。
指令数门禁可通过 `JIMU_INSTR_GATE=on|off` 或
`--instruction-gate on|off` 控制；两种模式都会记录指令数。

## 修改流程

1. 修改 `jimu-dse/docs/skills/isa/<name>.md`。
2. 提升 frontmatter 中的语义化版本号。
3. 执行 `skillctl.py sync` 和 `skillctl.py verify`。
4. 提交真源、版本快照、锁文件以及生成的 OpenCode 副本。

回滚到当前版本时，会用已归档内容覆盖未提升版本号的临时修改；这可以用于快速
撤销调试中的 skill 改动。
