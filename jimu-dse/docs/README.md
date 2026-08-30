# JIMU-DSE documentation

Use this page as the documentation entry point. Current operational documents are kept separate from dated reports and historical repository analyses.

## Current documents

- [Current project status (Chinese)](project-status.zh.md)
- [How to run](how-to-run.md) / [中文](how-to-run.zh.md)
- [Structured DAG artifacts](structured-dag.md) / [中文](structured-dag.zh.md)
- [Closed-loop design](npu-closed-loop-design.md) / [中文](npu-closed-loop-design.zh.md)
- [Skill management](skills/README.md) / [中文](skills/README.zh.md)

The base NPU architecture, ISA, firmware, build, and test documents remain in the repository-level [`docs/`](../../docs/).

## Evidence priority

When text and implementation disagree, prefer current source and independent tests, then machine-readable manifests/DAG artifacts and hashes, then current documents, dated reports, and finally archived analyses or agent narratives.

`jimu-dse/baseline/bert_layer.c` is the canonical correctness baseline. A run-local `candidate_best.c` is evidence from one search and is never promoted automatically.

## Other sections

- [`reports/`](reports/README.zh.md): dated technical analyses with explicit assumptions.
- [`archive/`](archive/README.zh.md): point-in-time upstream comparisons and early reports.
- [`../results/`](../results/README.md): local generated run evidence; new runs are ignored by Git.

## Workspace maintenance

```bash
# Preview first; both commands preserve venv and run history by default.
bash jimu-dse/scripts/clean_workspace.sh
bash jimu-dse/scripts/clean_workspace.sh --apply
```
