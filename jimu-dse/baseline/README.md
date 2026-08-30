# Canonical Optimization Baselines

Files in this directory are known-correct, unoptimized firmware references.
They are the default starting point for a fresh closed-loop run and are restored
to the working firmware path when a run exits.

Do not promote an optimized candidate by overwriting these files. To continue
optimization from an earlier result, select that result explicitly:

```bash
# Continue from the best result of a run.
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization \
  --start-from jimu-dse/results/run-<timestamp>

# Continue from a specific iteration.
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization \
  --start-from jimu-dse/results/run-<timestamp>/candidate_4.c
```

Each new run copies its selected source to
`results/run-*/optimization_baseline.c`. Metrics and diffs are relative to
that immutable per-run snapshot. `run_manifest.json` records both the
canonical baseline and the selected optimization starting source with SHA256.
