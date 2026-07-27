# Configurable Closed-Loop Firmware Optimization

The driver implements `PROBE → PROMPT → AGENT → GATE → SCORE → PROMOTE → LOOP`.
Goals, skills, acceptance, scoring, and convergence are declared in
`jimu-dse/goals/<name>/goal.yaml`.

## Install and inspect

```bash
pip install -r requirements.txt
make kernels
python3 jimu-dse/scripts/closed_loop.py list-goals
python3 jimu-dse/scripts/closed_loop.py validate-config --goal dram-optimization
python3 jimu-dse/scripts/closed_loop.py render-prompt --goal dram-optimization
```

`render-prompt` is a dry run: it validates configuration and renders skills
without invoking an agent or changing firmware.

## Run and resume

```bash
# Compatibility entry point
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal dram-optimization --agent opencode

# Direct CLI
python3 jimu-dse/scripts/closed_loop.py run \
  --goal combined --agent opencode --model provider/model

python3 jimu-dse/scripts/closed_loop.py inspect-run \
  jimu-dse/results/run-YYYYMMDD-HHMMSS-PID

python3 jimu-dse/scripts/closed_loop.py run --goal combined \
  --resume jimu-dse/results/run-YYYYMMDD-HHMMSS-PID
```

Resume loads `candidate_best.c` and requires the selected goal's fully resolved
configuration fingerprint to match. This prevents accidentally continuing a run
with different hardware, gates, skills, or scoring.

## Configuration precedence

Highest to lowest:

1. Explicit CLI values (`--agent`, `--model`).
2. Compatibility environment variables (`JIMU_MAX_ITER`,
   `JIMU_AGENT_TIMEOUT`, `OPENCODE_MODEL`).
3. Values in `goal.yaml`.

Every run stores the final value set in `resolved-config.yaml`.

## `goal.yaml` reference

All roots and sections reject unknown fields. Every repository path is resolved
against the repository root and may not escape it.

| Section | Fields |
|---|---|
| root | `schema_version`, `name`, `description` |
| `target` | `firmware`, `baseline`, `allowed_files`, exact `hardware` values, `sequence_lengths` |
| `agent` | `backend` (`pi`/`opencode`), `model`, `timeout_seconds`, `context_files` |
| `prompt` | `template`, `goal`, `constraints`, `self_verify` |
| `skills` | ordered `{name, path}` entries |
| `probe` | built-in `metrics`, `cycle_limit`, `dag.enabled` |
| `acceptance.gates` | named `allowed_files`, `build`, `probe`, or `command` gates |
| `acceptance.score` | `{metric, direction, weight, target?}` entries; weights sum to `1.0` |
| `loop` | `max_iterations`, `min_score_delta`, `max_no_improvement`, optional `target_score` |
| `artifacts` | switches for candidates, diffs, prompts, probes, and graphs |

Built-in metrics are `total_bytes`, `instr_count`, `mv_mul_count`,
`mat_rd_ops`, and `test_pass`. A score metric must also appear in
`probe.metrics`; directions are `minimize` or `maximize`.

Command gates support:

```yaml
- name: numerical-correctness
  type: command
  command: python3 -m pytest tests/integration/test_bert_e2e.py -k "dim{dim} and h{hidden}" -q
  timeout_seconds: 600
  success_codes: [0]
```

Command templates expose `{dim}`, `{hidden}`, and `{firmware}`.

## Prompt templates

The following fields are allowed:

| Variable | Content |
|---|---|
| `{goal_name}`, `{goal_description}` | Goal identity and optimization intent |
| `{iteration}`, `{target_file}`, `{hardware}` | Current execution context |
| `{metrics}`, `{clusters}` | Latest probe results and DAG cluster text |
| `{skills}` | Full ordered skill instructions |
| `{constraints}`, `{self_verify}` | Goal constraints and verification guidance |
| `{gate_commands}` | Configured gate commands/types |

Unknown variables fail configuration validation before a run starts.

## Acceptance and convergence

A candidate is never promoted unless every hard gate passes. Eligible
candidates are scored relative to the run-start baseline:

- minimize: `(baseline - candidate) / abs(baseline)`
- maximize: `(candidate - baseline) / abs(baseline)`
- total score: sum of normalized improvements multiplied by their weights

The candidate becomes the new best only when its score exceeds the best score
by at least `min_score_delta`. Otherwise the working firmware is restored to the
current best. The loop stops at `target_score`, after
`max_no_improvement` unsuccessful rounds, or at `max_iterations`.
The repository's original firmware is restored when the run exits.

## Creating a goal

1. Copy an existing directory under `jimu-dse/goals/`.
2. Change `name`, description, target hardware, prompt, ordered skills, gates,
   score weights, and loop thresholds.
3. Ensure every referenced path exists and all weights sum to `1.0`.
4. Run `validate-config`, then inspect `render-prompt`.
5. Start with `JIMU_MAX_ITER=1` and review the generated report before a longer run.

No Python or shell driver change is required for a new declarative goal.

## Run artifacts

Each `jimu-dse/results/run-*` directory contains:

- `resolved-config.yaml` and `baseline-probe.json`
- `prompt-N.txt`, `candidate-N.c`, `diff-N.patch`, and graph directories
- `iteration-N.json` with agent status, gates, raw metrics, score, and promotion
- `candidate_best.c`
- `run-summary.json` and human-readable `report.md`

Agent unavailable, timeout, no change, build failure, probe failure, gate
failure, and lack of score improvement are represented as distinct statuses.

## Troubleshooting

- `No module named yaml`: run `pip install -r requirements.txt`.
- Missing cross compiler: install `riscv64-unknown-elf-gcc` and set `CC` if needed.
- Invalid goal: run `validate-config`; it reports the exact field or path.
- Candidate rejected: inspect the relevant `iteration-N.json` gate output.
- Resume rejected: use the same resolved goal and overrides as the original run.
