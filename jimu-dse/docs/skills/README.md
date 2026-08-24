# NPU Optimization Skill Library

This directory contains skill definitions for the closed-loop FW-HW
co-optimization system. Each skill describes an optimization pattern
that an agent can apply to the NPU firmware.

## Directory Structure

```
skills/
  isa/                       # Active source of truth
    common-constraints.md    # Mandatory safety policy for every run
    vrf-cache.md             # G1 transformation
    dim-optimize.md          # G2/G3 transformation
  versions/<name>/<ver>.md   # Immutable rollback snapshots
  skills.lock.json           # Active version and SHA256 lock
```

## Skill Format

Each canonical English skill is a Markdown file with YAML frontmatter.
Optional `<name>.zh.md` translations inherit the canonical skill version and
are archived and synchronized with it. Required canonical fields:

- **name**: Unique identifier
- **version**: Semantic version (`MAJOR.MINOR.PATCH`)
- **description**: One-line purpose

The filename must match `name`. Never change a released version without
bumping its version; `skillctl` rejects such version collisions.

## Usage

```bash
# Archive active versions, update the lock, and publish to .opencode/skills/.
python3 jimu-dse/scripts/skillctl.py sync

# Confirm source, archive, lock, and OpenCode copies agree.
python3 jimu-dse/scripts/skillctl.py verify

# Inspect active and archived versions.
python3 jimu-dse/scripts/skillctl.py list

# Restore an archived version to the canonical source and republish it.
python3 jimu-dse/scripts/skillctl.py rollback vrf-cache 1.0.0
```

The closed-loop script runs `sync` automatically before every run. It creates
`skills_manifest.json`, `skills_bundle.md`, and embeds the ordered skill name,
version, and SHA256 data in `run_manifest.json`.

OpenCode receives every effective skill as an explicit `-f` input. PI supports
one `--skill` argument, so it receives the generated bundle. Skills are always
ordered as:

```text
common-constraints -> dag-analyze -> goal skills -> self-verify
```

Use `--prepare-only` to inspect the prompt, manifests, and exact bundle without
probing, invoking an agent, or modifying firmware.

`vrf-cache` 2.3 is staged: one candidate may introduce only one of L1
(intermediate caching), L2 (loop-invariant caching), or L3
(weight-stationary scheduling). G1 records before/after seq2 and seq6 probe
JSON files and applies the independent two-sequence metric gate.
The optional instruction-count part is controlled by `JIMU_INSTR_GATE=on|off`
or `--instruction-gate on|off`; counts are recorded in both modes.

`dag-analyze` 1.7 reads concrete seq2 and seq6 DAGs plus the complete BERT
validation matrix. DAG-PR6 emits `allocation_proof.json`; L1/L2 are selectable
only with complete cross-configuration proofs. L3 remains blocked until its
schedule, MRF, partial-sum and FP16-order proof is implemented.

## Versioning Workflow

1. Edit only `jimu-dse/docs/skills/isa/<name>.md`.
2. Bump its semantic `version`.
3. Run `skillctl.py sync` and `skillctl.py verify`.
4. Commit the source, snapshot, lock file, and generated OpenCode copy.

When switching versions, rollback archives the valid current version before
restoring the requested snapshot. Rolling back to the same version restores its
archived content and discards unversioned edits to that active file.
