# NPU Optimization Skill Library

Skills are Markdown instructions loaded by the configurable firmware
optimization loop. A goal references skills in application order; both `pi`
and `opencode` receive that same ordered list.

## Required format

Every skill starts with YAML front matter:

```markdown
---
name: vrf-cache
version: 1.0.0
category: isa
description: Replace DRAM round trips with on-chip VRF caching.
compatibility:
  schema_version: 1
---
```

`compatibility` is optional. The body should document applicability or trigger
conditions, transformation steps, constraints, verification, and expected
impact. Skill names must be unique inside a goal and must match the `name`
declared in front matter.

## Adding a skill

1. Add its Markdown file under the appropriate category.
2. Add the required front matter and operational instructions.
3. Reference it from `skills` in a goal's `goal.yaml`.
4. Run `python3 jimu-dse/scripts/closed_loop.py validate-config --goal NAME`.
5. Run `make opencode` if the skill should also be installed in OpenCode's
   project skill directory.

The Markdown source under this directory is authoritative; generated
`.opencode/skills/*/SKILL.md` files are disposable copies.
