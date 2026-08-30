---
name: common-constraints
version: 1.0.0
description: Mandatory repository and validation safety constraints for every JIMU optimization run
license: MIT
---

# Common Agent Constraints

These constraints apply to every optimization skill and override conflicting
instructions in later skills.

## Allowed Changes

- Modify only the target firmware source named in the run prompt.
- Read other project files as needed for analysis.
- Use read-only Git commands such as `git status`, `git diff`, `git log`, and
  `git show`.

## Forbidden Repository Operations

- Never run `git stash`, `git reset`, or `git checkout`.
- Never run equivalent destructive or state-restoring operations, including
  `git restore`, `git clean`, or commands that overwrite unrelated working-tree
  changes.
- Never commit, amend, rebase, merge, or switch branches during an optimization
  run.

## Forbidden Validation Changes

- Never modify files under `tests/`.
- Never modify the emulator, ISS, trace recorder, or hardware model, including
  files under `emulator/` and `iss/`.
- Never weaken tolerances, remove test cases, add skips/xfails, change golden
  outputs, or alter the validation command.
- Never report success from filtered console text. The independent acceptance
  gate and its exit status are authoritative.

If a requested optimization appears to require a forbidden change, leave the
firmware unchanged and explain the blocker.
