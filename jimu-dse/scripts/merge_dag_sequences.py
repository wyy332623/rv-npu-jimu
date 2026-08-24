#!/usr/bin/env python3
"""Merge concrete structured DAGs into compact multi-sequence evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _sequence_dir(value: str) -> tuple[int, Path]:
    try:
        seq_text, path_text = value.split("=", 1)
        seq_len = int(seq_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "sequence DAG must use SEQ_LEN=PATH, for example 2=results/dag/seq2"
        ) from exc
    if seq_len <= 0 or not path_text:
        raise argparse.ArgumentTypeError("SEQ_LEN must be positive and PATH non-empty")
    return seq_len, Path(path_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two or more concrete sequence DAG directories"
    )
    parser.add_argument(
        "--dag",
        action="append",
        required=True,
        type=_sequence_dir,
        metavar="SEQ_LEN=PATH",
        help="structured DAG directory; repeat for every sequence length",
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--proof-dag",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="additional validation-configuration DAG; repeat as needed",
    )
    parser.add_argument(
        "--required-config",
        action="append",
        default=[],
        metavar="CONFIG_ID",
        help=(
            "required identity such as dim4-h8-head2-seq6; missing graphs "
            "block allocation proof"
        ),
    )
    args = parser.parse_args()

    sequence_dirs = dict(args.dag)
    if len(sequence_dirs) != len(args.dag):
        parser.error("duplicate sequence length")
    if len(sequence_dirs) < 2:
        parser.error("at least two --dag inputs are required")
    all_input_dirs = list(sequence_dirs.values()) + list(args.proof_dag)
    missing = [
        str(path / "run_metadata.json")
        for path in all_input_dirs
        if not (path / "run_metadata.json").is_file()
    ]
    if missing:
        parser.error("missing structured DAG input: " + ", ".join(missing))

    from emulator.npu_dag_structured import write_multiseq_dag

    paths = write_multiseq_dag(
        args.output,
        sequence_dirs,
        proof_dirs=args.proof_dag,
        required_config_ids=args.required_config or None,
    )
    from dag_contract import write_contract

    paths.update(write_contract(args.output))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
