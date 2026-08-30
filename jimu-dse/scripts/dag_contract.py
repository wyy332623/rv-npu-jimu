#!/usr/bin/env python3
"""Select one deterministic DAG implementation contract for an agent turn."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LEVEL_ORDER = {"L1": 1, "L2": 2, "L3": 3}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eligible(macro: dict[str, Any]) -> bool:
    allocation = macro.get("allocation", {})
    return bool(
        macro.get("eligible")
        and macro.get("implementation_ready", macro.get("eligible"))
        and allocation.get("allocation_proven")
        and allocation.get("cross_config_proven")
        and allocation.get("validation_matrix_complete")
    )


def build_contract(dag_dir: Path) -> dict[str, Any]:
    macro_path = dag_dir / "macro_candidates.json"
    allocation_path = dag_dir / "allocation_proof.json"
    macros = json.loads(macro_path.read_text(encoding="utf-8"))
    metadata_path = dag_dir / "multiseq_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    workload = metadata.get("workload", {})
    target_file = workload.get("target_file") or "firmware/bert/bert_layer.c"
    eligible = [macro for macro in macros.get("macros", []) if _eligible(macro)]
    eligible.sort(
        key=lambda macro: (
            LEVEL_ORDER.get(str(macro.get("level")), 99),
            int(macro.get("rank", 9999)),
            -int(macro.get("priority_score", 0)),
            str(macro.get("id")),
        )
    )
    selected = eligible[0] if eligible else None
    contract: dict[str, Any] = {
        "schema": {"name": "jimu-next-macro-contract", "version": "1.0.0"},
        "status": "ready" if selected else "blocked-no-eligible-scope",
        "selection_policy": "lowest eligible level, then generated rank",
        "evidence_identity": {
            "macro_candidates_sha256": _sha256(macro_path),
            "allocation_proof_sha256": (
                _sha256(allocation_path) if allocation_path.is_file() else None
            ),
        },
        "agent_policy": {
            "authoritative_input": "next_macro_contract.json",
            "edit_scope": f"{target_file} only",
            "do_not_reprove": True,
            "do_not_regenerate_dag": True,
            "do_not_read_framework_source": True,
            "required_action": (
                "implement the selected source invariants, declare exactly "
                "the selected macro ID, run the supplied verification command"
            ),
        },
        "selected_macro": None,
        "blocked_macros": [
            {
                "id": macro.get("id"),
                "level": macro.get("level"),
                "status": macro.get("status"),
                "missing_proofs": macro.get("allocation", {}).get(
                    "missing_proofs", []
                ),
            }
            for macro in macros.get("macros", [])
            if not _eligible(macro)
        ],
    }
    if selected is None:
        return contract

    allocation = selected.get("allocation", {})
    contract["selected_macro"] = {
        "id": selected["id"],
        "level": selected.get("level"),
        "family": selected.get("family"),
        "status": selected.get("status"),
        "expected_dram_resources": selected.get(
            "expected_dram_resources", []
        ),
        "estimated_saving": selected.get("estimated_saving", {}),
        "required_source_invariants": allocation.get(
            "required_source_invariants", []
        ),
        "partial_sum_allocation": allocation.get(
            "partial_sum_allocation", {}
        ),
        "retained_output_allocation": allocation.get(
            "retained_output_allocation", {}
        ),
        "transient_allocation": allocation.get(
            "transient_allocation", {}
        ),
        "state_allocations": allocation.get("state_allocations", []),
        "l2_x_retention": allocation.get("l2_x_retention", {}),
        "synthesis_plan": allocation.get("synthesis_plan", {}),
        "schedule_signature": allocation.get("schedule_signature", {}),
        "reference_regions": allocation.get("regions", []),
        "configuration_proofs": allocation.get("config_results", []),
        "declaration": f"// JIMU_DAG_MACRO: {selected['id']}",
        "strict_scope": selected.get("scope_policy"),
    }
    return contract


def render_markdown(contract: dict[str, Any]) -> str:
    lines = ["# Next DAG Implementation Contract", ""]
    selected = contract.get("selected_macro")
    if not selected:
        lines.append("No eligible implementation scope is available.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            f"- Macro: `{selected['id']}`",
            f"- Level: `{selected['level']}`",
            f"- Declaration: `{selected['declaration']}`",
            "- The generated proof is authoritative; do not rebuild it in the agent turn.",
            "",
            "## Required source invariants",
            "",
        ]
    )
    invariants = selected.get("required_source_invariants", [])
    lines.extend(f"- {item}" for item in invariants)
    if not invariants:
        lines.append("- Follow the exact generated allocation regions.")
    lines.extend(
        [
            "",
            "## Exact DRAM scope",
            "",
            "| Tensor slice | Address |",
            "|---|---:|",
        ]
    )
    for resource in selected.get("expected_dram_resources", []):
        lines.append(
            f"| `{resource['tensor_slice']}` | "
            f"`{resource['dram_address_hex']}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_contract(dag_dir: Path, output: Path | None = None) -> dict[str, Path]:
    output = output or dag_dir / "next_macro_contract.json"
    summary = output.with_suffix(".md")
    contract = build_contract(dag_dir)
    output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary.write_text(render_markdown(contract), encoding="utf-8")
    return {"contract": output, "contract_summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dag-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name, path in write_contract(args.dag_dir, args.output).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
