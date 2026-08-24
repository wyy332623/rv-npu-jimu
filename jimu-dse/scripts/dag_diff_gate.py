#!/usr/bin/env python3
"""Compare baseline/candidate DAGs and validate an Agent candidate claim."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


CANDIDATE_MARKER = re.compile(
    r"JIMU_DAG_CANDIDATE\s*:\s*(candidate-dram-\d+)\b"
)
MACRO_MARKER = re.compile(
    r"JIMU_DAG_MACRO\s*:\s*(macro-dram-[a-z0-9-]+)\b"
)


def _kind_counts(micro_ops: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(node.get("kind", "UNKNOWN")) for node in micro_ops)


def _tensor_dram_counts(
    micro_ops: list[dict[str, Any]],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {}
    for node in micro_ops:
        for access_kind, resources in (
            ("DRAM_STORE", node.get("defs", [])),
            ("DRAM_LOAD", node.get("uses", [])),
        ):
            for resource in resources:
                if resource.get("space") != "DRAM":
                    continue
                tensor_slice = resource.get("slice")
                if not tensor_slice:
                    address = resource.get(
                        "address_hex", resource.get("address")
                    )
                    tensor_slice = f"DRAM[{address}]"
                counts.setdefault(str(tensor_slice), Counter())[access_kind] += 1
    return counts


def _resource_dram_counts(
    micro_ops: list[dict[str, Any]],
) -> dict[tuple[str, int], Counter[str]]:
    """Count traffic by exact structured Tensor slice and DRAM address."""
    counts: dict[tuple[str, int], Counter[str]] = {}
    for node in micro_ops:
        for access_kind, resources in (
            ("DRAM_STORE", node.get("defs", [])),
            ("DRAM_LOAD", node.get("uses", [])),
        ):
            for resource in resources:
                if resource.get("space") != "DRAM":
                    continue
                address = int(resource.get("address", -1))
                tensor_slice = resource.get("slice")
                if not tensor_slice:
                    tensor_slice = f"DRAM[0x{address:x}]"
                key = (str(tensor_slice), address)
                counts.setdefault(key, Counter())[access_kind] += 1
    return counts


def _metric_saving(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    before_bytes = int(before.get("total_bytes", 0))
    after_bytes = int(after.get("total_bytes", 0))
    return {
        "before_total_bytes": before_bytes,
        "after_total_bytes": after_bytes,
        "saved_bytes": before_bytes - after_bytes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vrf_pressure(lifetimes: list[dict[str, Any]] | None) -> dict[str, int] | None:
    if lifetimes is None:
        return None
    events: list[tuple[int, int]] = []
    resources: set[tuple[Any, Any]] = set()
    for lifetime in lifetimes:
        interval = lifetime.get("interval", {})
        resource = lifetime.get("resource", {})
        start = int(interval.get("start_index", 0))
        end = int(interval.get("end_index", start))
        events.append((start, 1))
        events.append((end + 1, -1))
        resources.add((resource.get("bank"), resource.get("row")))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return {
        "lifetime_count": len(lifetimes),
        "unique_resources": len(resources),
        "peak_live_ranges": peak,
    }


def evaluate_dag_evidence(
    before_candidates: dict[str, Any],
    before_micro_ops: list[dict[str, Any]],
    after_micro_ops: list[dict[str, Any]],
    candidate_source: str,
    *,
    before_macros: dict[str, Any] | None = None,
    before_seq2: dict[str, Any] | None = None,
    before_seq6: dict[str, Any] | None = None,
    after_seq2: dict[str, Any] | None = None,
    after_seq6: dict[str, Any] | None = None,
    before_edges: list[dict[str, Any]] | None = None,
    after_edges: list[dict[str, Any]] | None = None,
    before_lifetimes: list[dict[str, Any]] | None = None,
    after_lifetimes: list[dict[str, Any]] | None = None,
    before_metadata: dict[str, Any] | None = None,
    after_metadata: dict[str, Any] | None = None,
    before_micro_ops_sha256: str | None = None,
    after_micro_ops_sha256: str | None = None,
    candidate_required: bool = True,
    gate_enabled: bool = True,
) -> dict[str, Any]:
    """Return structural and measured evidence for one optimization iteration."""
    failures: list[str] = []
    warnings: list[str] = []
    candidate_markers = CANDIDATE_MARKER.findall(candidate_source)
    macro_markers = MACRO_MARKER.findall(candidate_source)
    markers = candidate_markers + macro_markers
    selected_id = markers[0] if len(markers) == 1 else None
    selected_kind = (
        "macro" if len(macro_markers) == 1 and len(markers) == 1 else
        "primitive" if len(candidate_markers) == 1 and len(markers) == 1 else
        None
    )

    if candidate_required:
        if not markers:
            if before_macros is None:
                failures.append("missing JIMU_DAG_CANDIDATE declaration")
            else:
                failures.append(
                    "missing JIMU_DAG_MACRO or JIMU_DAG_CANDIDATE declaration"
                )
        elif len(markers) != 1:
            failures.append(
                "candidate source must contain exactly one DAG declaration "
                "(JIMU_DAG_MACRO or JIMU_DAG_CANDIDATE)"
            )

    candidates = {
        candidate.get("id"): candidate
        for candidate in before_candidates.get("candidates", [])
    }
    macros = {
        macro.get("id"): macro
        for macro in (before_macros or {}).get("macros", [])
    }
    selected = (
        macros.get(selected_id)
        if selected_kind == "macro"
        else candidates.get(selected_id)
    )
    if selected_id is not None and selected is None:
        if selected_kind == "primitive":
            message = (
                "declared candidate is not eligible in baseline DAG: "
                f"{selected_id}"
            )
        else:
            message = (
                f"declared {selected_kind or 'candidate'} is not present in "
                f"baseline DAG: {selected_id}"
            )
        if candidate_required:
            failures.append(message)
        else:
            warnings.append(message)
    if selected_kind == "macro" and selected is not None:
        allocation = selected.get("allocation", {})
        if (
            not selected.get("eligible")
            or not allocation.get("allocation_proven")
            or not allocation.get("cross_config_proven")
            or not allocation.get("validation_matrix_complete")
        ):
            message = (
                "declared macro has no complete cross-config VRF allocation: "
                f"{selected_id}"
            )
            if candidate_required:
                failures.append(message)
            else:
                warnings.append(message)
        level_rank = {"L1": 1, "L2": 2, "L3": 3}
        selected_level = str(selected.get("level", "L1"))
        lower_eligible = [
            macro.get("id")
            for macro in macros.values()
            if macro.get("eligible")
            and level_rank.get(str(macro.get("level", "L1")), 99)
            < level_rank.get(selected_level, 99)
        ]
        if level_rank.get(selected_level, 99) > 1:
            lower_eligible.extend(
                str(candidate.get("id"))
                for candidate in candidates.values()
                if candidate.get("eligible")
            )
        if lower_eligible:
            message = (
                f"declared {selected_level} macro before lower level is complete: "
                + ", ".join(sorted(map(str, lower_eligible)))
            )
            if candidate_required:
                failures.append(message)
            else:
                warnings.append(message)
    elif selected_kind == "primitive" and selected is not None:
        eligible_l1 = [
            macro.get("id")
            for macro in macros.values()
            if macro.get("eligible") and str(macro.get("level", "L1")) == "L1"
        ]
        if eligible_l1:
            message = (
                "primitive fallback declared while an eligible L1 macro exists: "
                + ", ".join(sorted(map(str, eligible_l1)))
            )
            if candidate_required:
                failures.append(message)
            else:
                warnings.append(message)

    before_kinds = _kind_counts(before_micro_ops)
    after_kinds = _kind_counts(after_micro_ops)
    all_kinds = sorted(set(before_kinds) | set(after_kinds))
    kind_delta = {
        kind: {
            "before": before_kinds[kind],
            "after": after_kinds[kind],
            "removed": before_kinds[kind] - after_kinds[kind],
        }
        for kind in all_kinds
    }

    before_tensors = _tensor_dram_counts(before_micro_ops)
    after_tensors = _tensor_dram_counts(after_micro_ops)
    tensor_delta = []
    for tensor_slice in sorted(set(before_tensors) | set(after_tensors)):
        before_access = before_tensors.get(tensor_slice, Counter())
        after_access = after_tensors.get(tensor_slice, Counter())
        record = {
            "tensor_slice": tensor_slice,
            "before_store_ops": before_access["DRAM_STORE"],
            "after_store_ops": after_access["DRAM_STORE"],
            "removed_store_ops": (
                before_access["DRAM_STORE"] - after_access["DRAM_STORE"]
            ),
            "before_load_ops": before_access["DRAM_LOAD"],
            "after_load_ops": after_access["DRAM_LOAD"],
            "removed_load_ops": (
                before_access["DRAM_LOAD"] - after_access["DRAM_LOAD"]
            ),
        }
        if record["removed_store_ops"] or record["removed_load_ops"]:
            tensor_delta.append(record)

    before_resources = _resource_dram_counts(before_micro_ops)
    after_resources = _resource_dram_counts(after_micro_ops)
    resource_delta = []
    resource_delta_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for key in sorted(set(before_resources) | set(after_resources)):
        before_access = before_resources.get(key, Counter())
        after_access = after_resources.get(key, Counter())
        record = {
            "tensor_slice": key[0],
            "dram_address": key[1],
            "dram_address_hex": f"0x{key[1]:x}",
            "before_store_ops": before_access["DRAM_STORE"],
            "after_store_ops": after_access["DRAM_STORE"],
            "removed_store_ops": (
                before_access["DRAM_STORE"] - after_access["DRAM_STORE"]
            ),
            "before_load_ops": before_access["DRAM_LOAD"],
            "after_load_ops": after_access["DRAM_LOAD"],
            "removed_load_ops": (
                before_access["DRAM_LOAD"] - after_access["DRAM_LOAD"]
            ),
        }
        record["structural_reduction"] = (
            record["removed_store_ops"] >= 0
            and record["removed_load_ops"] >= 0
            and record["removed_store_ops"] + record["removed_load_ops"] > 0
        )
        resource_delta_by_key[key] = record
        if record["removed_store_ops"] or record["removed_load_ops"]:
            resource_delta.append(record)

    selected_tensor_delta = None
    selected_resource_delta: list[dict[str, Any]] = []
    if selected is not None and selected_kind == "primitive":
        tensor_slice = str(selected.get("tensor_slice"))
        before_access = before_tensors.get(tensor_slice, Counter())
        after_access = after_tensors.get(tensor_slice, Counter())
        removed_stores = (
            before_access["DRAM_STORE"] - after_access["DRAM_STORE"]
        )
        removed_loads = (
            before_access["DRAM_LOAD"] - after_access["DRAM_LOAD"]
        )
        selected_tensor_delta = {
            "tensor_slice": tensor_slice,
            "before_store_ops": before_access["DRAM_STORE"],
            "after_store_ops": after_access["DRAM_STORE"],
            "removed_store_ops": removed_stores,
            "before_load_ops": before_access["DRAM_LOAD"],
            "after_load_ops": after_access["DRAM_LOAD"],
            "removed_load_ops": removed_loads,
            "structural_reduction": removed_stores + removed_loads > 0,
        }
        if not selected_tensor_delta["structural_reduction"]:
            observed_reductions = [
                delta["tensor_slice"]
                for delta in tensor_delta
                if delta["removed_store_ops"] > 0
                or delta["removed_load_ops"] > 0
            ]
            if before_micro_ops == after_micro_ops:
                detail = (
                    "; before/after micro-op DAGs are identical, so the "
                    "baseline evidence may have been overwritten"
                )
            elif observed_reductions:
                detail = (
                    "; observed reductions belong to: "
                    + ", ".join(observed_reductions)
                )
            else:
                detail = "; no Tensor DRAM-op reduction was observed"
            message = (
                "declared candidate has no matching Tensor DRAM-op reduction"
                + detail
            )
            if candidate_required:
                failures.append(message)
            else:
                warnings.append(message)
    elif selected is not None and selected_kind == "macro":
        expected_keys = {
            (str(resource["tensor_slice"]), int(resource["dram_address"]))
            for resource in selected.get("expected_dram_resources", [])
        }
        for key in sorted(expected_keys):
            delta = resource_delta_by_key.get(
                key,
                {
                    "tensor_slice": key[0],
                    "dram_address": key[1],
                    "dram_address_hex": f"0x{key[1]:x}",
                    "before_store_ops": 0,
                    "after_store_ops": 0,
                    "removed_store_ops": 0,
                    "before_load_ops": 0,
                    "after_load_ops": 0,
                    "removed_load_ops": 0,
                    "structural_reduction": False,
                },
            )
            selected_resource_delta.append(delta)
            if not delta["structural_reduction"]:
                failures.append(
                    "declared macro member has no exact-address DRAM-op "
                    f"reduction: {key[0]} @ 0x{key[1]:x}"
                )
        outside_reductions = [
            delta
            for key, delta in resource_delta_by_key.items()
            if key not in expected_keys and delta["structural_reduction"]
        ]
        if outside_reductions:
            detail = ", ".join(
                f"{delta['tensor_slice']}@{delta['dram_address_hex']}"
                for delta in outside_reductions[:8]
            )
            failures.append(
                "declared macro reduced DRAM resources outside its strict "
                f"scope: {detail}"
            )

    seq2 = _metric_saving(before_seq2, after_seq2)
    seq6 = _metric_saving(before_seq6, after_seq6)
    if candidate_required:
        if seq2 is None or seq6 is None:
            failures.append("candidate metric evidence is incomplete")
        else:
            if (
                seq2["before_total_bytes"] <= 0
                or seq2["after_total_bytes"] <= 0
                or seq6["before_total_bytes"] <= 0
                or seq6["after_total_bytes"] <= 0
            ):
                failures.append("candidate metric evidence contains zero bytes")
            if seq2["saved_bytes"] < 0:
                failures.append("seq2 measured DRAM traffic regressed")
            if seq6["saved_bytes"] <= 0:
                failures.append("seq6 measured DRAM traffic did not improve")

    estimate = None
    if selected is not None:
        saving = selected.get("estimated_saving", {})
        projected_seq2 = int(saving.get("projected_seq2_bytes", 0))
        projected_seq6 = int(saving.get("projected_seq6_bytes", 0))
        estimate = {
            "projected_seq2_bytes": projected_seq2,
            "projected_seq6_bytes": projected_seq6,
            "projection_assumption": saving.get("projection_assumption"),
            "actual_to_projected_seq6_ratio": (
                seq6["saved_bytes"] / projected_seq6
                if seq6 is not None and projected_seq6 > 0
                else None
            ),
        }
        if candidate_required and projected_seq6 <= 0:
            failures.append("declared candidate has no positive seq6 estimate")
        ratio = estimate["actual_to_projected_seq6_ratio"]
        if ratio is not None and (ratio < 0.5 or ratio > 1.5):
            warnings.append(
                "measured seq6 saving differs from the PR3 estimate by more "
                "than 50%; inspect the heuristic and DAG delta"
            )

    consistency_pass = not failures
    before_config = (
        before_metadata.get("firmware_config") if before_metadata else None
    )
    after_config = (
        after_metadata.get("firmware_config") if after_metadata else None
    )
    if before_config is not None and after_config is not None:
        if before_config != after_config:
            failures.append("before/after DAG firmware_config differs")
            consistency_pass = False
    before_edge_types = Counter(
        str(edge.get("type", "UNKNOWN")) for edge in (before_edges or [])
    )
    after_edge_types = Counter(
        str(edge.get("type", "UNKNOWN")) for edge in (after_edges or [])
    )
    edge_type_delta = {
        edge_type: {
            "before": before_edge_types[edge_type],
            "after": after_edge_types[edge_type],
            "removed": (
                before_edge_types[edge_type] - after_edge_types[edge_type]
            ),
        }
        for edge_type in sorted(set(before_edge_types) | set(after_edge_types))
    }
    return {
        "schema": {
            "name": "jimu-npu-dag-diff-gate",
            "version": "1.2.0",
        },
        "gate_enabled": gate_enabled,
        "candidate_required": candidate_required,
        "evidence_pass": consistency_pass if gate_enabled else True,
        "consistency_pass": consistency_pass,
        "failure_reasons": failures,
        "warnings": warnings,
        "declaration": {
            "markers": markers,
            "candidate_markers": candidate_markers,
            "macro_markers": macro_markers,
            "selected_kind": selected_kind,
            "selected_candidate_id": (
                selected_id if selected_kind == "primitive" else None
            ),
            "selected_macro_id": (
                selected_id if selected_kind == "macro" else None
            ),
            "valid_unique_marker": len(markers) == 1,
        },
        "selected_candidate": selected,
        "selected_tensor_delta": selected_tensor_delta,
        "selected_resource_delta": selected_resource_delta,
        "tensor_delta": tensor_delta,
        "resource_delta": resource_delta,
        "metric_evidence": {
            "seq2": seq2,
            "seq6": seq6,
            "estimate": estimate,
        },
        "dag_delta": {
            "before_node_count": len(before_micro_ops),
            "after_node_count": len(after_micro_ops),
            "removed_node_count": len(before_micro_ops) - len(after_micro_ops),
            "kind_delta": kind_delta,
            "before_edge_count": (
                len(before_edges) if before_edges is not None else None
            ),
            "after_edge_count": (
                len(after_edges) if after_edges is not None else None
            ),
            "edge_type_delta": edge_type_delta,
            "vrf_pressure": {
                "before": _vrf_pressure(before_lifetimes),
                "after": _vrf_pressure(after_lifetimes),
            },
        },
        "artifact_identity": {
            "before": {
                "firmware_config": before_config,
                "elf_sha256": (
                    before_metadata.get("elf", {}).get("sha256")
                    if before_metadata
                    else None
                ),
                "micro_ops_sha256": before_micro_ops_sha256,
            },
            "after": {
                "firmware_config": after_config,
                "elf_sha256": (
                    after_metadata.get("elf", {}).get("sha256")
                    if after_metadata
                    else None
                ),
                "micro_ops_sha256": after_micro_ops_sha256,
            },
            "micro_ops_identical": before_micro_ops == after_micro_ops,
        },
        "analysis_limits": {
            "candidate_identity": (
                "candidate IDs are valid only against the supplied baseline DAG"
            ),
            "tensor_matching": "exact structured Tensor slice",
            "macro_matching": "exact structured Tensor slice and DRAM address",
            "macro_scope": (
                "every member must decrease and reductions outside the "
                "declared macro are rejected"
            ),
            "estimate_policy": (
                "estimate deviation is reported as a warning; measured traffic "
                "and structural direction are authoritative"
            ),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    status = "PASS" if result["evidence_pass"] else "FAIL"
    declaration = result["declaration"]
    selected_id = (
        declaration.get("selected_macro_id")
        or declaration.get("selected_candidate_id")
        or "-"
    )
    lines = [
        "# DAG Evidence Gate",
        "",
        f"Result: **{status}**",
        "",
        f"- Gate enabled: `{str(result['gate_enabled']).lower()}`",
        f"- Candidate required: `{str(result['candidate_required']).lower()}`",
        f"- Selected kind: `{declaration.get('selected_kind') or '-'}`",
        f"- Selected candidate: `{selected_id}`",
        f"- Raw consistency: `{str(result['consistency_pass']).lower()}`",
    ]
    tensor_delta = result.get("selected_tensor_delta")
    if tensor_delta is not None:
        lines.extend(
            [
                f"- Tensor: `{tensor_delta['tensor_slice']}`",
                (
                    "- Tensor DRAM ops removed: "
                    f"store={tensor_delta['removed_store_ops']}, "
                    f"load={tensor_delta['removed_load_ops']}"
                ),
            ]
        )
    selected_resources = result.get("selected_resource_delta", [])
    if selected_resources:
        reduced = sum(
            1 for resource in selected_resources
            if resource["structural_reduction"]
        )
        lines.append(
            f"- Macro resources reduced: {reduced}/{len(selected_resources)}"
        )
    metrics = result["metric_evidence"]
    for sequence in ("seq2", "seq6"):
        metric = metrics.get(sequence)
        if metric is not None:
            lines.append(
                f"- {sequence} DRAM: {metric['before_total_bytes']} B -> "
                f"{metric['after_total_bytes']} B "
                f"(saved {metric['saved_bytes']} B)"
            )
    estimate = metrics.get("estimate")
    if estimate is not None:
        lines.append(
            "- PR3 estimate: "
            f"seq2={estimate['projected_seq2_bytes']} B, "
            f"seq6={estimate['projected_seq6_bytes']} B"
        )
    identity = result.get("artifact_identity", {})
    before_identity = identity.get("before", {})
    after_identity = identity.get("after", {})
    if before_identity.get("micro_ops_sha256"):
        lines.extend(
            [
                "- Before DAG SHA256: "
                f"`{before_identity['micro_ops_sha256']}`",
                "- After DAG SHA256: "
                f"`{after_identity['micro_ops_sha256']}`",
                "- Micro-op DAGs identical: "
                f"`{str(identity.get('micro_ops_identical')).lower()}`",
            ]
        )
    if result["failure_reasons"]:
        lines.extend(["", "## Failure reasons", ""])
        lines.extend(f"- {reason}" for reason in result["failure_reasons"])
    if result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    lines.extend(
        [
            "",
            "## Micro-op delta",
            "",
            "| Kind | Before | After | Removed |",
            "|---|---:|---:|---:|",
        ]
    )
    for kind, delta in result["dag_delta"]["kind_delta"].items():
        if delta["removed"] != 0 or kind in {"DRAM_LOAD", "DRAM_STORE"}:
            lines.append(
                f"| {kind} | {delta['before']} | {delta['after']} | "
                f"{delta['removed']} |"
            )
    if result["tensor_delta"]:
        lines.extend(
            [
                "",
                "## Changed Tensor DRAM operations",
                "",
                "| Tensor slice | Stores removed | Loads removed |",
                "|---|---:|---:|",
            ]
        )
        for delta in result["tensor_delta"]:
            lines.append(
                f"| `{delta['tensor_slice']}` | "
                f"{delta['removed_store_ops']} | "
                f"{delta['removed_load_ops']} |"
            )
    if selected_resources:
        lines.extend(
            [
                "",
                "## Declared macro resources",
                "",
                "| Tensor slice | Address | Stores removed | Loads removed |",
                "|---|---:|---:|---:|",
            ]
        )
        for delta in selected_resources:
            lines.append(
                f"| `{delta['tensor_slice']}` | "
                f"`{delta['dram_address_hex']}` | "
                f"{delta['removed_store_ops']} | "
                f"{delta['removed_load_ops']} |"
            )
    edge_count_before = result["dag_delta"]["before_edge_count"]
    edge_count_after = result["dag_delta"]["after_edge_count"]
    if edge_count_before is not None and edge_count_after is not None:
        lines.extend(
            [
                "",
                "## Dependency and VRF pressure",
                "",
                f"- Edges: {edge_count_before} -> {edge_count_after}",
            ]
        )
        pressure = result["dag_delta"]["vrf_pressure"]
        if pressure["before"] is not None and pressure["after"] is not None:
            lines.append(
                "- Peak VRF live ranges: "
                f"{pressure['before']['peak_live_ranges']} -> "
                f"{pressure['after']['peak_live_ranges']}"
            )
    return "\n".join(lines).rstrip() + "\n"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dag", required=True, type=Path)
    parser.add_argument("--after-dag", required=True, type=Path)
    parser.add_argument("--candidate-source", required=True, type=Path)
    parser.add_argument("--before-seq2", type=Path)
    parser.add_argument("--before-seq6", type=Path)
    parser.add_argument("--after-seq2", type=Path)
    parser.add_argument("--after-seq6", type=Path)
    parser.add_argument(
        "--candidate-required", choices=("on", "off"), default="on"
    )
    parser.add_argument("--gate", choices=("on", "off"), default="on")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    before_micro_ops_path = args.before_dag / "micro_ops.jsonl"
    after_micro_ops_path = args.after_dag / "micro_ops.jsonl"
    before_macros_path = args.before_dag / "macro_candidates.json"
    result = evaluate_dag_evidence(
        load_json(args.before_dag / "candidates.json"),
        load_jsonl(before_micro_ops_path),
        load_jsonl(after_micro_ops_path),
        args.candidate_source.read_text(encoding="utf-8"),
        before_macros=(
            load_json(before_macros_path)
            if before_macros_path.is_file()
            else None
        ),
        before_seq2=load_json(args.before_seq2) if args.before_seq2 else None,
        before_seq6=load_json(args.before_seq6) if args.before_seq6 else None,
        after_seq2=load_json(args.after_seq2) if args.after_seq2 else None,
        after_seq6=load_json(args.after_seq6) if args.after_seq6 else None,
        before_edges=load_jsonl(args.before_dag / "edges.jsonl"),
        after_edges=load_jsonl(args.after_dag / "edges.jsonl"),
        before_lifetimes=load_json(args.before_dag / "lifetimes.json"),
        after_lifetimes=load_json(args.after_dag / "lifetimes.json"),
        before_metadata=load_json(args.before_dag / "run_metadata.json"),
        after_metadata=load_json(args.after_dag / "run_metadata.json"),
        before_micro_ops_sha256=_sha256_file(before_micro_ops_path),
        after_micro_ops_sha256=_sha256_file(after_micro_ops_path),
        candidate_required=args.candidate_required == "on",
        gate_enabled=args.gate == "on",
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary.write_text(render_markdown(result), encoding="utf-8")
    return 0 if result["evidence_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
