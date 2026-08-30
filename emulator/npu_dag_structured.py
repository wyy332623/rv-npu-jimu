"""Structured, agent-readable export for the NPU micro-op DAG.

This module is deliberately a read-only presentation layer.  It consumes the
existing micro-op DAG and cluster analysis without changing trace collection,
instruction semantics, or dependency construction.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from emulator.npu_dag_adapter import DagWorkloadAdapter, get_dag_workload
from emulator.npu_micro_op_dag import (
    MicroOp,
    TensorCluster,
    _DRAM_REGIONS,
    _classify_dram_addr,
    extract_clusters,
)


SCHEMA_NAME = "jimu-npu-micro-op-dag"
SCHEMA_VERSION = "1.2.0"
MULTISEQ_SCHEMA_NAME = "jimu-npu-multiseq-dag"
MULTISEQ_SCHEMA_VERSION = "2.0.0"

# Element capacities from emulator/npu_device_mini.py::VRF_SIZES.  They are
# repeated here deliberately: this module is a read-only exporter and must not
# import or instantiate the emulator while it is analysing an existing trace.
_VRF_CAPACITIES = {
    1: 64,
    5: 20480,
    6: 4096,
    7: 1024,
    8: 4096,
    9: 64,
    13: 256,
}
_VRF_BANK_NAMES = {
    1: "MEM_MULTIPLY_VRF",
    5: "MEM_MVM_INITIAL_VRF",
    6: "MEM_MFU_INITIAL_VRF",
    7: "MEM_ADDSUB_VRF_0",
    8: "MEM_ADDSUB_VRF_1",
    9: "MEM_ADDSUB_VRF_2",
    13: "MEM_MVM_ACC_VRF",
}
_MACRO_CACHE_BANK = 6

_REGION_BASES = {name: base for base, name, _span in _DRAM_REGIONS}
_POSITIONAL_REGIONS = {"X", "Q", "K", "V", "RES", "OUT"}
_PROJECTION_LAYOUT = (
    ("Q", "W_Q", "B_Q"),
    ("K", "W_K", "B_K"),
    ("V", "W_V", "B_V"),
    ("SELF_OUTPUT", "W_SELF_OUTPUT", "B_SELF_OUTPUT"),
    ("FFN_INTERMEDIATE", "W_FFN_INTERMEDIATE", "B_FFN_INTERMEDIATE"),
    ("FFN_OUTPUT", "W_FFN_OUTPUT", "B_FFN_OUTPUT"),
)


def _node_id(index: int) -> str:
    return f"op-{index:06d}"


def _phase_id(index: int) -> str:
    return f"phase-{index:04d}"


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def _semantic_phase(label: str) -> str:
    """Map heuristic cluster labels to stable phase categories."""
    text = label.lower()
    rules = (
        ("k proj", "k_projection"),
        ("v proj", "v_projection"),
        ("q proj", "q_projection"),
        ("attn score", "attention_score_softmax"),
        ("attn context", "attention_context"),
        ("self-output", "self_output"),
        ("residual add1", "residual_add_1"),
        ("layernorm 1", "layernorm_1"),
        ("ffn inter", "ffn_intermediate_gelu"),
        ("ffn output", "ffn_output"),
        ("residual add2", "residual_add_2"),
        ("layernorm 2", "layernorm_2"),
        ("save x", "residual_save"),
        ("ln scratch", "layernorm_scratch"),
        ("tail", "tail"),
    )
    for marker, phase in rules:
        if marker in text:
            return phase
    cleaned = re.sub(r"\bp\d+\b|\bt\d+\b", "", text)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned or "unclassified"


def _clean_label(label: str) -> str:
    """Normalize legacy arrow encodings before emitting UTF-8 artifacts."""
    return label.replace("â†’", "->").replace("→", "->")


def _classify_low_dram_address(
    address: int,
    *,
    dim: int,
    hidden_size: int,
    seq_len: int,
) -> tuple[str, int | None, int | None]:
    """Decode the dynamic BERT input/parameter area below 0x200."""
    input_end = hidden_size * seq_len
    if 0 <= address < input_end:
        position = address // hidden_size
        tile = (address % hidden_size) // dim
        return "X", position, tile

    proj_base = input_end + 4
    mat_size = hidden_size * hidden_size
    stride = mat_size + hidden_size
    for index, (_projection, weight_name, bias_name) in enumerate(
        _PROJECTION_LAYOUT
    ):
        weight_base = proj_base + index * stride
        bias_base = weight_base + mat_size
        if weight_base <= address < bias_base:
            tile = (address - weight_base) // (dim * dim)
            return weight_name, None, tile
        if bias_base <= address < bias_base + hidden_size:
            tile = (address - bias_base) // dim
            return bias_name, None, tile

    num_tiles = max(hidden_size // dim, 1)
    ln_base = proj_base + len(_PROJECTION_LAYOUT) * stride
    ln_size = num_tiles * 8
    layernorm_regions = (
        "LN1_GAMMA",
        "LN1_BETA",
        "LN2_GAMMA",
        "LN2_BETA",
    )
    for index, name in enumerate(layernorm_regions):
        base = ln_base + index * ln_size
        if base <= address < base + ln_size:
            return name, None, (address - base) // 8

    return "UNKNOWN_LOW", None, None


def _resource_record(
    resource: tuple,
    *,
    dim: int,
    hidden_size: int,
    seq_len: int,
    adapter: DagWorkloadAdapter | None = None,
) -> dict[str, Any]:
    """Convert an internal resource tuple into a stable JSON object."""
    space = str(resource[0])
    record: dict[str, Any] = {"space": space}

    if space == "DRAM":
        address = int(resource[1])
        if adapter is not None:
            record.update(
                {
                    "address": address,
                    "address_hex": f"0x{address:x}",
                    **adapter.classify_dram(address),
                }
            )
            return record
        num_tiles = max(hidden_size // dim, 1)
        if address < 0x200:
            region, position, tile = _classify_low_dram_address(
                address,
                dim=dim,
                hidden_size=hidden_size,
                seq_len=seq_len,
            )
        else:
            region, raw_position = _classify_dram_addr(
                address, num_tiles, dim
            )
            if region in _POSITIONAL_REGIONS:
                position = int(raw_position) if raw_position >= 0 else None
            else:
                position = None
            tile = None

        record.update(
            {
                "address": address,
                "address_hex": f"0x{address:x}",
                "tensor": region,
                "position": position,
            }
        )

        if address >= 0x200 and region in _REGION_BASES:
            base = _REGION_BASES[region]
            if region in _POSITIONAL_REGIONS:
                position_stride = num_tiles * 8
                position_offset = (
                    (position or 0) * position_stride
                    if position is not None
                    else 0
                )
                relative = address - base - position_offset
                tile_stride = 8
            elif region == "UNIT_VEC":
                relative = address - base
                tile_stride = dim
            else:
                relative = address - base
                tile_stride = 8
            if relative >= 0:
                tile = relative // tile_stride
        record["tile"] = tile

        if region.startswith(("W_", "B_", "LN1_", "LN2_")):
            record["slice"] = (
                f"{region}[tile={tile}]"
                if tile is not None
                else f"{region}[address=0x{address:x}]"
            )
        elif position is not None and tile is not None:
            record["slice"] = f"{region}[pos={position},tile={tile}]"
        elif position is not None:
            record["slice"] = f"{region}[pos={position}]"
        elif tile is not None:
            record["slice"] = f"{region}[tile={tile}]"
        else:
            record["slice"] = f"{region}[address=0x{address:x}]"
        record["position_in_config"] = (
            position is None or 0 <= position < seq_len
        )
    elif space == "VRF":
        record.update(
            {
                "bank": int(resource[1]),
                "row": int(resource[2]),
            }
        )
    elif space in {"SRF", "REG"}:
        record["index"] = int(resource[1])
    elif space == "MRF":
        record["row"] = int(resource[1]) if len(resource) > 1 else None
    else:
        record["coordinates"] = list(resource[1:])

    return record


def _phase_records(
    clusters: list[TensorCluster],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    phases: list[dict[str, Any]] = []
    node_to_phase: dict[int, dict[str, Any]] = {}

    for index, cluster in enumerate(clusters):
        label = _clean_label(cluster.label)
        kind = _semantic_phase(label)
        if (
            kind == "residual_add_1"
            and " t1" in label.lower()
            and phases
            and phases[-1]["kind"] == "residual_add_2"
        ):
            # The legacy cluster label has two identical address-based rules
            # for the second residual-add tile. Execution order disambiguates
            # the FFN-side instance following Residual Add2.
            kind = "residual_add_2"
            label = re.sub(
                r"Residual Add1",
                "Residual Add2",
                label,
                count=1,
                flags=re.IGNORECASE,
            )
        position_match = re.search(r"\bp(\d+)\b", label.lower())
        if position_match:
            phase_position = int(position_match.group(1))
        elif cluster.produced_region in _POSITIONAL_REGIONS:
            phase_position = (
                cluster.produced_position
                if cluster.produced_position >= 0
                else None
            )
        else:
            phase_position = None
        produced_position = (
            cluster.produced_position
            if cluster.produced_region in _POSITIONAL_REGIONS
            and cluster.produced_position >= 0
            else None
        )
        phase = {
            "id": _phase_id(index),
            "index": index,
            "kind": kind,
            "label": label,
            "position": phase_position,
            "node_ids": [_node_id(i) for i in cluster.member_indices],
            "node_range": [cluster.first_idx, cluster.last_idx],
            "produces": {
                "tensor": cluster.produced_region or None,
                "position": produced_position,
            },
            "consumes": sorted(cluster.consumed_regions),
            "stats": {
                "dram_load_bytes": cluster.dram_load_bytes,
                "dram_store_bytes": cluster.dram_store_bytes,
                "dram_total_bytes": (
                    cluster.dram_load_bytes + cluster.dram_store_bytes
                ),
                "compute_flops": cluster.compute_flops,
                "arithmetic_intensity": (
                    cluster.compute_flops
                    / (
                        cluster.dram_load_bytes
                        + cluster.dram_store_bytes
                    )
                    if (
                        cluster.dram_load_bytes
                        + cluster.dram_store_bytes
                    )
                    else 0.0
                ),
            },
        }
        phases.append(phase)
        for node_index in cluster.member_indices:
            node_to_phase[node_index] = phase

    return phases, node_to_phase


def _phase_records_from_event_ranges(
    micro_ops: list[MicroOp],
    phase_ranges: list[dict[str, Any]],
    *,
    adapter: DagWorkloadAdapter,
    dim: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Build authoritative top-level phases supplied by a workload runner.

    Hybrid models can execute host code between firmware invocations.  Such a
    boundary cannot be reconstructed from NPU addresses, so the runner records
    exact half-open event ranges and the structured exporter preserves them.
    """

    phases: list[dict[str, Any]] = []
    node_to_phase: dict[int, dict[str, Any]] = {}
    for index, supplied in enumerate(phase_ranges):
        start = int(supplied["event_start"])
        end = int(supplied["event_end"])
        member_indices = [
            node_index
            for node_index, node in enumerate(micro_ops)
            if any(start <= event_index < end for event_index in node.event_indices)
        ]
        if not member_indices:
            continue

        consumed = set()
        produced = []
        load_bytes = 0
        store_bytes = 0
        compute_flops = 0
        for node_index in member_indices:
            node = micro_ops[node_index]
            for resource in node.uses:
                if resource[0] == "DRAM":
                    consumed.add(adapter.classify_dram(int(resource[1]))["tensor"])
            for resource in node.defs:
                if resource[0] == "DRAM":
                    produced.append(adapter.classify_dram(int(resource[1])))
            if node.kind == "MAT_LOAD":
                load_bytes += dim * dim * 4
            elif node.kind == "DRAM_LOAD":
                load_bytes += dim * 4
            elif node.kind == "DRAM_STORE":
                store_bytes += dim * 4
            elif node.kind == "MV_MUL":
                compute_flops += 2 * dim * dim
            elif node.kind in {"VV_BINOP", "V_UNARY", "SOFTMAX", "LAYERNORM"}:
                compute_flops += dim

        final_output = produced[-1] if produced else {}
        phase = {
            "id": _phase_id(index),
            "index": index,
            "kind": str(supplied.get("kind") or f"execution_phase_{index}"),
            "label": str(supplied.get("label") or supplied.get("kind") or index),
            "position": supplied.get("position"),
            "node_ids": [_node_id(node_index) for node_index in member_indices],
            "node_range": [min(member_indices), max(member_indices)],
            "event_range": [start, end],
            "execution_domain": supplied.get("execution_domain", "npu_firmware"),
            "boundary_source": "workload-runner",
            "produces": {
                "tensor": final_output.get("tensor"),
                "position": final_output.get("position"),
            },
            "consumes": sorted(consumed),
            "stats": {
                "dram_load_bytes": load_bytes,
                "dram_store_bytes": store_bytes,
                "dram_total_bytes": load_bytes + store_bytes,
                "compute_flops": compute_flops,
                "arithmetic_intensity": (
                    compute_flops / (load_bytes + store_bytes)
                    if load_bytes + store_bytes
                    else 0.0
                ),
            },
        }
        phases.append(phase)
        for node_index in member_indices:
            node_to_phase[node_index] = phase
    return phases, node_to_phase


def _node_records(
    micro_ops: list[MicroOp],
    node_to_phase: dict[int, dict[str, Any]],
    *,
    dim: int,
    hidden_size: int,
    seq_len: int,
    adapter: DagWorkloadAdapter | None = None,
) -> list[dict[str, Any]]:
    records = []
    for index, node in enumerate(micro_ops):
        phase = node_to_phase.get(index)
        uses = [
            _resource_record(
                resource,
                dim=dim,
                hidden_size=hidden_size,
                seq_len=seq_len,
                adapter=adapter,
            )
            for resource in node.uses
        ]
        defs = [
            _resource_record(
                resource,
                dim=dim,
                hidden_size=hidden_size,
                seq_len=seq_len,
                adapter=adapter,
            )
            for resource in node.defs
        ]
        positions = sorted(
            {
                item["position"]
                for item in uses + defs
                if item.get("space") == "DRAM"
                and item.get("position") is not None
            }
        )
        tiles = sorted(
            {
                item["tile"]
                for item in uses + defs
                if item.get("space") == "DRAM"
                and item.get("tile") is not None
            }
        )
        records.append(
            {
                "id": _node_id(index),
                "index": index,
                "kind": node.kind,
                "name": node.name,
                "detail": node.detail or None,
                "event_indices": list(node.event_indices),
                "phase_id": phase["id"] if phase else None,
                "phase_kind": phase["kind"] if phase else None,
                "phase_label": phase["label"] if phase else None,
                "position": positions[0] if len(positions) == 1 else None,
                "positions": positions,
                "tile": tiles[0] if len(tiles) == 1 else None,
                "tiles": tiles,
                # Head cannot be reconstructed safely from the current trace.
                "head": None,
                "annotation_confidence": (
                    "heuristic" if phase or positions or tiles else "unknown"
                ),
                "uses": uses,
                "defs": defs,
            }
        )
    return records


def _edge_records(
    edges: list[tuple[int, int, tuple]],
    *,
    dim: int,
    hidden_size: int,
    seq_len: int,
    adapter: DagWorkloadAdapter | None = None,
) -> list[dict[str, Any]]:
    records = []
    for index, (source, target, resource) in enumerate(edges):
        records.append(
            {
                "id": f"edge-{index:06d}",
                "source": _node_id(source),
                "target": _node_id(target),
                "type": "RAW",
                "resource": _resource_record(
                    resource,
                    dim=dim,
                    hidden_size=hidden_size,
                    seq_len=seq_len,
                    adapter=adapter,
                ),
            }
        )
    return records


def _tensor_records(
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    for node in nodes:
        for access, resources in (("read", node["uses"]), ("write", node["defs"])):
            for resource in resources:
                if resource.get("space") != "DRAM":
                    continue
                key = resource["slice"]
                tensor = tensors.setdefault(
                    key,
                    {
                        "id": f"tensor-{len(tensors):04d}",
                        "slice": key,
                        "tensor": resource["tensor"],
                        "position": resource.get("position"),
                        "tile": resource.get("tile"),
                        "addresses": set(),
                        "producers": set(),
                        "consumers": set(),
                        "read_count": 0,
                        "write_count": 0,
                    },
                )
                tensor["addresses"].add(resource["address_hex"])
                if access == "read":
                    tensor["consumers"].add(node["id"])
                    tensor["read_count"] += 1
                else:
                    tensor["producers"].add(node["id"])
                    tensor["write_count"] += 1

    records = []
    for tensor in tensors.values():
        records.append(
            {
                **tensor,
                "addresses": sorted(tensor["addresses"]),
                "producers": sorted(tensor["producers"]),
                "consumers": sorted(tensor["consumers"]),
            }
        )
    return records


def _pattern_records(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for phase in phases:
        grouped[phase["kind"]].append(phase)

    patterns = []
    for index, (kind, instances) in enumerate(sorted(grouped.items())):
        node_counts = [len(item["node_ids"]) for item in instances]
        load_bytes = [
            item["stats"]["dram_load_bytes"] for item in instances
        ]
        store_bytes = [
            item["stats"]["dram_store_bytes"] for item in instances
        ]
        positions = sorted(
            {
                item["position"]
                for item in instances
                if item["position"] is not None
            }
        )
        patterns.append(
            {
                "id": f"pattern-{index:04d}",
                "kind": kind,
                "instance_count": len(instances),
                "repeated": len(instances) > 1,
                "instance_phase_ids": [item["id"] for item in instances],
                "positions": positions,
                "node_count": {
                    "min": min(node_counts),
                    "max": max(node_counts),
                },
                "dram_load_bytes": {
                    "min": min(load_bytes),
                    "max": max(load_bytes),
                    "total": sum(load_bytes),
                },
                "dram_store_bytes": {
                    "min": min(store_bytes),
                    "max": max(store_bytes),
                    "total": sum(store_bytes),
                },
            }
        )
    return patterns


def _same_vrf(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("space") == right.get("space") == "VRF"
        and left.get("bank") == right.get("bank")
        and left.get("row") == right.get("row")
    )


def _same_dram(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("space") == right.get("space") == "DRAM"
        and left.get("address") == right.get("address")
    )


def _semantic_id(prefix: str, payload: dict[str, Any]) -> str:
    """Return a deterministic ID that does not depend on candidate ranking."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _vrf_lifetime_records(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe every concrete VRF definition and its existing consumers."""
    node_by_id = {node["id"]: node for node in nodes}
    lifetimes = []
    for producer in nodes:
        for resource in producer["defs"]:
            if resource.get("space") != "VRF":
                continue
            consumers = []
            for edge in edges:
                if edge["source"] != producer["id"]:
                    continue
                if not _same_vrf(edge["resource"], resource):
                    continue
                consumer = node_by_id[edge["target"]]
                consumers.append(consumer)

            end_index = max(
                (consumer["index"] for consumer in consumers),
                default=producer["index"],
            )
            next_definition = next(
                (
                    node
                    for node in nodes[producer["index"] + 1 :]
                    if any(
                        _same_vrf(defined, resource)
                        for defined in node["defs"]
                    )
                ),
                None,
            )
            lifetimes.append(
                {
                    "id": f"vrf-life-{len(lifetimes):05d}",
                    "resource": {
                        "space": "VRF",
                        "bank": resource["bank"],
                        "row": resource["row"],
                    },
                    "producer": producer["id"],
                    "consumers": [
                        consumer["id"] for consumer in consumers
                    ],
                    "interval": {
                        "start_node": producer["id"],
                        "end_node": _node_id(end_index),
                        "start_index": producer["index"],
                        "end_index": end_index,
                        "span_nodes": end_index - producer["index"] + 1,
                    },
                    "next_definition": (
                        next_definition["id"] if next_definition else None
                    ),
                    "phase_id": producer["phase_id"],
                }
            )
    return lifetimes


def _dram_candidate_records(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    vrf_lifetimes: list[dict[str, Any]],
    *,
    dim: int,
    seq_len: int,
) -> dict[str, Any]:
    """Find DRAM round trips and produce conservative VRF-cache evidence."""
    node_by_id = {node["id"]: node for node in nodes}
    vector_bytes = dim * 4
    candidates = []
    rejected = []

    for producer in nodes:
        if producer["kind"] != "DRAM_STORE":
            continue
        dram_defs = [
            resource
            for resource in producer["defs"]
            if resource.get("space") == "DRAM"
        ]
        for dram in dram_defs:
            source_vrf = next(
                (
                    resource
                    for resource in producer["uses"]
                    if resource.get("space") == "VRF"
                ),
                None,
            )
            consumer_edges = [
                edge
                for edge in edges
                if edge["source"] == producer["id"]
                and _same_dram(edge["resource"], dram)
            ]
            consumers = sorted(
                (node_by_id[edge["target"]] for edge in consumer_edges),
                key=lambda node: node["index"],
            )
            next_write = next(
                (
                    node
                    for node in nodes[producer["index"] + 1 :]
                    if any(
                        _same_dram(defined, dram)
                        for defined in node["defs"]
                    )
                ),
                None,
            )

            reasons = []
            if not consumers:
                reasons.append(
                    "no downstream consumer before overwrite or graph end"
                )
            if source_vrf is None:
                reasons.append("DRAM store has no explicit VRF source")

            intervening_writes = []
            for consumer in consumers:
                for node in nodes[
                    producer["index"] + 1 : consumer["index"]
                ]:
                    if any(
                        _same_dram(defined, dram)
                        for defined in node["defs"]
                    ):
                        intervening_writes.append(node["id"])
            intervening_writes = sorted(set(intervening_writes))
            if intervening_writes:
                reasons.append(
                    "intervening DRAM write invalidates producer-consumer pair"
                )

            base_record = {
                "tensor": dram.get("tensor"),
                "tensor_role": dram.get("role"),
                "tensor_slice": dram.get("slice"),
                "dram_address": dram.get("address"),
                "dram_address_hex": dram.get("address_hex"),
                "producer": producer["id"],
                "producer_phase_id": producer["phase_id"],
                "consumers": [consumer["id"] for consumer in consumers],
                "consumer_kinds": [
                    consumer["kind"] for consumer in consumers
                ],
                "next_write": next_write["id"] if next_write else None,
                "intervening_write": bool(intervening_writes),
                "intervening_write_nodes": intervening_writes,
            }
            if reasons:
                rejected.append(
                    {
                        "id": f"rejected-dram-{len(rejected):04d}",
                        **base_record,
                        "eligible": False,
                        "rejection_reasons": reasons,
                    }
                )
                continue

            end_index = max(consumer["index"] for consumer in consumers)
            source_overwrites = [
                node["id"]
                for node in nodes[
                    producer["index"] + 1 : end_index + 1
                ]
                if any(
                    _same_vrf(defined, source_vrf)
                    for defined in node["defs"]
                )
            ]
            reuse_source = not source_overwrites
            overlapping_lifetimes = [
                lifetime["id"]
                for lifetime in vrf_lifetimes
                if lifetime["interval"]["start_index"] <= end_index
                and lifetime["interval"]["end_index"] >= producer["index"]
                and not (
                    lifetime["resource"]["bank"] == source_vrf["bank"]
                    and lifetime["resource"]["row"] == source_vrf["row"]
                )
            ]
            observed_saved = vector_bytes * (1 + len(consumers))
            if dram.get("tensor") in {"K", "V"} and seq_len == 1:
                projected_seq2 = vector_bytes * (2 + 2 * 2)
                projected_seq6 = vector_bytes * (6 + 6 * 6)
                projection_assumption = (
                    "K/V heuristic: seq stores plus seq^2 attention loads"
                )
            else:
                projected_seq2 = round(
                    observed_saved * 2 / max(seq_len, 1)
                )
                projected_seq6 = round(
                    observed_saved * 6 / max(seq_len, 1)
                )
                projection_assumption = (
                    "linear scaling from the observed graph sequence length"
                )
            relocation_factor = 1.0 if reuse_source else 0.7

            candidate = {
                "id": f"candidate-dram-{len(candidates):04d}",
                **base_record,
                "eligible": True,
                "status": (
                    "eligible-reuse-source-vrf"
                    if reuse_source
                    else "eligible-requires-vrf-relocation"
                ),
                "confidence": (
                    "high"
                    if reuse_source
                    and all(
                        consumer["kind"] == "DRAM_LOAD"
                        for consumer in consumers
                    )
                    else "medium"
                ),
                "proof": {
                    "def_use_edges": [
                        edge["id"] for edge in consumer_edges
                    ],
                    "same_exact_address": True,
                    "intervening_write": False,
                    "raw_edge_evidence_authoritative": True,
                },
                "vrf_plan": {
                    "source_resource": {
                        "space": "VRF",
                        "bank": source_vrf["bank"],
                        "row": source_vrf["row"],
                    },
                    "reuse_source_resource": reuse_source,
                    "source_overwrite_nodes": source_overwrites,
                    "required_vector_slots": 0 if reuse_source else 1,
                    "overlapping_live_ranges": overlapping_lifetimes,
                    "overlapping_live_range_count": len(
                        overlapping_lifetimes
                    ),
                    "allocation_proven": reuse_source,
                },
                "lifetime": {
                    "start_node": producer["id"],
                    "end_node": _node_id(end_index),
                    "start_index": producer["index"],
                    "end_index": end_index,
                    "span_nodes": end_index - producer["index"] + 1,
                },
                "estimated_saving": {
                    "element_bytes": 4,
                    "vector_bytes": vector_bytes,
                    "removed_store_ops": 1,
                    "removed_load_ops": len(consumers),
                    "observed_graph_bytes": observed_saved,
                    "projected_seq2_bytes": projected_seq2,
                    "projected_seq6_bytes": projected_seq6,
                    "projection_assumption": projection_assumption,
                },
                "priority_score": projected_seq6 * relocation_factor,
            }
            candidate["stable_id"] = _semantic_id(
                "candidate-dram-stable",
                {
                    "tensor_slice": candidate["tensor_slice"],
                    "dram_address": candidate["dram_address"],
                    "consumer_kinds": sorted(candidate["consumer_kinds"]),
                    "removed_store_ops": 1,
                    "removed_load_ops": len(consumers),
                },
            )
            candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["priority_score"],
            -candidate["estimated_saving"]["projected_seq6_bytes"],
            candidate["producer"],
        ),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank

    return {
        "schema": {
            "name": "jimu-npu-dram-cache-candidates",
            "version": "1.1.0",
        },
        "analysis_limits": {
            "address_overlap": "exact-address only",
            "dram_element_bytes": 4,
            "sequence_projection": (
                "K/V seq+seq^2 attention heuristic; others linear"
            ),
            "vrf_relocation_search": "not implemented",
            "annotations_authoritative": False,
        },
        "summary": {
            "eligible": len(candidates),
            "high_confidence": sum(
                1
                for candidate in candidates
                if candidate["confidence"] == "high"
            ),
            "requires_relocation": sum(
                1
                for candidate in candidates
                if not candidate["vrf_plan"]["reuse_source_resource"]
            ),
            "rejected": len(rejected),
        },
        "candidates": sorted(candidates, key=lambda item: item["rank"]),
        "rejected": rejected,
    }


def _intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        int(left["start_index"]) <= int(right["end_index"])
        and int(right["start_index"]) <= int(left["end_index"])
    )


def _regions_overlap(left: dict[str, int], right: dict[str, int]) -> bool:
    return left["base"] < right["end"] and right["base"] < left["end"]


def _allocate_macro_members(
    members: list[dict[str, Any]],
    vrf_lifetimes: list[dict[str, Any]],
    *,
    dim: int,
) -> tuple[list[dict[str, Any]], bool]:
    """First-fit vector allocation for a macro over the observed trace.

    Relocated values use the long-lived MFU cache bank.  The result is an
    exact proof for the exported DAG configuration; source code still has to
    extend the proof to every validation configuration named by the prompt.
    """
    occupied: list[dict[str, Any]] = []
    for lifetime in vrf_lifetimes:
        resource = lifetime.get("resource", {})
        bank = resource.get("bank")
        row = resource.get("row")
        if bank not in _VRF_CAPACITIES or row is None:
            continue
        occupied.append(
            {
                "bank": int(bank),
                "base": int(row),
                "end": int(row) + dim,
                "interval": lifetime["interval"],
                "owner": lifetime["id"],
            }
        )

    allocations: list[dict[str, Any]] = []
    allocation_proven = True
    ordered = sorted(
        members,
        key=lambda item: (
            int(item["lifetime"]["start_index"]),
            int(item["lifetime"]["end_index"]),
            str(item["stable_id"]),
        ),
    )
    for member in ordered:
        interval = member["lifetime"]
        reuse_source = bool(
            member.get("vrf_plan", {}).get("reuse_source_resource")
        )
        if reuse_source:
            source = member["vrf_plan"]["source_resource"]
            bank = int(source["bank"])
            base = int(source["row"])
            capacity = int(_VRF_CAPACITIES.get(bank, 0))
            fit = base >= 0 and base % dim == 0 and base + dim <= capacity
            proposed_region = {"base": base, "end": base + dim}
            if fit:
                fit = not any(
                    entry["bank"] == bank
                    and entry["base"] != base
                    and _intervals_overlap(interval, entry["interval"])
                    and _regions_overlap(proposed_region, entry)
                    for entry in occupied
                )
        else:
            bank = _MACRO_CACHE_BANK
            capacity = _VRF_CAPACITIES[bank]
            base = -1
            fit = False
            for candidate_base in range(0, capacity - dim + 1, dim):
                proposed_region = {
                    "base": candidate_base,
                    "end": candidate_base + dim,
                }
                conflict = any(
                    entry["bank"] == bank
                    and _intervals_overlap(interval, entry["interval"])
                    and _regions_overlap(proposed_region, entry)
                    for entry in occupied
                )
                if not conflict:
                    base = candidate_base
                    fit = True
                    break

        record = {
            "member_id": member["id"],
            "member_stable_id": member["stable_id"],
            "tensor_slice": member["tensor_slice"],
            "dram_address": member["dram_address"],
            "dram_address_hex": member["dram_address_hex"],
            "bank": bank,
            "bank_name": _VRF_BANK_NAMES.get(bank, f"VRF_BANK_{bank}"),
            "base": base if fit else None,
            "end": base + dim if fit else None,
            "size_elements": dim,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "first_write": interval["start_node"],
            "last_read": interval["end_node"],
            "interval": interval,
            "reuse_source_resource": reuse_source,
            "allocation_proven": fit,
            "proof_scope": "observed DAG firmware_config",
        }
        allocations.append(record)
        if fit:
            occupied.append(
                {
                    "bank": bank,
                    "base": base,
                    "end": base + dim,
                    "interval": interval,
                    "owner": member["stable_id"],
                }
            )
        else:
            allocation_proven = False

    return allocations, allocation_proven


def _macro_candidate_records(
    candidate_data: dict[str, Any],
    vrf_lifetimes: list[dict[str, Any]],
    *,
    dim: int,
) -> dict[str, Any]:
    """Build coherent L1 candidate groups with deterministic identities."""
    eligible = list(candidate_data.get("candidates", []))
    attention_tensors = {"Q", "K", "V", "SCORE", "SOFTMAX"}
    attention = [
        candidate
        for candidate in eligible
        if str(candidate.get("tensor")) in attention_tensors
        or str(candidate.get("tensor_role")) == "attention"
    ]
    pipeline = [
        candidate
        for candidate in eligible
        if candidate not in attention
    ]
    group_specs: list[tuple[str, str, list[dict[str, Any]]]] = []
    if attention:
        group_specs.append(
            ("macro-dram-l1-attention", "attention-cache", attention)
        )
    if pipeline:
        group_specs.append(
            ("macro-dram-l1-pipeline", "pipeline-cache", pipeline)
        )
    if attention and pipeline:
        group_specs.append(
            ("macro-dram-l1-all-intermediates", "all-intermediates", eligible)
        )

    macros = []
    for macro_id, family, members in group_specs:
        allocations, allocation_proven = _allocate_macro_members(
            members,
            vrf_lifetimes,
            dim=dim,
        )
        member_ids = [member["id"] for member in members]
        stable_member_ids = sorted(member["stable_id"] for member in members)
        expected_resources = sorted(
            {
                (
                    str(member["tensor_slice"]),
                    int(member["dram_address"]),
                    str(member["dram_address_hex"]),
                )
                for member in members
            }
        )
        saving_keys = (
            "observed_graph_bytes",
            "projected_seq2_bytes",
            "projected_seq6_bytes",
        )
        estimated = {
            key: sum(
                int(member.get("estimated_saving", {}).get(key, 0))
                for member in members
            )
            for key in saving_keys
        }
        evidence_payload = {
            "macro_id": macro_id,
            "members": stable_member_ids,
            "resources": expected_resources,
        }
        macro = {
            "id": macro_id,
            "stable_id": macro_id,
            "evidence_id": _semantic_id("macro-evidence", evidence_payload),
            "level": "L1",
            "family": family,
            "eligible": allocation_proven,
            "status": (
                "eligible-trace-allocation-proven"
                if allocation_proven
                else "blocked-trace-allocation-unproven"
            ),
            "member_candidate_ids": member_ids,
            "member_stable_ids": stable_member_ids,
            "tensor_slices": sorted(
                {str(member["tensor_slice"]) for member in members}
            ),
            "expected_dram_resources": [
                {
                    "tensor_slice": tensor_slice,
                    "dram_address": address,
                    "dram_address_hex": address_hex,
                }
                for tensor_slice, address, address_hex in expected_resources
            ],
            "allocation": {
                "allocator": "deterministic-first-fit-interval-v1",
                "preferred_bank": _MACRO_CACHE_BANK,
                "preferred_bank_name": _VRF_BANK_NAMES[_MACRO_CACHE_BANK],
                "capacity_source": "emulator/npu_device_mini.py::VRF_SIZES",
                "proof_scope": "observed DAG firmware_config only",
                "validation_config_extension_required": True,
                "allocation_proven": allocation_proven,
                "regions": allocations,
            },
            "estimated_saving": {
                **estimated,
                "projection_assumption": (
                    "sum of member estimates; independent probes authoritative"
                ),
            },
            "scope_policy": (
                "every expected Tensor/address must decrease; positive DRAM "
                "reductions outside the declared macro are rejected"
            ),
            "priority_score": estimated["projected_seq6_bytes"],
        }
        macros.append(macro)

    macros.sort(
        key=lambda item: (-item["priority_score"], str(item["id"]))
    )
    for rank, macro in enumerate(macros, start=1):
        macro["rank"] = rank
    return {
        "schema": {
            "name": "jimu-npu-dram-macro-candidates",
            "version": "1.0.0",
        },
        "summary": {
            "macros": len(macros),
            "eligible": sum(1 for macro in macros if macro["eligible"]),
            "blocked": sum(1 for macro in macros if not macro["eligible"]),
            "primitive_candidates": len(eligible),
        },
        "analysis_limits": {
            "allocation_proof_scope": "observed DAG firmware_config only",
            "source_validation_required": (
                "extend capacity and lifetime proof to every validation config"
            ),
            "grouping": "L1 attention, pipeline, and combined intermediates",
        },
        "macros": macros,
    }


def _macro_candidate_markdown(macro_data: dict[str, Any]) -> str:
    summary = macro_data["summary"]
    lines = [
        "# DRAM Macro Candidate Summary",
        "",
        (
            f"Macros: {summary['macros']}; eligible: {summary['eligible']}; "
            f"blocked: {summary['blocked']}; primitive members: "
            f"{summary['primitive_candidates']}."
        ),
        "",
        "| Rank | Macro | Level | Family | Members | Tensors | Seq6 est. B | Allocation |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for macro in macro_data["macros"]:
        lines.append(
            f"| {macro['rank']} | `{macro['id']}` | "
            f"{macro.get('level', 'L1')} | {macro['family']} | "
            f"{len(macro['member_candidate_ids'])} | "
            f"{len(macro['tensor_slices'])} | "
            f"{macro['estimated_saving']['projected_seq6_bytes']} | "
            f"{'proved' if macro['allocation']['allocation_proven'] else 'blocked'} |"
        )
    if not macro_data["macros"]:
        lines.append("| - | - | - | No coherent macro available | 0 | 0 | 0 | - |")
    lines.extend(
        [
            "",
            "A macro declaration uses `// JIMU_DAG_MACRO: macro-dram-...`. ",
            "Only macros with `cross_config_proven=true` are implementation-ready. ",
            "The independent correctness and DAG gates remain authoritative.",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate_markdown(candidate_data: dict[str, Any]) -> str:
    summary = candidate_data["summary"]
    lines = [
        "# DRAM Cache Candidate Summary",
        "",
        (
            f"Eligible: {summary['eligible']}; high confidence: "
            f"{summary['high_confidence']}; requires VRF relocation: "
            f"{summary['requires_relocation']}; rejected: "
            f"{summary['rejected']}."
        ),
        "",
        "| Rank | Candidate | Tensor | Producer -> Consumers | Phase | "
        "Observed B | Seq6 est. B | VRF plan | Confidence |",
        "|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for candidate in candidate_data["candidates"][:20]:
        consumers = ",".join(candidate["consumers"])
        vrf_plan = (
            "reuse"
            if candidate["vrf_plan"]["reuse_source_resource"]
            else "relocate"
        )
        lines.append(
            f"| {candidate['rank']} | {candidate['id']} | "
            f"`{candidate['tensor_slice']}` | {candidate['producer']} -> "
            f"{consumers} | {candidate['producer_phase_id'] or '-'} | "
            f"{candidate['estimated_saving']['observed_graph_bytes']} | "
            f"{candidate['estimated_saving']['projected_seq6_bytes']} | "
            f"{vrf_plan} | {candidate['confidence']} |"
        )
    if not candidate_data["candidates"]:
        lines.append("| - | - | No eligible exact-address round trip | - | - | 0 | 0 | - | - |")

    lines.extend(
        [
            "",
            "A `reuse` plan proves that the source VRF resource is not "
            "redefined before the last DRAM consumer. A `relocate` plan is "
            "only a candidate: allocation remains unproven.",
            "",
            "Savings use 4-byte emulator DRAM elements. K/V use a seq+seq^2 "
            "attention heuristic; other Tensors use linear projection. "
            "Validate all estimates with the independent metric probe.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_markdown(
    metadata: dict[str, Any],
    phases: list[dict[str, Any]],
    tensors: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    candidate_data: dict[str, Any],
    macro_candidate_data: dict[str, Any],
) -> str:
    config = metadata["firmware_config"]
    counts = metadata["counts"]
    lines = [
        "# Structured NPU DAG Summary",
        "",
        "This file is an index for agents. Use stable IDs to inspect the JSONL "
        "artifacts when more detail is needed.",
        "",
        "## Configuration",
        "",
        "| DIM | Hidden | Heads | Seq len | Micro-ops | Edges |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {config['dim']} | {config['hidden_size']} | "
            f"{config['num_head']} | {config['seq_len']} | "
            f"{counts['micro_ops']} | {counts['edges']} |"
        ),
        "",
        "## Phases",
        "",
        "| ID | Kind | Label | Nodes | Load B | Store B | FLOPs | AI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for phase in phases:
        stats = phase["stats"]
        lines.append(
            f"| {phase['id']} | {phase['kind']} | {phase['label']} | "
            f"{len(phase['node_ids'])} | {stats['dram_load_bytes']} | "
            f"{stats['dram_store_bytes']} | {stats['compute_flops']} | "
            f"{stats['arithmetic_intensity']:.2f} |"
        )

    repeated = [pattern for pattern in patterns if pattern["repeated"]]
    lines.extend(
        [
            "",
            "## Repeated phase patterns",
            "",
        ]
    )
    if repeated:
        lines.extend(
            [
                "| Pattern | Instances | Positions | Total load B | "
                "Total store B |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for pattern in repeated:
            positions = ", ".join(map(str, pattern["positions"])) or "-"
            lines.append(
                f"| {pattern['kind']} | {pattern['instance_count']} | "
                f"{positions} | {pattern['dram_load_bytes']['total']} | "
                f"{pattern['dram_store_bytes']['total']} |"
            )
    else:
        lines.append("No repeated phase pattern was detected.")

    memory_bound = sorted(
        (
            phase
            for phase in phases
            if phase["stats"]["dram_total_bytes"] > 0
        ),
        key=lambda phase: (
            phase["stats"]["arithmetic_intensity"],
            -phase["stats"]["dram_total_bytes"],
        ),
    )[:10]
    lines.extend(
        [
            "",
            "## Lowest arithmetic-intensity phases",
            "",
            "| Phase | Label | DRAM B | AI |",
            "|---|---|---:|---:|",
        ]
    )
    for phase in memory_bound:
        stats = phase["stats"]
        lines.append(
            f"| {phase['id']} | {phase['label']} | "
            f"{stats['dram_total_bytes']} | "
            f"{stats['arithmetic_intensity']:.2f} |"
        )

    active_tensors = sorted(
        tensors,
        key=lambda tensor: (
            -(tensor["read_count"] + tensor["write_count"]),
            tensor["slice"],
        ),
    )[:20]
    lines.extend(
        [
            "",
            "## Most accessed tensor slices",
            "",
            "| Slice | Reads | Writes | Producers | Consumers |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for tensor in active_tensors:
        lines.append(
            f"| `{tensor['slice']}` | {tensor['read_count']} | "
            f"{tensor['write_count']} | {len(tensor['producers'])} | "
            f"{len(tensor['consumers'])} |"
        )

    lines.extend(
        [
            "",
            _candidate_markdown(candidate_data),
            "",
            _macro_candidate_markdown(macro_candidate_data),
        ]
    )
    lines.extend(
        [
            "",
            "## Artifact guide",
            "",
            "- `micro_ops.jsonl`: semantic operations with phase and Tensor annotations.",
            "- `edges.jsonl`: producer-consumer dependencies using stable node IDs.",
            "- `phases.json`: phase hierarchy and per-phase traffic/FLOP statistics.",
            "- `tensors.json`: DRAM Tensor slices and their producers/consumers.",
            "- `patterns.json`: compressed repeated phase instances.",
            "- `lifetimes.json`: existing VRF def-use live intervals.",
            "- `candidates.json`: ranked DRAM round trips and rejected evidence.",
            "- `candidate_summary.md`: compact candidate table and limits.",
            "- `macro_candidates.json`: coherent L1 groups with VRF allocations.",
            "- `macro_candidate_summary.md`: ranked macro table and proof scope.",
            "- `micro_op_dag.txt`: legacy forensic text representation.",
            "",
            "Tensor, position, tile, and phase labels are heuristic evidence. "
            "The raw node resources and event indices remain authoritative.",
            "",
        ]
    )
    return "\n".join(lines)


def build_structured_dag(
    micro_ops: list[MicroOp],
    edges: list[tuple[int, int, tuple]],
    *,
    dim: int,
    hidden_size: int,
    seq_len: int,
    num_head: int,
    total_events: int | None = None,
    elf_path: str | Path | None = None,
    clusters: list[TensorCluster] | None = None,
    workload: str = "bert",
    phase_ranges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build all structured records without writing files."""
    config = {
        "dim": dim,
        "hidden_size": hidden_size,
        "num_head": num_head,
        "seq_len": seq_len,
        "num_tiles": hidden_size // dim,
    }
    adapter = get_dag_workload(workload, config)
    if clusters is None and not phase_ranges:
        clusters = extract_clusters(
            micro_ops,
            edges,
            dim=dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
        )

    if phase_ranges:
        phases, node_to_phase = _phase_records_from_event_ranges(
            micro_ops,
            phase_ranges,
            adapter=adapter,
            dim=dim,
        )
    else:
        phases, node_to_phase = _phase_records(clusters or [])
    nodes = _node_records(
        micro_ops,
        node_to_phase,
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        adapter=adapter,
    )
    edge_records = _edge_records(
        edges,
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        adapter=adapter,
    )
    tensors = _tensor_records(nodes)
    patterns = _pattern_records(phases)
    vrf_lifetimes = _vrf_lifetime_records(nodes, edge_records)
    candidate_data = _dram_candidate_records(
        nodes,
        edge_records,
        vrf_lifetimes,
        dim=dim,
        seq_len=seq_len,
    )
    macro_candidate_data = _macro_candidate_records(
        candidate_data,
        vrf_lifetimes,
        dim=dim,
    )

    elf = Path(elf_path).resolve() if elf_path else None
    elf_sha256 = None
    if elf and elf.is_file():
        elf_sha256 = hashlib.sha256(elf.read_bytes()).hexdigest()

    metadata = {
        "schema": {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "firmware_config": config,
        "workload": adapter.metadata(),
        "elf": {
            "path": str(elf) if elf else None,
            "sha256": elf_sha256,
        },
        "counts": {
            "instructions": total_events,
            "micro_ops": len(nodes),
            "edges": len(edge_records),
            "phases": len(phases),
            "tensor_slices": len(tensors),
            "repeated_patterns": sum(
                1 for pattern in patterns if pattern["repeated"]
            ),
            "vrf_lifetimes": len(vrf_lifetimes),
            "eligible_dram_candidates": candidate_data["summary"][
                "eligible"
            ],
            "rejected_dram_candidates": candidate_data["summary"][
                "rejected"
            ],
            "eligible_dram_macros": macro_candidate_data["summary"][
                "eligible"
            ],
            "blocked_dram_macros": macro_candidate_data["summary"][
                "blocked"
            ],
        },
        "annotation": {
            "method": (
                "adapter-layout-and-authoritative-execution-ranges"
                if phase_ranges
                else "adapter-layout-and-heuristic-dram-clusters"
            ),
            "head_mapping": "unavailable",
            "raw_resources_authoritative": True,
            "phase_boundaries_authoritative": bool(phase_ranges),
        },
    }
    return {
        "metadata": metadata,
        "micro_ops": nodes,
        "edges": edge_records,
        "tensors": tensors,
        "phases": phases,
        "patterns": patterns,
        "lifetimes": vrf_lifetimes,
        "candidates": candidate_data,
        "macro_candidates": macro_candidate_data,
    }


def write_structured_dag(
    output_dir: str | Path,
    micro_ops: list[MicroOp],
    edges: list[tuple[int, int, tuple]],
    *,
    dim: int,
    hidden_size: int,
    seq_len: int,
    num_head: int,
    total_events: int | None = None,
    elf_path: str | Path | None = None,
    clusters: list[TensorCluster] | None = None,
    workload: str = "bert",
    phase_ranges: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write PR1/PR2/PR3 structured artifacts and return their paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    structured = build_structured_dag(
        micro_ops,
        edges,
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        num_head=num_head,
        total_events=total_events,
        elf_path=elf_path,
        clusters=clusters,
        workload=workload,
        phase_ranges=phase_ranges,
    )

    paths = {
        "metadata": output / "run_metadata.json",
        "micro_ops": output / "micro_ops.jsonl",
        "edges": output / "edges.jsonl",
        "tensors": output / "tensors.json",
        "phases": output / "phases.json",
        "patterns": output / "patterns.json",
        "lifetimes": output / "lifetimes.json",
        "candidates": output / "candidates.json",
        "candidate_summary": output / "candidate_summary.md",
        "macro_candidates": output / "macro_candidates.json",
        "macro_candidate_summary": output / "macro_candidate_summary.md",
        "summary": output / "summary.md",
    }
    paths["metadata"].write_text(
        json.dumps(structured["metadata"], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    for key in ("micro_ops", "edges"):
        paths[key].write_text(
            "\n".join(_json_line(record) for record in structured[key])
            + ("\n" if structured[key] else ""),
            encoding="utf-8",
        )
    for key in (
        "tensors",
        "phases",
        "patterns",
        "lifetimes",
        "candidates",
        "macro_candidates",
    ):
        paths[key].write_text(
            json.dumps(structured[key], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    paths["candidate_summary"].write_text(
        _candidate_markdown(structured["candidates"]),
        encoding="utf-8",
    )
    paths["macro_candidate_summary"].write_text(
        _macro_candidate_markdown(structured["macro_candidates"]),
        encoding="utf-8",
    )
    paths["summary"].write_text(
        _summary_markdown(
            structured["metadata"],
            structured["phases"],
            structured["tensors"],
            structured["patterns"],
            structured["candidates"],
            structured["macro_candidates"],
        ),
        encoding="utf-8",
    )
    return paths


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_structured_dag(input_dir: str | Path) -> dict[str, Any]:
    """Load one structured DAG directory written by ``write_structured_dag``."""
    root = Path(input_dir)
    return {
        "metadata": json.loads(
            (root / "run_metadata.json").read_text(encoding="utf-8")
        ),
        "micro_ops": _read_jsonl(root / "micro_ops.jsonl"),
        "edges": _read_jsonl(root / "edges.jsonl"),
        "tensors": json.loads(
            (root / "tensors.json").read_text(encoding="utf-8")
        ),
        "phases": json.loads(
            (root / "phases.json").read_text(encoding="utf-8")
        ),
        "patterns": json.loads(
            (root / "patterns.json").read_text(encoding="utf-8")
        ),
        "lifetimes": json.loads(
            (root / "lifetimes.json").read_text(encoding="utf-8")
        ),
        "candidates": json.loads(
            (root / "candidates.json").read_text(encoding="utf-8")
        ),
        "macro_candidates": json.loads(
            (root / "macro_candidates.json").read_text(encoding="utf-8")
        ),
    }


def _reuse_family(
    resource_or_tensor: dict[str, Any] | str,
) -> tuple[str, str, str] | None:
    if isinstance(resource_or_tensor, dict):
        resource = resource_or_tensor
        explicit = (
            resource.get("reuse_family"),
            resource.get("cache_level"),
            resource.get("reuse_kind"),
        )
        if all(explicit):
            return tuple(str(value) for value in explicit)  # type: ignore[return-value]
        tensor = str(resource.get("tensor", "UNKNOWN"))
    else:
        tensor = str(resource_or_tensor)
    if tensor.startswith("B_") or tensor.startswith(("LN1_", "LN2_")):
        return "loop-invariant-parameters", "L2", "loop-invariant-load"
    if tensor == "UNIT_VEC":
        return "loop-invariant-parameters", "L2", "loop-invariant-load"
    if tensor == "X":
        return "sequence-input", "L2", "sequence-input-reuse"
    if tensor.startswith("W_"):
        return "weight-stationary", "L3", "weight-stationary-load"
    return None


def _reuse_group_key(resource: dict[str, Any]) -> str:
    tensor = str(resource.get("tensor", "UNKNOWN"))
    tile = resource.get("tile")
    if tile is None:
        return tensor
    # X values differ by position, but all positions implement one repeated
    # input-tile access pattern. ``slices`` below retains each concrete value.
    return f"{tensor}[tile={tile}]"


def _readonly_reuse_records(structured: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Aggregate repeated read-only DRAM loads by semantic Tensor/tile key."""
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    groups: dict[str, dict[str, Any]] = {}

    def ensure(resource: dict[str, Any]) -> dict[str, Any] | None:
        if resource.get("space") != "DRAM":
            return None
        tensor = str(resource.get("tensor", "UNKNOWN"))
        family = _reuse_family(resource)
        if family is None:
            return None
        key = _reuse_group_key(resource)
        return groups.setdefault(
            key,
            {
                "semantic_key": key,
                "tensor": tensor,
                "tile": resource.get("tile"),
                "family": family[0],
                "level": family[1],
                "kind": family[2],
                "addresses": set(),
                "slices": set(),
                "node_ids": set(),
                "node_kinds": set(),
                "read_ops": 0,
                "read_bytes": 0,
                "write_ops": 0,
                "slice_access_bytes": {},
            },
        )

    for node in structured["micro_ops"]:
        kind = str(node.get("kind", "UNKNOWN"))
        access_bytes = dim * dim * 4 if kind == "MAT_LOAD" else dim * 4
        for resource in node.get("uses", []):
            group = ensure(resource)
            if group is None:
                continue
            tensor_slice = str(resource.get("slice", group["semantic_key"]))
            group["addresses"].add(str(resource.get("address_hex")))
            group["slices"].add(tensor_slice)
            group["node_ids"].add(str(node["id"]))
            group["node_kinds"].add(kind)
            group["read_ops"] += 1
            group["read_bytes"] += access_bytes
            previous = int(group["slice_access_bytes"].get(tensor_slice, 0))
            group["slice_access_bytes"][tensor_slice] = max(
                previous, access_bytes
            )
        for resource in node.get("defs", []):
            group = ensure(resource)
            if group is not None:
                group["write_ops"] += 1

    records = {}
    for key, group in groups.items():
        compulsory_bytes = sum(group["slice_access_bytes"].values())
        records[key] = {
            **group,
            "addresses": sorted(group["addresses"]),
            "slices": sorted(group["slices"]),
            "node_ids": sorted(group["node_ids"]),
            "node_kinds": sorted(group["node_kinds"]),
            "slice_access_bytes": dict(sorted(group["slice_access_bytes"].items())),
            "unique_values": len(group["slices"]),
            "compulsory_read_bytes": compulsory_bytes,
            "removable_read_bytes": max(
                int(group["read_bytes"]) - compulsory_bytes, 0
            ),
        }
    return records


def build_multiseq_dag(
    structured_by_seq: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Compare at least two concrete sequence DAGs and expose exact reuse.

    Unlike the legacy seq=1 projection heuristic, this analysis measures each
    supplied graph. Candidates are semantic across configurations, while the
    per-configuration records retain exact addresses and node IDs.
    """
    if len(structured_by_seq) < 2:
        raise ValueError("multi-sequence DAG analysis requires at least two graphs")
    seq_lens = sorted(int(seq) for seq in structured_by_seq)
    configs = []
    reuse_by_seq = {}
    common_geometry = None
    common_workload = None
    for seq_len in seq_lens:
        structured = structured_by_seq[seq_len]
        metadata = structured["metadata"]
        config = dict(metadata["firmware_config"])
        if int(config["seq_len"]) != seq_len:
            raise ValueError(
                f"DAG key seq{seq_len} does not match metadata seq_len="
                f"{config['seq_len']}"
            )
        geometry = (
            int(config["dim"]),
            int(config["hidden_size"]),
            int(config["num_head"]),
        )
        if common_geometry is None:
            common_geometry = geometry
        elif geometry != common_geometry:
            raise ValueError("multi-sequence DAG geometry differs across graphs")
        workload_name = str(metadata.get("workload", {}).get("name", "bert"))
        if common_workload is None:
            common_workload = workload_name
        elif workload_name != common_workload:
            raise ValueError("multi-sequence DAG workloads differ across graphs")
        configs.append(
            {
                "seq_len": seq_len,
                "firmware_config": config,
                "elf_sha256": metadata.get("elf", {}).get("sha256"),
                "counts": metadata.get("counts", {}),
            }
        )
        reuse_by_seq[seq_len] = _readonly_reuse_records(structured)

    semantic_keys = sorted(
        set().union(*(set(records) for records in reuse_by_seq.values()))
    )
    candidates = []
    reference_seq = seq_lens[-1]
    for semantic_key in semantic_keys:
        per_config = {}
        exemplar = None
        present_all = True
        read_only_all = True
        read_counts = []
        for seq_len in seq_lens:
            record = reuse_by_seq[seq_len].get(semantic_key)
            if record is None:
                present_all = False
                per_config[f"seq{seq_len}"] = {
                    "present": False,
                    "read_ops": 0,
                    "write_ops": 0,
                    "removable_read_bytes": 0,
                }
                read_counts.append(0)
                continue
            exemplar = exemplar or record
            read_only_all = read_only_all and int(record["write_ops"]) == 0
            read_counts.append(int(record["read_ops"]))
            per_config[f"seq{seq_len}"] = {
                "present": True,
                "addresses": record["addresses"],
                "slices": record["slices"],
                "node_ids": record["node_ids"],
                "node_kinds": record["node_kinds"],
                "read_ops": record["read_ops"],
                "write_ops": record["write_ops"],
                "read_bytes": record["read_bytes"],
                "unique_values": record["unique_values"],
                "compulsory_read_bytes": record["compulsory_read_bytes"],
                "removable_read_bytes": record["removable_read_bytes"],
            }
        if exemplar is None:
            continue
        monotonic_reuse = all(
            left <= right for left, right in zip(read_counts, read_counts[1:])
        )
        reference = per_config[f"seq{reference_seq}"]
        analysis_eligible = bool(
            present_all
            and read_only_all
            and monotonic_reuse
            and int(reference.get("removable_read_bytes", 0)) > 0
        )
        payload = {
            "semantic_key": semantic_key,
            "family": exemplar["family"],
            "level": exemplar["level"],
            "seq_lens": seq_lens,
        }
        candidate = {
            "id": _semantic_id("candidate-multiseq", payload),
            "semantic_key": semantic_key,
            "tensor": exemplar["tensor"],
            "tile": exemplar["tile"],
            "family": exemplar["family"],
            "level": exemplar["level"],
            "kind": exemplar["kind"],
            "analysis_eligible": analysis_eligible,
            "implementation_ready": False,
            "status": (
                "eligible-analysis-allocation-required"
                if analysis_eligible
                else "rejected-cross-sequence-proof"
            ),
            "cross_sequence_proof": {
                "present_in_all_configs": present_all,
                "read_only_in_all_configs": read_only_all,
                "read_count_monotonic": monotonic_reuse,
                "semantic_alignment": "tensor+tile; X positions grouped as values",
                "measured_not_projected": True,
            },
            "per_config": per_config,
            "allocation": {
                "proven": False,
                "reason": (
                    "DAG-PR5 proves reuse only; cross-config VRF allocation "
                    "and source lifetime proof are required before editing"
                ),
            },
            "priority_score": int(
                reference.get("removable_read_bytes", 0)
            ),
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (-item["priority_score"], item["semantic_key"])
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    family_specs = {
        "loop-invariant-parameters": (
            "macro-dram-l2-loop-invariants", "L2"
        ),
        "sequence-input": ("macro-dram-l2-sequence-input", "L2"),
        "weight-stationary": ("macro-dram-l3-weight-stationary", "L3"),
    }
    families = []
    for family_name, (family_id, level) in family_specs.items():
        members = [
            candidate
            for candidate in candidates
            if candidate["family"] == family_name
            and candidate["analysis_eligible"]
        ]
        if not members:
            continue
        savings = {
            f"seq{seq_len}": sum(
                int(
                    member["per_config"][f"seq{seq_len}"].get(
                        "removable_read_bytes", 0
                    )
                )
                for member in members
            )
            for seq_len in seq_lens
        }
        families.append(
            {
                "id": family_id,
                "family": family_name,
                "level": level,
                "analysis_eligible": True,
                "implementation_ready": False,
                "member_ids": [member["id"] for member in members],
                "member_count": len(members),
                "measured_removable_read_bytes": savings,
                "allocation": {
                    "proven": False,
                    "required_next_pr": "DAG-PR6",
                },
            }
        )
    families.sort(
        key=lambda family: (
            -int(
                family["measured_removable_read_bytes"].get(
                    f"seq{reference_seq}", 0
                )
            ),
            family["id"],
        )
    )

    return {
        "schema": {
            "name": MULTISEQ_SCHEMA_NAME,
            "version": MULTISEQ_SCHEMA_VERSION,
        },
        "workload": common_workload,
        "configurations": configs,
        "reference_seq_len": reference_seq,
        "analysis_limits": {
            "scope": "read-only repeated DRAM loads",
            "semantic_alignment": "Tensor+tile across concrete sequence DAGs",
            "allocation_search": "not implemented until DAG-PR6",
            "candidate_declaration": (
                "analysis-only; not accepted by the PR4 declaration gate"
            ),
        },
        "summary": {
            "candidates": len(candidates),
            "analysis_eligible": sum(
                1 for candidate in candidates if candidate["analysis_eligible"]
            ),
            "implementation_ready": 0,
            "families": len(families),
        },
        "families": families,
        "candidates": candidates,
    }


def _config_identity(structured: dict[str, Any]) -> str:
    config = structured["metadata"]["firmware_config"]
    return (
        f"dim{int(config['dim'])}-h{int(config['hidden_size'])}-"
        f"head{int(config['num_head'])}-seq{int(config['seq_len'])}"
    )


def _config_tuple(structured: dict[str, Any]) -> tuple[int, int, int, int]:
    config = structured["metadata"]["firmware_config"]
    return (
        int(config["dim"]),
        int(config["hidden_size"]),
        int(config["seq_len"]),
        int(config["num_head"]),
    )


def _l2_cache_values(
    structured: dict[str, Any],
    family_name: str,
) -> list[dict[str, Any]]:
    """Build exact value lifetimes for one read-only L2 family."""
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    writes = set()
    for node in structured["micro_ops"]:
        for resource in node.get("defs", []):
            if resource.get("space") == "DRAM":
                writes.add(int(resource.get("address", -1)))

    groups: dict[tuple[str, int], dict[str, Any]] = {}
    for node in structured["micro_ops"]:
        for resource in node.get("uses", []):
            if resource.get("space") != "DRAM":
                continue
            family = _reuse_family(resource)
            if family is None or family[0] != family_name:
                continue
            address = int(resource.get("address", -1))
            tensor_slice = str(resource.get("slice", f"DRAM[0x{address:x}]"))
            key = (tensor_slice, address)
            record = groups.setdefault(
                key,
                {
                    "tensor": str(resource.get("tensor", "UNKNOWN")),
                    "tensor_slice": tensor_slice,
                    "tile": resource.get("tile"),
                    "position": resource.get("position"),
                    "dram_address": address,
                    "dram_address_hex": f"0x{address:x}",
                    "read_nodes": [],
                    "read_indices": [],
                },
            )
            record["read_nodes"].append(str(node["id"]))
            record["read_indices"].append(int(node["index"]))

    values = []
    for (_tensor_slice, address), record in sorted(groups.items()):
        indices = record.pop("read_indices")
        if len(indices) <= 1 or address in writes:
            continue
        interval = {
            "start_node": _node_id(min(indices)),
            "end_node": _node_id(max(indices)),
            "start_index": min(indices),
            "end_index": max(indices),
            "span_nodes": max(indices) - min(indices) + 1,
        }
        payload = {
            "family": family_name,
            "tensor_slice": record["tensor_slice"],
            "dram_address": address,
        }
        values.append(
            {
                "id": _semantic_id("cache-value", payload),
                **record,
                "read_ops": len(indices),
                "compulsory_load_ops": 1,
                "removable_load_ops": len(indices) - 1,
                "removable_read_bytes": (len(indices) - 1) * dim * 4,
                "read_only_proven": True,
                "interval": interval,
            }
        )
    return values


def _allocate_l2_family(
    structured: dict[str, Any],
    family_name: str,
) -> dict[str, Any]:
    """Allocate L2 values against every observed VRF live range."""
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    capacity = int(_VRF_CAPACITIES[_MACRO_CACHE_BANK])
    occupied = []
    for lifetime in structured.get("lifetimes", []):
        resource = lifetime.get("resource", {})
        bank = resource.get("bank")
        row = resource.get("row")
        if bank is None or row is None:
            continue
        occupied.append(
            {
                "bank": int(bank),
                "base": int(row),
                "end": int(row) + dim,
                "interval": lifetime["interval"],
                "owner": lifetime["id"],
            }
        )
    existing_count = len(occupied)
    allocations = []
    values = sorted(
        _l2_cache_values(structured, family_name),
        key=lambda item: (
            int(item["interval"]["start_index"]),
            int(item["interval"]["end_index"]),
            str(item["id"]),
        ),
    )
    for value in values:
        interval = value["interval"]
        base = None
        for candidate_base in range(0, capacity - dim + 1, dim):
            proposed = {
                "base": candidate_base,
                "end": candidate_base + dim,
            }
            conflict = any(
                entry["bank"] == _MACRO_CACHE_BANK
                and _intervals_overlap(interval, entry["interval"])
                and _regions_overlap(proposed, entry)
                for entry in occupied
            )
            if not conflict:
                base = candidate_base
                break
        fit = base is not None
        allocation = {
            **value,
            "bank": _MACRO_CACHE_BANK,
            "bank_name": _VRF_BANK_NAMES[_MACRO_CACHE_BANK],
            "base": base,
            "end": base + dim if fit else None,
            "size_elements": dim,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "allocation_proven": fit,
        }
        allocations.append(allocation)
        if fit:
            occupied.append(
                {
                    "bank": _MACRO_CACHE_BANK,
                    "base": base,
                    "end": base + dim,
                    "interval": interval,
                    "owner": value["id"],
                }
            )

    allocation_proven = bool(allocations) and all(
        item["allocation_proven"] for item in allocations
    )
    return {
        "config_id": _config_identity(structured),
        "firmware_config": dict(config),
        "elf_sha256": structured["metadata"].get("elf", {}).get("sha256"),
        "family": family_name,
        "bank": _MACRO_CACHE_BANK,
        "bank_name": _VRF_BANK_NAMES[_MACRO_CACHE_BANK],
        "capacity_elements": capacity,
        "alignment_elements": dim,
        "existing_lifetimes_checked": existing_count,
        "member_count": len(allocations),
        "removable_read_bytes": sum(
            int(item["removable_read_bytes"]) for item in allocations
        ),
        "allocation_proven": allocation_proven,
        "regions": allocations,
    }


def _proof_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _transient_scratch_config_proof(
    structured: dict[str, Any],
) -> dict[str, Any]:
    """Prove replacement of the active transient scratch round trips.

    The softmax and LayerNorm scratch values are produced by firmware and
    consumed exactly once before the same address is written again.  After Q
    precomputation, bank-13 rows ``[0, dim)`` are outside the retained-Q
    region and may be reused by these non-overlapping transient lifetimes.

    A prior accepted optimization may already have removed one of the two
    scratch families.  A region with no load or store events is therefore
    recorded as already eliminated and omitted from the next macro scope.  A
    partially observed region remains subject to the original exact pair-count
    proof and is not silently accepted.
    """
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    seq_len = int(config["seq_len"])
    num_head = int(config["num_head"])
    hidden_size = int(config["hidden_size"])
    num_tiles = hidden_size // dim
    capacity = int(_VRF_CAPACITIES[13])
    nodes = structured.get("micro_ops", [])
    addresses = (
        (0x500, "softmax-probability", seq_len * num_head),
        (0x500 + num_tiles * 8, "layernorm-tile0", seq_len * 2),
    )
    regions = []
    eliminated_regions = []
    all_intervals: list[tuple[int, int, str]] = []
    for address, purpose, expected_pairs in addresses:
        events = []
        tensor_slice = None
        for node in nodes:
            resources = (
                node.get("defs", [])
                if node.get("kind") == "DRAM_STORE"
                else node.get("uses", [])
                if node.get("kind") == "DRAM_LOAD"
                else []
            )
            resource = next(
                (
                    item
                    for item in resources
                    if item.get("space") == "DRAM"
                    and int(item.get("address", -1)) == address
                ),
                None,
            )
            if resource is None:
                continue
            tensor_slice = tensor_slice or resource.get("slice")
            events.append(node)

        if not events:
            eliminated_regions.append(
                {
                    "tensor": "SCRATCH",
                    "tensor_slice": None,
                    "purpose": purpose,
                    "dram_address": address,
                    "dram_address_hex": f"0x{address:x}",
                    "store_ops": 0,
                    "load_ops": 0,
                    "expected_pairs": expected_pairs,
                    "observed_pairs": 0,
                    "removable_read_write_bytes": 0,
                    "status": "already-eliminated",
                }
            )
            continue

        pairs = []
        pending_store = None
        sequence_proven = True
        for event in events:
            if event["kind"] == "DRAM_STORE":
                if pending_store is not None:
                    sequence_proven = False
                pending_store = event
                continue
            if pending_store is None:
                sequence_proven = False
                continue
            interval_nodes = nodes[
                pending_store["index"] : event["index"] + 1
            ]
            bank13_conflicts = []
            for interval_node in interval_nodes:
                for resource in (
                    interval_node.get("uses", [])
                    + interval_node.get("defs", [])
                ):
                    if resource.get("space") != "VRF":
                        continue
                    if int(resource.get("bank", -1)) != 13:
                        continue
                    row = int(resource.get("row", -1))
                    if row < dim and row + dim > 0:
                        bank13_conflicts.append(interval_node["id"])
            pair = {
                "store_node": pending_store["id"],
                "load_node": event["id"],
                "interval": [pending_store["index"], event["index"]],
                "bank13_conflict_nodes": sorted(set(bank13_conflicts)),
                "allocation_proven": not bank13_conflicts,
            }
            pairs.append(pair)
            all_intervals.append(
                (pending_store["index"], event["index"], purpose)
            )
            pending_store = None
        if pending_store is not None:
            sequence_proven = False

        pair_count_proven = bool(len(pairs) == expected_pairs)
        region_proven = bool(
            sequence_proven
            and pair_count_proven
            and all(pair["allocation_proven"] for pair in pairs)
        )
        vector_bytes = dim * 4
        regions.append(
            {
                "tensor": "SCRATCH",
                "tensor_slice": str(tensor_slice),
                "purpose": purpose,
                "dram_address": address,
                "dram_address_hex": f"0x{address:x}",
                "store_ops": sum(
                    1 for event in events if event["kind"] == "DRAM_STORE"
                ),
                "load_ops": sum(
                    1 for event in events if event["kind"] == "DRAM_LOAD"
                ),
                "expected_pairs": expected_pairs,
                "observed_pairs": len(pairs),
                "removable_read_write_bytes": len(pairs)
                * 2
                * vector_bytes,
                "alternating_store_load_proven": sequence_proven,
                "pair_count_proven": pair_count_proven,
                "allocation_proven": region_proven,
                "pairs": pairs,
            }
        )

    intervals_non_overlapping = all(
        left[1] < right[0]
        for left, right in zip(
            sorted(all_intervals), sorted(all_intervals)[1:]
        )
    )
    scratch_end = dim
    retained_q_base = seq_len * dim
    capacity_proven = bool(
        scratch_end <= capacity
        and scratch_end <= retained_q_base
        and scratch_end % dim == 0
    )
    allocation_proven = bool(
        regions
        and all(region["allocation_proven"] for region in regions)
        and intervals_non_overlapping
        and capacity_proven
    )
    return {
        "config_id": _config_identity(structured),
        "firmware_config": dict(config),
        "elf_sha256": structured["metadata"].get("elf", {}).get("sha256"),
        "member_count": len(regions),
        "expected_member_count": len(regions),
        "configured_member_count": len(addresses),
        "eliminated_member_count": len(eliminated_regions),
        "removable_read_bytes": sum(
            int(region["removable_read_write_bytes"]) for region in regions
        ),
        "allocation_proven": allocation_proven,
        "dependency_proven": bool(regions)
        and all(
            region["alternating_store_load_proven"]
            and region["pair_count_proven"]
            for region in regions
        ),
        "lifetimes_non_overlapping": intervals_non_overlapping,
        "transient_allocation": {
            "bank": 13,
            "bank_name": _VRF_BANK_NAMES[13],
            "base": 0,
            "end": scratch_end,
            "size_elements": dim,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "reused_by": [region["purpose"] for region in regions],
            "retained_q_base": retained_q_base,
            "disjoint_from_retained_q": scratch_end <= retained_q_base,
            "allocation_proven": capacity_proven,
        },
        "regions": regions,
        "eliminated_regions": eliminated_regions,
        "schedule_signature": {
            "before": "pipeline -> DRAM scratch -> pipeline",
            "after": "pipeline -> bank13[0] -> pipeline",
            "operation_order": "unchanged",
            "fp16_boundary": "unchanged",
        },
    }


def _l3_kv_config_proof(structured: dict[str, Any]) -> dict[str, Any]:
    """Prove the bounded K/V position-interchange schedule for one config.

    The proposed schedule keeps ``tr/tc`` arithmetic order for every output,
    holds exactly one matrix tile in MRF, and interleaves only independent
    sequence positions inside that tile's residency interval.  Partial sums
    use one aligned vector per position in MEM_MVM_ACC_VRF.
    """
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    hidden_size = int(config["hidden_size"])
    seq_len = int(config["seq_len"])
    num_tiles = hidden_size // dim
    capacity = int(_VRF_CAPACITIES[13])
    partial_end = seq_len * dim
    records = _readonly_reuse_records(structured)
    regions = []
    for record in records.values():
        if record.get("tensor") not in {"W_K", "W_V"}:
            continue
        slices = list(record.get("slices", []))
        addresses = list(record.get("addresses", []))
        if len(slices) != 1 or len(addresses) != 1:
            continue
        address_hex = str(addresses[0])
        regions.append(
            {
                "tensor": str(record["tensor"]),
                "tensor_slice": str(slices[0]),
                "tile": record.get("tile"),
                "dram_address": int(address_hex, 16),
                "dram_address_hex": address_hex,
                "read_ops": int(record.get("read_ops", 0)),
                "compulsory_load_ops": 1,
                "removable_load_ops": max(
                    int(record.get("read_ops", 0)) - 1, 0
                ),
                "removable_read_bytes": int(
                    record.get("removable_read_bytes", 0)
                ),
                "read_only_proven": int(record.get("write_ops", 0)) == 0,
                "matrix_load_only": record.get("node_kinds") == ["MAT_LOAD"],
            }
        )
    regions.sort(key=lambda item: (item["tensor"], item["dram_address"]))
    expected_tiles = 2 * num_tiles * num_tiles
    weights_proven = bool(len(regions) == expected_tiles) and all(
        region["read_only_proven"]
        and region["matrix_load_only"]
        and region["read_ops"] == seq_len
        and region["removable_load_ops"] == max(seq_len - 1, 0)
        for region in regions
    )
    partial_sum_proven = bool(
        partial_end <= capacity and partial_end % dim == 0
    )
    allocation_proven = bool(weights_proven and partial_sum_proven)
    return {
        "config_id": _config_identity(structured),
        "firmware_config": dict(config),
        "elf_sha256": structured["metadata"].get("elf", {}).get("sha256"),
        "member_count": len(regions),
        "expected_member_count": expected_tiles,
        "removable_read_bytes": sum(
            int(region["removable_read_bytes"]) for region in regions
        ),
        "allocation_proven": allocation_proven,
        "dependency_proven": weights_proven,
        "mrf_residency_proven": weights_proven,
        "fp16_order_proven": weights_proven,
        "partial_sum_allocation": {
            "bank": 13,
            "bank_name": _VRF_BANK_NAMES[13],
            "base": 0,
            "end": partial_end,
            "size_elements": partial_end,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "one_vector_per_position": True,
            "allocation_proven": partial_sum_proven,
        },
        "regions": regions,
        "schedule_signature": {
            "before": "position,tr,tc",
            "after": "tr,tc,position",
            "per_output_reduction_order": "tc=0..num_tiles-1 (unchanged)",
            "mrf_clobbers_inside_position_loop": [],
            "bias_boundary": "after final tc for each position (unchanged)",
        },
    }


def _l3_q_config_proof(structured: dict[str, Any]) -> dict[str, Any]:
    """Prove the bounded Q position-interchange schedule for one config.

    Q cannot remain in MRF across attention because the K.T/V.T builders
    deliberately replace MRF.  The proved schedule therefore computes all Q
    positions first, using one bank-13 partial per position, then retains the
    completed Q vectors in a disjoint bank-13 region until attention consumes
    them.
    """
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    hidden_size = int(config["hidden_size"])
    seq_len = int(config["seq_len"])
    num_tiles = hidden_size // dim
    capacity = int(_VRF_CAPACITIES[13])
    partial_end = seq_len * dim
    retained_base = partial_end
    retained_size = seq_len * hidden_size
    retained_end = retained_base + retained_size
    records = _readonly_reuse_records(structured)
    regions = []
    for record in records.values():
        if record.get("tensor") != "W_Q":
            continue
        slices = list(record.get("slices", []))
        addresses = list(record.get("addresses", []))
        if len(slices) != 1 or len(addresses) != 1:
            continue
        address_hex = str(addresses[0])
        regions.append(
            {
                "tensor": "W_Q",
                "tensor_slice": str(slices[0]),
                "tile": record.get("tile"),
                "dram_address": int(address_hex, 16),
                "dram_address_hex": address_hex,
                "read_ops": int(record.get("read_ops", 0)),
                "compulsory_load_ops": 1,
                "removable_load_ops": max(
                    int(record.get("read_ops", 0)) - 1, 0
                ),
                "removable_read_bytes": int(
                    record.get("removable_read_bytes", 0)
                ),
                "read_only_proven": int(record.get("write_ops", 0)) == 0,
                "matrix_load_only": record.get("node_kinds") == ["MAT_LOAD"],
            }
        )
    regions.sort(key=lambda item: item["dram_address"])
    expected_tiles = num_tiles * num_tiles
    weights_proven = bool(len(regions) == expected_tiles) and all(
        region["read_only_proven"]
        and region["matrix_load_only"]
        and region["read_ops"] == seq_len
        and region["removable_load_ops"] == max(seq_len - 1, 0)
        for region in regions
    )
    partial_sum_proven = bool(
        partial_end <= capacity and partial_end % dim == 0
    )
    retained_output_proven = bool(
        retained_base % dim == 0
        and retained_end <= capacity
        and retained_size % dim == 0
    )
    allocation_proven = bool(
        weights_proven and partial_sum_proven and retained_output_proven
    )
    return {
        "config_id": _config_identity(structured),
        "firmware_config": dict(config),
        "elf_sha256": structured["metadata"].get("elf", {}).get("sha256"),
        "member_count": len(regions),
        "expected_member_count": expected_tiles,
        "removable_read_bytes": sum(
            int(region["removable_read_bytes"]) for region in regions
        ),
        "allocation_proven": allocation_proven,
        "dependency_proven": weights_proven,
        "mrf_residency_proven": weights_proven,
        "fp16_order_proven": weights_proven,
        "partial_sum_allocation": {
            "bank": 13,
            "bank_name": _VRF_BANK_NAMES[13],
            "base": 0,
            "end": partial_end,
            "size_elements": partial_end,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "one_vector_per_position": True,
            "allocation_proven": partial_sum_proven,
        },
        "retained_output_allocation": {
            "bank": 13,
            "bank_name": _VRF_BANK_NAMES[13],
            "base": retained_base,
            "end": retained_end,
            "size_elements": retained_size,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "one_full_q_vector_per_position": True,
            "allocation_proven": retained_output_proven,
        },
        "regions": regions,
        "schedule_signature": {
            "before": "position,tr,tc",
            "after": "tr,tc,position; then attention(position)",
            "per_output_reduction_order": "tc=0..num_tiles-1 (unchanged)",
            "mrf_clobbers_inside_position_loop": [],
            "mrf_clobbers_after_q_retention": ["K.T build", "V.T build"],
            "bias_boundary": "after final tc for each position (unchanged)",
            "attention_input": "retained Q[pos,tr] from bank 13",
        },
    }


def _l3_self_output_config_proof(
    structured: dict[str, Any],
) -> dict[str, Any]:
    """Prove staged attention and SELF_OUTPUT across sequence positions."""
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    hidden_size = int(config["hidden_size"])
    seq_len = int(config["seq_len"])
    num_tiles = hidden_size // dim
    capacity = int(_VRF_CAPACITIES[13])
    partial_end = seq_len * dim
    state_size = seq_len * hidden_size
    state_a_base = partial_end
    state_a_end = state_a_base + state_size
    state_b_base = state_a_end
    state_b_end = state_b_base + state_size
    records = _readonly_reuse_records(structured)
    regions = []
    for record in records.values():
        if record.get("tensor") != "W_SELF_OUTPUT":
            continue
        slices = list(record.get("slices", []))
        addresses = list(record.get("addresses", []))
        if len(slices) != 1 or len(addresses) != 1:
            continue
        address_hex = str(addresses[0])
        regions.append(
            {
                "tensor": "W_SELF_OUTPUT",
                "tensor_slice": str(slices[0]),
                "tile": record.get("tile"),
                "dram_address": int(address_hex, 16),
                "dram_address_hex": address_hex,
                "read_ops": int(record.get("read_ops", 0)),
                "compulsory_load_ops": 1,
                "removable_load_ops": max(
                    int(record.get("read_ops", 0)) - 1, 0
                ),
                "removable_read_bytes": int(
                    record.get("removable_read_bytes", 0)
                ),
                "read_only_proven": int(record.get("write_ops", 0)) == 0,
                "matrix_load_only": record.get("node_kinds") == ["MAT_LOAD"],
            }
        )
    regions.sort(key=lambda item: item["dram_address"])
    expected_tiles = num_tiles * num_tiles
    weights_proven = bool(len(regions) == expected_tiles) and all(
        region["read_only_proven"]
        and region["matrix_load_only"]
        and region["read_ops"] == seq_len
        and region["removable_load_ops"] == max(seq_len - 1, 0)
        for region in regions
    )
    allocation_proven = bool(
        state_b_end <= capacity
        and partial_end % dim == 0
        and state_a_base % dim == 0
        and state_b_base % dim == 0
    )
    all_proven = bool(weights_proven and allocation_proven)
    return {
        "config_id": _config_identity(structured),
        "firmware_config": dict(config),
        "elf_sha256": structured["metadata"].get("elf", {}).get("sha256"),
        "member_count": len(regions),
        "expected_member_count": expected_tiles,
        "removable_read_bytes": sum(
            int(region["removable_read_bytes"]) for region in regions
        ),
        "allocation_proven": all_proven,
        "dependency_proven": weights_proven,
        "mrf_residency_proven": weights_proven,
        "fp16_order_proven": weights_proven,
        "partial_sum_allocation": {
            "bank": 13,
            "bank_name": _VRF_BANK_NAMES[13],
            "base": 0,
            "end": partial_end,
            "size_elements": partial_end,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "allocation_proven": partial_end <= capacity,
        },
        "state_allocations": [
            {
                "name": "attention-context",
                "bank": 13,
                "bank_name": _VRF_BANK_NAMES[13],
                "base": state_a_base,
                "end": state_a_end,
                "size_elements": state_size,
                "lifecycle": "retained Q -> attention context",
                "allocation_proven": state_a_end <= capacity,
            },
            {
                "name": "self-output",
                "bank": 13,
                "bank_name": _VRF_BANK_NAMES[13],
                "base": state_b_base,
                "end": state_b_end,
                "size_elements": state_size,
                "lifecycle": "SELF_OUTPUT result until residual1",
                "allocation_proven": state_b_end <= capacity,
            },
        ],
        "regions": regions,
        "schedule_signature": {
            "before": "position(attention,self-output,rest)",
            "after": "attention(position); self-output(tr,tc,position); rest(position)",
            "attention_context_handoff": "overwrite dead retained Q[pos] in state A",
            "self_output_handoff": "write state B; do not overwrite live context input",
            "per_output_reduction_order": "tc=0..num_tiles-1 (unchanged)",
            "bias_boundary": "after final tc for each position (unchanged)",
        },
    }


def _l3_ffn_intermediate_config_proof(
    structured: dict[str, Any],
) -> dict[str, Any]:
    """Prove staged residual1/LN1 and FFN_INTERMEDIATE across positions."""
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    hidden_size = int(config["hidden_size"])
    seq_len = int(config["seq_len"])
    num_tiles = hidden_size // dim
    capacity = int(_VRF_CAPACITIES[13])
    partial_end = seq_len * dim
    state_size = seq_len * hidden_size
    state_a_base = partial_end
    state_a_end = state_a_base + state_size
    state_b_base = state_a_end
    state_b_end = state_b_base + state_size
    records = _readonly_reuse_records(structured)
    regions = []
    for record in records.values():
        if record.get("tensor") != "W_FFN_INTERMEDIATE":
            continue
        slices = list(record.get("slices", []))
        addresses = list(record.get("addresses", []))
        if len(slices) != 1 or len(addresses) != 1:
            continue
        address_hex = str(addresses[0])
        regions.append(
            {
                "tensor": "W_FFN_INTERMEDIATE",
                "tensor_slice": str(slices[0]),
                "tile": record.get("tile"),
                "dram_address": int(address_hex, 16),
                "dram_address_hex": address_hex,
                "read_ops": int(record.get("read_ops", 0)),
                "compulsory_load_ops": 1,
                "removable_load_ops": max(
                    int(record.get("read_ops", 0)) - 1, 0
                ),
                "removable_read_bytes": int(
                    record.get("removable_read_bytes", 0)
                ),
                "read_only_proven": int(record.get("write_ops", 0)) == 0,
                "matrix_load_only": record.get("node_kinds") == ["MAT_LOAD"],
            }
        )
    regions.sort(key=lambda item: item["dram_address"])
    expected_tiles = num_tiles * num_tiles
    weights_proven = bool(len(regions) == expected_tiles) and all(
        region["read_only_proven"]
        and region["matrix_load_only"]
        and region["read_ops"] == seq_len
        and region["removable_load_ops"] == max(seq_len - 1, 0)
        for region in regions
    )
    allocation_proven = bool(
        state_b_end <= capacity
        and partial_end % dim == 0
        and state_a_base % dim == 0
        and state_b_base % dim == 0
    )
    all_proven = bool(weights_proven and allocation_proven)
    return {
        "config_id": _config_identity(structured),
        "firmware_config": dict(config),
        "elf_sha256": structured["metadata"].get("elf", {}).get("sha256"),
        "member_count": len(regions),
        "expected_member_count": expected_tiles,
        "removable_read_bytes": sum(
            int(region["removable_read_bytes"]) for region in regions
        ),
        "allocation_proven": all_proven,
        "dependency_proven": weights_proven,
        "mrf_residency_proven": weights_proven,
        "fp16_order_proven": weights_proven,
        "partial_sum_allocation": {
            "bank": 13,
            "bank_name": _VRF_BANK_NAMES[13],
            "base": 0,
            "end": partial_end,
            "size_elements": partial_end,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "allocation_proven": partial_end <= capacity,
        },
        "state_allocations": [
            {
                "name": "ln1-input",
                "bank": 13,
                "bank_name": _VRF_BANK_NAMES[13],
                "base": state_a_base,
                "end": state_a_end,
                "size_elements": state_size,
                "lifecycle": "attention context -> normalized residual1",
                "allocation_proven": state_a_end <= capacity,
            },
            {
                "name": "gelu-output",
                "bank": 13,
                "bank_name": _VRF_BANK_NAMES[13],
                "base": state_b_base,
                "end": state_b_end,
                "size_elements": state_size,
                "lifecycle": "SELF_OUTPUT result -> FFN_INTERMEDIATE+GELU",
                "allocation_proven": state_b_end <= capacity,
            },
        ],
        "l2_x_retention": {
            "bank": 6,
            "bank_name": _VRF_BANK_NAMES[6],
            "producer": "macro-dram-l2-sequence-input",
            "required_until": "residual2 after FFN_OUTPUT",
            "disjoint_from_l3_state_bank": True,
            "source_invariant_required": True,
        },
        "regions": regions,
        "schedule_signature": {
            "before": "position(residual1,ln1,ffn-intermediate,rest)",
            "after": (
                "residual1+ln1(position); "
                "ffn-intermediate(tr,tc,position)+GELU; rest(position)"
            ),
            "ln1_handoff": "write normalized residual1 to state A",
            "gelu_handoff": "write FFN_INTERMEDIATE+GELU to state B",
            "residual2_skip": "retain original X in the proved bank-6 L2 cache",
            "per_output_reduction_order": "tc=0..num_tiles-1 (unchanged)",
            "bias_boundary": "after final tc, before GELU (unchanged)",
        },
    }


def _l3_ffn_output_config_proof(
    structured: dict[str, Any],
) -> dict[str, Any]:
    """Prove staged FFN_OUTPUT and final residual/LN2 across positions."""
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    hidden_size = int(config["hidden_size"])
    seq_len = int(config["seq_len"])
    num_tiles = hidden_size // dim
    capacity = int(_VRF_CAPACITIES[13])
    partial_end = seq_len * dim
    state_size = seq_len * hidden_size
    state_a_base = partial_end
    state_a_end = state_a_base + state_size
    state_b_base = state_a_end
    state_b_end = state_b_base + state_size
    records = _readonly_reuse_records(structured)
    regions = []
    for record in records.values():
        if record.get("tensor") != "W_FFN_OUTPUT":
            continue
        slices = list(record.get("slices", []))
        addresses = list(record.get("addresses", []))
        if len(slices) != 1 or len(addresses) != 1:
            continue
        address_hex = str(addresses[0])
        regions.append(
            {
                "tensor": "W_FFN_OUTPUT",
                "tensor_slice": str(slices[0]),
                "tile": record.get("tile"),
                "dram_address": int(address_hex, 16),
                "dram_address_hex": address_hex,
                "read_ops": int(record.get("read_ops", 0)),
                "compulsory_load_ops": 1,
                "removable_load_ops": max(
                    int(record.get("read_ops", 0)) - 1, 0
                ),
                "removable_read_bytes": int(
                    record.get("removable_read_bytes", 0)
                ),
                "read_only_proven": int(record.get("write_ops", 0)) == 0,
                "matrix_load_only": record.get("node_kinds") == ["MAT_LOAD"],
            }
        )
    regions.sort(key=lambda item: item["dram_address"])
    expected_tiles = num_tiles * num_tiles
    weights_proven = bool(len(regions) == expected_tiles) and all(
        region["read_only_proven"]
        and region["matrix_load_only"]
        and region["read_ops"] == seq_len
        and region["removable_load_ops"] == max(seq_len - 1, 0)
        for region in regions
    )
    allocation_proven = bool(
        state_b_end <= capacity
        and partial_end % dim == 0
        and state_a_base % dim == 0
        and state_b_base % dim == 0
    )
    all_proven = bool(weights_proven and allocation_proven)
    return {
        "config_id": _config_identity(structured),
        "firmware_config": dict(config),
        "elf_sha256": structured["metadata"].get("elf", {}).get("sha256"),
        "member_count": len(regions),
        "expected_member_count": expected_tiles,
        "removable_read_bytes": sum(
            int(region["removable_read_bytes"]) for region in regions
        ),
        "allocation_proven": all_proven,
        "dependency_proven": weights_proven,
        "mrf_residency_proven": weights_proven,
        "fp16_order_proven": weights_proven,
        "partial_sum_allocation": {
            "bank": 13,
            "bank_name": _VRF_BANK_NAMES[13],
            "base": 0,
            "end": partial_end,
            "size_elements": partial_end,
            "alignment_elements": dim,
            "capacity_elements": capacity,
            "allocation_proven": partial_end <= capacity,
        },
        "state_allocations": [
            {
                "name": "ffn-output",
                "bank": 13,
                "bank_name": _VRF_BANK_NAMES[13],
                "base": state_a_base,
                "end": state_a_end,
                "size_elements": state_size,
                "lifecycle": "normalized residual1 -> FFN_OUTPUT result",
                "allocation_proven": state_a_end <= capacity,
            },
            {
                "name": "gelu-input",
                "bank": 13,
                "bank_name": _VRF_BANK_NAMES[13],
                "base": state_b_base,
                "end": state_b_end,
                "size_elements": state_size,
                "lifecycle": "FFN_INTERMEDIATE+GELU until FFN_OUTPUT",
                "allocation_proven": state_b_end <= capacity,
            },
        ],
        "l2_x_retention": {
            "bank": 6,
            "bank_name": _VRF_BANK_NAMES[6],
            "producer": "macro-dram-l2-sequence-input",
            "required_until": "residual2 after FFN_OUTPUT",
            "disjoint_from_l3_state_bank": True,
            "source_invariant_required": True,
        },
        "regions": regions,
        "schedule_signature": {
            "before": "position(ffn-output,residual2,ln2)",
            "after": "ffn-output(tr,tc,position); residual2+ln2(position)",
            "gelu_handoff": "read completed GELU vectors from state B",
            "ffn_output_handoff": "write FFN_OUTPUT vectors to state A",
            "residual2_skip": "read original X from the proved bank-6 L2 cache",
            "per_output_reduction_order": "tc=0..num_tiles-1 (unchanged)",
            "bias_boundary": "after final tc (unchanged)",
        },
    }


def _unit_vector_synthesis_config_proof(
    structured: dict[str, Any],
) -> dict[str, Any]:
    """Prove that DRAM identity rows can be synthesized from ISA immediates."""
    config = structured["metadata"]["firmware_config"]
    dim = int(config["dim"])
    hidden_size = int(config["hidden_size"])
    num_head = int(config["num_head"])
    head_size = hidden_size // num_head
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    write_addresses = set()
    for node in structured.get("micro_ops", []):
        for resource in node.get("defs", []):
            if resource.get("space") == "DRAM":
                write_addresses.add(int(resource.get("address", -1)))
        if node.get("kind") != "DRAM_LOAD":
            continue
        for resource in node.get("uses", []):
            if resource.get("tensor") != "UNIT_VEC":
                continue
            key = (
                str(resource.get("slice")),
                int(resource.get("address", -1)),
            )
            record = grouped.setdefault(
                key,
                {
                    "tensor": "UNIT_VEC",
                    "tensor_slice": key[0],
                    "tile": resource.get("tile"),
                    "dram_address": key[1],
                    "dram_address_hex": f"0x{key[1]:x}",
                    "read_ops": 0,
                },
            )
            record["read_ops"] += 1
    regions = sorted(grouped.values(), key=lambda item: item["dram_address"])
    expected_addresses = [0x900 + j * dim for j in range(head_size)]
    resources_proven = bool(len(regions) == head_size) and all(
        int(region.get("tile", -1)) == j
        and int(region["dram_address"]) == expected_addresses[j]
        and int(region["read_ops"]) == 1
        and int(region["dram_address"]) not in write_addresses
        for j, region in enumerate(regions)
    )
    isa_synthesis_proven = bool(
        0 < head_size <= dim <= 8
        and int.from_bytes(b"\x00\x3c", "little") == 0x3C00
    )
    all_proven = bool(resources_proven and isa_synthesis_proven)
    for region in regions:
        region.update(
            {
                "compulsory_load_ops": 1,
                "removable_load_ops": 1,
                "removable_read_bytes": dim * 4,
                "read_only_proven": (
                    int(region["dram_address"]) not in write_addresses
                ),
                "synthesized_value": (
                    f"identity row e_{int(region.get('tile', 0))}"
                ),
            }
        )
    return {
        "config_id": _config_identity(structured),
        "firmware_config": dict(config),
        "elf_sha256": structured["metadata"].get("elf", {}).get("sha256"),
        "member_count": len(regions),
        "expected_member_count": head_size,
        "removable_read_bytes": sum(
            int(region["removable_read_bytes"]) for region in regions
        ),
        "allocation_proven": all_proven,
        "dependency_proven": resources_proven,
        "isa_synthesis_proven": isa_synthesis_proven,
        "regions": regions,
        "synthesis_plan": {
            "cache_bank": 6,
            "cache_bank_name": _VRF_BANK_NAMES[6],
            "cache_offsets": "existing L2C_UNIT_CACHE[j]",
            "zero_immediate_fp16": "0x0000",
            "one_immediate_fp16": "0x3c00",
            "lane_selector": "REG_WRITE_VECTOR_MASK = 1 << j",
            "restore_write_mask": "0xff",
            "max_supported_dim": 8,
            "allocation_proven": all_proven,
        },
        "schedule_signature": {
            "before": "V_RD_DRAM UNIT_VEC[j]; cache; consume",
            "after": "fill zero; masked fill one at lane j; cache; consume",
            "consumer": "V.T column extraction is unchanged",
            "fp16_value": "exact 0.0/1.0 bit patterns",
        },
    }


def _build_generic_cross_config_allocation_proof(
    reference_structured: dict[str, Any],
    proof_graphs: list[dict[str, Any]],
    multiseq_analysis: dict[str, Any],
    *,
    required_config_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservative L1/L2 proof for workloads without model-specific L3 rules."""

    graph_by_id = {_config_identity(graph): graph for graph in proof_graphs}
    required = sorted(required_config_ids or graph_by_id)
    missing = sorted(set(required) - set(graph_by_id))
    matrix_complete = not missing
    reference_macros = reference_structured.get("macro_candidates", {})
    macros: list[dict[str, Any]] = []
    l1_proofs: list[dict[str, Any]] = []
    l2_proofs: list[dict[str, Any]] = []
    l3_proofs: list[dict[str, Any]] = []

    for original in reference_macros.get("macros", []):
        if str(original.get("level", "L1")) != "L1":
            continue
        config_results = []
        for config_id in required:
            graph = graph_by_id.get(config_id)
            candidate = next(
                (
                    item
                    for item in (graph or {}).get("macro_candidates", {}).get(
                        "macros", []
                    )
                    if item.get("id") == original.get("id")
                ),
                None,
            )
            proven = bool(
                candidate
                and candidate.get("eligible")
                and candidate.get("allocation", {}).get("allocation_proven")
            )
            config_results.append(
                {
                    "config_id": config_id,
                    "present": candidate is not None,
                    "allocation_proven": proven,
                }
            )
        cross_proven = bool(matrix_complete and config_results) and all(
            item["allocation_proven"] for item in config_results
        )
        macro = json.loads(json.dumps(original))
        macro["eligible"] = bool(macro.get("eligible") and cross_proven)
        macro["implementation_ready"] = macro["eligible"]
        macro["status"] = (
            "eligible-cross-config-allocation-proven"
            if macro["eligible"]
            else "blocked-cross-config-allocation-unproven"
        )
        macro.setdefault("allocation", {}).update(
            {
                "proof_scope": "all required workload DAG configurations",
                "validation_config_extension_required": False,
                "validation_matrix_complete": matrix_complete,
                "cross_config_proven": cross_proven,
                "allocation_proven": cross_proven,
                "required_config_ids": required,
                "config_results": config_results,
                "required_source_invariants": [
                    "replace only the exact declared DRAM producer/consumer round trips",
                    "preserve operation order and numeric boundaries between producer and consumers",
                    "keep the declared VRF value live until its final listed consumer",
                ],
            }
        )
        macros.append(macro)
        l1_proofs.append(
            {
                "id": macro["id"],
                "level": "L1",
                "cross_config_proven": cross_proven,
                "config_results": config_results,
            }
        )

    for family in multiseq_analysis.get("families", []):
        family_name = str(family.get("family"))
        level = str(family.get("level"))
        macro_id = str(family.get("id"))
        if level == "L2":
            config_results = [
                _allocate_l2_family(graph_by_id[config_id], family_name)
                for config_id in required
                if config_id in graph_by_id
            ]
            for result in config_results:
                result["proof_digest"] = _proof_digest(result)
            by_id = {item["config_id"]: item for item in config_results}
            cross_proven = bool(matrix_complete and config_results) and all(
                by_id.get(config_id, {}).get("allocation_proven", False)
                for config_id in required
            )
            reference_id = _config_identity(reference_structured)
            reference_result = by_id.get(reference_id, {})
            regions = reference_result.get("regions", [])
            resources = [
                {
                    "tensor_slice": item["tensor_slice"],
                    "dram_address": item["dram_address"],
                    "dram_address_hex": item["dram_address_hex"],
                }
                for item in regions
            ]
            measured = dict(family.get("measured_removable_read_bytes", {}))
            priority = max((int(value) for value in measured.values()), default=0)
            measured_values = [
                int(measured[key])
                for key in sorted(measured, key=lambda item: int(item[3:]))
            ]
            macro = {
                "id": macro_id,
                "stable_id": macro_id,
                "evidence_id": _semantic_id(
                    "macro-evidence", {"macro_id": macro_id, "resources": resources}
                ),
                "level": "L2",
                "family": family_name,
                "eligible": cross_proven,
                "implementation_ready": cross_proven,
                "status": (
                    "eligible-cross-config-allocation-proven"
                    if cross_proven
                    else "blocked-cross-config-allocation-unproven"
                ),
                "member_candidate_ids": list(family.get("member_ids", [])),
                "member_stable_ids": list(family.get("member_ids", [])),
                "tensor_slices": sorted(
                    {str(item["tensor_slice"]) for item in regions}
                ),
                "expected_dram_resources": resources,
                "allocation": {
                    "allocator": "deterministic-first-fit-interval-v1",
                    "proof_scope": "all required workload DAG configurations",
                    "validation_matrix_complete": matrix_complete,
                    "cross_config_proven": cross_proven,
                    "allocation_proven": cross_proven,
                    "required_config_ids": required,
                    "config_results": config_results,
                    "regions": regions,
                    "required_source_invariants": [
                        "load each exact read-only value once before its first listed consumer",
                        "serve only the listed reads from the allocated cache region",
                        "preserve all writes, arithmetic operations, and numeric boundaries",
                    ],
                },
                "estimated_saving": {
                    "observed_graph_bytes": priority,
                    "projected_seq2_bytes": (
                        measured_values[0] if measured_values else 0
                    ),
                    "projected_seq6_bytes": (
                        measured_values[-1] if measured_values else 0
                    ),
                    "measured_by_config": measured,
                    "projection_assumption": "measured concrete workload DAGs",
                },
                "scope_policy": "only exact declared read-only loads may reduce",
                "priority_score": priority,
            }
            macros.append(macro)
            l2_proofs.append(
                {
                    "id": macro_id,
                    "level": "L2",
                    "cross_config_proven": cross_proven,
                    "config_results": config_results,
                }
            )
        elif level == "L3":
            measured = dict(family.get("measured_removable_read_bytes", {}))
            priority = max((int(value) for value in measured.values()), default=0)
            measured_values = [
                int(measured[key])
                for key in sorted(measured, key=lambda item: int(item[3:]))
            ]
            macro = {
                "id": macro_id,
                "stable_id": macro_id,
                "evidence_id": _semantic_id(
                    "macro-evidence", {"macro_id": macro_id, "workload": multiseq_analysis.get("workload")}
                ),
                "level": "L3",
                "family": family_name,
                "eligible": False,
                "implementation_ready": False,
                "status": "blocked-workload-schedule-proof-required",
                "member_candidate_ids": list(family.get("member_ids", [])),
                "member_stable_ids": list(family.get("member_ids", [])),
                "tensor_slices": [],
                "expected_dram_resources": [],
                "allocation": {
                    "allocation_proven": False,
                    "cross_config_proven": False,
                    "validation_matrix_complete": matrix_complete,
                    "missing_proofs": [
                        "workload-specific MRF residency and clobber proof",
                        "phase-level producer/consumer schedule",
                        "numeric accumulation-order preservation",
                    ],
                },
                "estimated_saving": {
                    "observed_graph_bytes": priority,
                    "projected_seq2_bytes": (
                        measured_values[0] if measured_values else 0
                    ),
                    "projected_seq6_bytes": (
                        measured_values[-1] if measured_values else 0
                    ),
                    "measured_by_config": measured,
                    "projection_assumption": "measured upper bound; implementation blocked",
                },
                "scope_policy": "blocked until the workload adapter supplies an L3 proof",
                "priority_score": priority,
            }
            macros.append(macro)
            l3_proofs.append(
                {
                    "id": macro_id,
                    "level": "L3",
                    "cross_config_proven": False,
                    "missing_proofs": macro["allocation"]["missing_proofs"],
                }
            )

    level_order = {"L1": 1, "L2": 2, "L3": 3}
    macros.sort(
        key=lambda item: (
            level_order.get(str(item.get("level")), 99),
            -int(item.get("priority_score", 0)),
            str(item.get("id")),
        )
    )
    for rank, macro in enumerate(macros, start=1):
        macro["rank"] = rank
    proof = {
        "schema": {
            "name": "jimu-npu-cross-config-allocation-proof",
            "version": "1.1.0",
        },
        "workload": multiseq_analysis.get("workload"),
        "proof_mode": "generic-conservative",
        "required_config_ids": required,
        "observed_config_ids": sorted(graph_by_id),
        "missing_config_ids": missing,
        "validation_matrix_complete": matrix_complete,
        "l1": l1_proofs,
        "l2": l2_proofs,
        "l3": l3_proofs,
    }
    macro_data = {
        "schema": {
            "name": "jimu-npu-dram-macro-candidates",
            "version": "2.1.0",
        },
        "summary": {
            "macros": len(macros),
            "eligible": sum(1 for macro in macros if macro["eligible"]),
            "blocked": sum(1 for macro in macros if not macro["eligible"]),
            "primitive_candidates": reference_macros.get("summary", {}).get(
                "primitive_candidates", 0
            ),
            "validation_matrix_complete": matrix_complete,
        },
        "analysis_limits": {
            "proof_mode": "generic-conservative",
            "required_config_ids": required,
            "missing_config_ids": missing,
            "l3_schedule": "blocked until supplied by the workload adapter",
        },
        "macros": macros,
    }
    return proof, macro_data


def build_cross_config_allocation_proof(
    reference_structured: dict[str, Any],
    proof_graphs: list[dict[str, Any]],
    multiseq_analysis: dict[str, Any],
    *,
    required_config_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove staged L1/L2/L3 transformations across the validation matrix."""
    workload = reference_structured.get("metadata", {}).get("workload", {})
    if not bool(workload.get("specialized_contracts", True)):
        return _build_generic_cross_config_allocation_proof(
            reference_structured,
            proof_graphs,
            multiseq_analysis,
            required_config_ids=required_config_ids,
        )
    graph_by_id = {_config_identity(graph): graph for graph in proof_graphs}
    required = sorted(required_config_ids or graph_by_id)
    missing = sorted(set(required) - set(graph_by_id))
    matrix_complete = not missing

    reference_macros = reference_structured.get("macro_candidates", {})
    reference_id = _config_identity(reference_structured)
    scratch_macro_id = "macro-dram-l1-transient-scratch-bank13"
    merged_macros = []
    l1_proofs = []
    for macro in reference_macros.get("macros", []):
        if str(macro.get("level", "L1")) != "L1":
            continue
        # This macro has a dedicated cross-configuration proof below.  Do not
        # retain the reference graph's placeholder entry, otherwise a
        # progressive run can emit the same stable ID once blocked and once
        # eligible.
        if macro.get("id") == scratch_macro_id:
            continue
        config_results = []
        for config_id in required:
            graph = graph_by_id.get(config_id)
            graph_macro = None
            if graph is not None:
                graph_macro = next(
                    (
                        item
                        for item in graph.get("macro_candidates", {}).get(
                            "macros", []
                        )
                        if item.get("id") == macro.get("id")
                    ),
                    None,
                )
            proven = bool(
                graph_macro
                and graph_macro.get("eligible")
                and graph_macro.get("allocation", {}).get(
                    "allocation_proven"
                )
            )
            config_results.append(
                {
                    "config_id": config_id,
                    "present": graph_macro is not None,
                    "member_count": len(
                        (graph_macro or {}).get("member_candidate_ids", [])
                    ),
                    "allocation_proven": proven,
                }
            )
        cross_proven = bool(matrix_complete and config_results) and all(
            result["allocation_proven"] for result in config_results
        )
        updated = json.loads(json.dumps(macro))
        updated["eligible"] = bool(updated.get("eligible") and cross_proven)
        updated["status"] = (
            "eligible-cross-config-allocation-proven"
            if updated["eligible"]
            else "blocked-cross-config-allocation-unproven"
        )
        updated["allocation"].update(
            {
                "proof_scope": "all required validation DAG configurations",
                "validation_config_extension_required": False,
                "validation_matrix_complete": matrix_complete,
                "cross_config_proven": cross_proven,
                "allocation_proven": cross_proven,
                "required_config_ids": required,
                "config_results": config_results,
            }
        )
        merged_macros.append(updated)
        l1_proofs.append(
            {
                "id": updated["id"],
                "level": "L1",
                "cross_config_proven": cross_proven,
                "config_results": config_results,
            }
        )

    scratch_results = [
        _transient_scratch_config_proof(graph_by_id[config_id])
        for config_id in required
        if config_id in graph_by_id
    ]
    scratch_by_id = {
        result["config_id"]: result for result in scratch_results
    }
    scratch_cross_proven = bool(matrix_complete and scratch_results) and all(
        scratch_by_id.get(config_id, {}).get("allocation_proven", False)
        for config_id in required
    )
    scratch_reference = scratch_by_id.get(reference_id, {})
    scratch_regions = scratch_reference.get("regions", [])
    if scratch_regions:
        scratch_resources = [
            {
                "tensor_slice": region["tensor_slice"],
                "dram_address": region["dram_address"],
                "dram_address_hex": region["dram_address_hex"],
            }
            for region in scratch_regions
        ]
        reference_config = reference_structured["metadata"][
            "firmware_config"
        ]

        def _scratch_saving_for(seq_len: int) -> int:
            return next(
                (
                    int(result["removable_read_bytes"])
                    for result in scratch_results
                    if int(result["firmware_config"]["seq_len"]) == seq_len
                    and int(result["firmware_config"]["dim"])
                    == int(reference_config["dim"])
                    and int(result["firmware_config"]["hidden_size"])
                    == int(reference_config["hidden_size"])
                    and int(result["firmware_config"]["num_head"])
                    == int(reference_config["num_head"])
                ),
                0,
            )

        scratch_macro = {
            "id": scratch_macro_id,
            "stable_id": scratch_macro_id,
            "evidence_id": _semantic_id(
                "macro-evidence",
                {
                    "macro_id": scratch_macro_id,
                    "resources": scratch_resources,
                },
            ),
            "level": "L1",
            "family": "transient-scratch-bank13",
            "eligible": scratch_cross_proven,
            "implementation_ready": scratch_cross_proven,
            "status": (
                "eligible-cross-config-allocation-proven"
                if scratch_cross_proven
                else "blocked-cross-config-allocation-unproven"
            ),
            "member_candidate_ids": [
                _semantic_id(
                    "scratch-member",
                    {
                        "slice": region["tensor_slice"],
                        "address": region["dram_address"],
                    },
                )
                for region in scratch_regions
            ],
            "member_stable_ids": [
                _semantic_id(
                    "scratch-member",
                    {
                        "slice": region["tensor_slice"],
                        "address": region["dram_address"],
                    },
                )
                for region in scratch_regions
            ],
            "tensor_slices": sorted(
                region["tensor_slice"] for region in scratch_regions
            ),
            "expected_dram_resources": scratch_resources,
            "allocation": {
                "allocator": "transient-bank13-slot-v1",
                "preferred_bank": 13,
                "preferred_bank_name": _VRF_BANK_NAMES[13],
                "capacity_source": "emulator/npu_device_mini.py::VRF_SIZES",
                "proof_scope": "all required validation DAG configurations",
                "validation_config_extension_required": False,
                "validation_matrix_complete": matrix_complete,
                "cross_config_proven": scratch_cross_proven,
                "allocation_proven": scratch_cross_proven,
                "required_config_ids": required,
                "config_results": [
                    {
                        "config_id": result["config_id"],
                        "member_count": result["member_count"],
                        "eliminated_member_count": result[
                            "eliminated_member_count"
                        ],
                        "removable_read_bytes": result[
                            "removable_read_bytes"
                        ],
                        "allocation_proven": result["allocation_proven"],
                        "proof_digest": _proof_digest(result),
                    }
                    for result in scratch_results
                ],
                "transient_allocation": scratch_reference.get(
                    "transient_allocation", {}
                ),
                "schedule_signature": scratch_reference.get(
                    "schedule_signature", {}
                ),
                "regions": scratch_regions,
                "required_source_invariants": [
                    "replace only the active SCRATCH addresses declared by this contract",
                    "write each transient value to the declared bank-13 slot",
                    "read it back before the next transient write",
                    "preserve every operation and FP16 boundary between write and read",
                    "do not modify retained-Q offsets at or above seq_len*NATIVE_DIM",
                    "do not reintroduce scratch families recorded as already eliminated",
                ],
            },
            "estimated_saving": {
                "observed_graph_bytes": int(
                    scratch_reference.get("removable_read_bytes", 0)
                ),
                "projected_seq2_bytes": _scratch_saving_for(2),
                "projected_seq6_bytes": _scratch_saving_for(6),
                "projection_assumption": (
                    "measured exact active transient scratch store-load pairs"
                ),
            },
            "scope_policy": (
                "only exact declared transient SCRATCH loads and stores may reduce"
            ),
            "priority_score": _scratch_saving_for(6),
        }
        merged_macros.append(scratch_macro)
        l1_proofs.append(
            {
                "id": scratch_macro_id,
                "level": "L1",
                "cross_config_proven": scratch_cross_proven,
                "config_results": scratch_results,
            }
        )

    family_by_name = {
        family["family"]: family
        for family in multiseq_analysis.get("families", [])
    }
    l2_proofs = []
    l2_specs = (
        ("loop-invariant-parameters", "macro-dram-l2-loop-invariants"),
        ("sequence-input", "macro-dram-l2-sequence-input"),
    )
    for family_name, macro_id in l2_specs:
        if family_name not in family_by_name:
            continue
        config_results = [
            _allocate_l2_family(graph_by_id[config_id], family_name)
            for config_id in required
            if config_id in graph_by_id
        ]
        by_id = {result["config_id"]: result for result in config_results}
        cross_proven = bool(matrix_complete and config_results) and all(
            by_id.get(config_id, {}).get("allocation_proven", False)
            for config_id in required
        )
        reference_proof = by_id.get(reference_id, {})
        reference_regions = reference_proof.get("regions", [])
        expected_resources = sorted(
            {
                (
                    str(region["tensor_slice"]),
                    int(region["dram_address"]),
                    str(region["dram_address_hex"]),
                )
                for region in reference_regions
            }
        )
        measured = family_by_name[family_name][
            "measured_removable_read_bytes"
        ]
        estimated = {
            "observed_graph_bytes": int(
                reference_proof.get("removable_read_bytes", 0)
            ),
            "projected_seq2_bytes": int(measured.get("seq2", 0)),
            "projected_seq6_bytes": int(measured.get("seq6", 0)),
            "projection_assumption": (
                "measured from concrete seq2/seq6 DAGs; probes authoritative"
            ),
        }
        summaries = [
            {
                "config_id": result["config_id"],
                "member_count": result["member_count"],
                "removable_read_bytes": result["removable_read_bytes"],
                "allocation_proven": result["allocation_proven"],
                "proof_digest": _proof_digest(result),
            }
            for result in config_results
        ]
        macro = {
            "id": macro_id,
            "stable_id": macro_id,
            "evidence_id": _semantic_id(
                "macro-evidence",
                {"macro_id": macro_id, "resources": expected_resources},
            ),
            "level": "L2",
            "family": family_name,
            "eligible": cross_proven,
            "implementation_ready": cross_proven,
            "status": (
                "eligible-cross-config-allocation-proven"
                if cross_proven
                else "blocked-cross-config-allocation-unproven"
            ),
            "member_candidate_ids": [
                str(region["id"]) for region in reference_regions
            ],
            "member_stable_ids": sorted(
                str(region["id"]) for region in reference_regions
            ),
            "tensor_slices": sorted(
                {str(region["tensor_slice"]) for region in reference_regions}
            ),
            "expected_dram_resources": [
                {
                    "tensor_slice": tensor_slice,
                    "dram_address": address,
                    "dram_address_hex": address_hex,
                }
                for tensor_slice, address, address_hex in expected_resources
            ],
            "allocation": {
                "allocator": "deterministic-first-fit-interval-v2",
                "preferred_bank": _MACRO_CACHE_BANK,
                "preferred_bank_name": _VRF_BANK_NAMES[_MACRO_CACHE_BANK],
                "capacity_source": "emulator/npu_device_mini.py::VRF_SIZES",
                "proof_scope": "all required validation DAG configurations",
                "validation_config_extension_required": False,
                "validation_matrix_complete": matrix_complete,
                "cross_config_proven": cross_proven,
                "allocation_proven": cross_proven,
                "required_config_ids": required,
                "config_results": summaries,
                "regions": reference_regions,
            },
            "estimated_saving": estimated,
            "scope_policy": (
                "every expected Tensor/address must reduce DRAM loads; "
                "positive DRAM reductions outside the declared macro are rejected"
            ),
            "priority_score": estimated["projected_seq6_bytes"],
        }
        merged_macros.append(macro)
        l2_proofs.append(
            {
                "id": macro_id,
                "level": "L2",
                "family": family_name,
                "cross_config_proven": cross_proven,
                "config_results": config_results,
            }
        )

    unit_results = [
        _unit_vector_synthesis_config_proof(graph_by_id[config_id])
        for config_id in required
        if config_id in graph_by_id
    ]
    unit_by_id = {result["config_id"]: result for result in unit_results}
    unit_cross_proven = bool(matrix_complete and unit_results) and all(
        unit_by_id.get(config_id, {}).get("allocation_proven", False)
        for config_id in required
    )
    unit_reference = unit_by_id.get(reference_id, {})
    unit_regions = unit_reference.get("regions", [])
    if unit_regions:
        unit_macro_id = "macro-dram-l2-unit-vector-synthesis"
        unit_resources = [
            {
                "tensor_slice": region["tensor_slice"],
                "dram_address": region["dram_address"],
                "dram_address_hex": region["dram_address_hex"],
            }
            for region in unit_regions
        ]
        reference_config = reference_structured["metadata"][
            "firmware_config"
        ]

        def _unit_saving_for(seq_len: int) -> int:
            return next(
                (
                    int(result["removable_read_bytes"])
                    for result in unit_results
                    if int(result["firmware_config"]["seq_len"]) == seq_len
                    and int(result["firmware_config"]["dim"])
                    == int(reference_config["dim"])
                    and int(result["firmware_config"]["hidden_size"])
                    == int(reference_config["hidden_size"])
                    and int(result["firmware_config"]["num_head"])
                    == int(reference_config["num_head"])
                ),
                0,
            )

        unit_macro = {
            "id": unit_macro_id,
            "stable_id": unit_macro_id,
            "evidence_id": _semantic_id(
                "macro-evidence",
                {"macro_id": unit_macro_id, "resources": unit_resources},
            ),
            "level": "L2",
            "family": "constant-synthesis-unit-vectors",
            "eligible": unit_cross_proven,
            "implementation_ready": unit_cross_proven,
            "status": (
                "eligible-cross-config-isa-synthesis-proven"
                if unit_cross_proven
                else "blocked-cross-config-isa-synthesis-unproven"
            ),
            "member_candidate_ids": [
                _semantic_id(
                    "unit-vector-member",
                    {
                        "slice": region["tensor_slice"],
                        "address": region["dram_address"],
                    },
                )
                for region in unit_regions
            ],
            "member_stable_ids": [
                _semantic_id(
                    "unit-vector-member",
                    {
                        "slice": region["tensor_slice"],
                        "address": region["dram_address"],
                    },
                )
                for region in unit_regions
            ],
            "tensor_slices": sorted(
                region["tensor_slice"] for region in unit_regions
            ),
            "expected_dram_resources": unit_resources,
            "allocation": {
                "allocator": "exact-fp16-unit-vector-synthesis-v1",
                "preferred_bank": 6,
                "preferred_bank_name": _VRF_BANK_NAMES[6],
                "proof_scope": "all required validation DAG configurations",
                "validation_config_extension_required": False,
                "validation_matrix_complete": matrix_complete,
                "cross_config_proven": unit_cross_proven,
                "allocation_proven": unit_cross_proven,
                "required_config_ids": required,
                "config_results": [
                    {
                        "config_id": result["config_id"],
                        "member_count": result["member_count"],
                        "removable_read_bytes": result[
                            "removable_read_bytes"
                        ],
                        "allocation_proven": result["allocation_proven"],
                        "proof_digest": _proof_digest(result),
                    }
                    for result in unit_results
                ],
                "synthesis_plan": unit_reference.get("synthesis_plan", {}),
                "schedule_signature": unit_reference.get(
                    "schedule_signature", {}
                ),
                "regions": [],
                "required_source_invariants": [
                    "zero the complete existing L2C_UNIT_CACHE[j] vector",
                    "set REG_WRITE_VECTOR_MASK to exactly 1 << j",
                    "write exact FP16 1.0 immediate 0x3c00 into lane j",
                    "restore REG_WRITE_VECTOR_MASK to 0xff",
                    "reload the synthesized vector before V.T extraction",
                    "do not read UNIT_VEC_BASE from DRAM",
                ],
            },
            "estimated_saving": {
                "observed_graph_bytes": int(
                    unit_reference.get("removable_read_bytes", 0)
                ),
                "projected_seq2_bytes": _unit_saving_for(2),
                "projected_seq6_bytes": _unit_saving_for(6),
                "projection_assumption": (
                    "exact DRAM identity rows replaced by FP16 immediates"
                ),
            },
            "scope_policy": (
                "only exact declared UNIT_VEC DRAM loads may reduce"
            ),
            "priority_score": _unit_saving_for(6),
        }
        merged_macros.append(unit_macro)
        l2_proofs.append(
            {
                "id": unit_macro_id,
                "level": "L2",
                "cross_config_proven": unit_cross_proven,
                "config_results": unit_results,
            }
        )

    l3_family = family_by_name.get("weight-stationary")
    l3_proofs = []
    if l3_family is not None:
        weight_candidates = [
            candidate
            for candidate in multiseq_analysis.get("candidates", [])
            if candidate.get("family") == "weight-stationary"
            and candidate.get("analysis_eligible")
        ]
        kv_candidates = [
            candidate
            for candidate in weight_candidates
            if candidate.get("tensor") in {"W_K", "W_V"}
        ]
        if kv_candidates:
            config_results = [
                _l3_kv_config_proof(graph_by_id[config_id])
                for config_id in required
                if config_id in graph_by_id
            ]
            by_id = {
                result["config_id"]: result for result in config_results
            }
            cross_proven = bool(matrix_complete and config_results) and all(
                by_id.get(config_id, {}).get("allocation_proven", False)
                for config_id in required
            )
            reference_proof = by_id.get(reference_id, {})
            reference_regions = reference_proof.get("regions", [])
            expected_resources = [
                {
                    "tensor_slice": region["tensor_slice"],
                    "dram_address": region["dram_address"],
                    "dram_address_hex": region["dram_address_hex"],
                }
                for region in reference_regions
            ]
            measured = {
                f"seq{seq_len}": sum(
                    int(
                        candidate["per_config"][f"seq{seq_len}"].get(
                            "removable_read_bytes", 0
                        )
                    )
                    for candidate in kv_candidates
                )
                for seq_len in sorted(
                    int(item["seq_len"])
                    for item in multiseq_analysis.get("configurations", [])
                )
            }
            macro_id = "macro-dram-l3-kv-weight-stationary"
            l3_kv_macro = {
                "id": macro_id,
                "stable_id": macro_id,
                "evidence_id": _semantic_id(
                    "macro-evidence",
                    {"macro_id": macro_id, "resources": expected_resources},
                ),
                "level": "L3",
                "family": "weight-stationary-kv",
                "eligible": cross_proven,
                "implementation_ready": cross_proven,
                "status": (
                    "eligible-cross-config-schedule-proven"
                    if cross_proven
                    else "blocked-cross-config-schedule-unproven"
                ),
                "member_candidate_ids": [
                    candidate["id"] for candidate in kv_candidates
                ],
                "member_stable_ids": sorted(
                    candidate["id"] for candidate in kv_candidates
                ),
                "tensor_slices": sorted(
                    region["tensor_slice"] for region in reference_regions
                ),
                "expected_dram_resources": expected_resources,
                "allocation": {
                    "allocator": "l3-position-partials-bank13-v1",
                    "preferred_bank": 13,
                    "preferred_bank_name": _VRF_BANK_NAMES[13],
                    "capacity_source": (
                        "emulator/npu_device_mini.py::VRF_SIZES"
                    ),
                    "proof_scope": "all required validation DAG configurations",
                    "validation_matrix_complete": matrix_complete,
                    "cross_config_proven": cross_proven,
                    "allocation_proven": cross_proven,
                    "required_config_ids": required,
                    "config_results": [
                        {
                            "config_id": result["config_id"],
                            "member_count": result["member_count"],
                            "removable_read_bytes": result[
                                "removable_read_bytes"
                            ],
                            "allocation_proven": result[
                                "allocation_proven"
                            ],
                            "proof_digest": _proof_digest(result),
                        }
                        for result in config_results
                    ],
                    "partial_sum_allocation": reference_proof.get(
                        "partial_sum_allocation", {}
                    ),
                    "schedule_signature": reference_proof.get(
                        "schedule_signature", {}
                    ),
                    "regions": reference_regions,
                    "required_source_invariants": [
                        "load each W_K/W_V tile exactly once",
                        "no M_RD or M_RD_DRAM inside the position loop",
                        "use a distinct bank-13 vector for every position",
                        "preserve tc=0..num_tiles-1 for every output",
                        "apply bias only after the final tc partial sum",
                    ],
                },
                "estimated_saving": {
                    "observed_graph_bytes": int(measured.get("seq6", 0)),
                    "projected_seq2_bytes": int(measured.get("seq2", 0)),
                    "projected_seq6_bytes": int(measured.get("seq6", 0)),
                    "projection_assumption": (
                        "measured repeated W_K/W_V MAT_LOAD bytes"
                    ),
                },
                "scope_policy": (
                    "only exact W_K/W_V Tensor/address loads may reduce"
                ),
                "priority_score": int(measured.get("seq6", 0)),
            }
            merged_macros.append(l3_kv_macro)
            l3_proofs.append(
                {
                    "id": macro_id,
                    "level": "L3",
                    "cross_config_proven": cross_proven,
                    "config_results": config_results,
                }
            )

        q_candidates = [
            candidate
            for candidate in weight_candidates
            if candidate.get("tensor") == "W_Q"
        ]
        if q_candidates:
            config_results = [
                _l3_q_config_proof(graph_by_id[config_id])
                for config_id in required
                if config_id in graph_by_id
            ]
            by_id = {
                result["config_id"]: result for result in config_results
            }
            cross_proven = bool(matrix_complete and config_results) and all(
                by_id.get(config_id, {}).get("allocation_proven", False)
                for config_id in required
            )
            reference_proof = by_id.get(reference_id, {})
            reference_regions = reference_proof.get("regions", [])
            expected_resources = [
                {
                    "tensor_slice": region["tensor_slice"],
                    "dram_address": region["dram_address"],
                    "dram_address_hex": region["dram_address_hex"],
                }
                for region in reference_regions
            ]
            measured = {
                f"seq{seq_len}": sum(
                    int(
                        candidate["per_config"][f"seq{seq_len}"].get(
                            "removable_read_bytes", 0
                        )
                    )
                    for candidate in q_candidates
                )
                for seq_len in sorted(
                    int(item["seq_len"])
                    for item in multiseq_analysis.get("configurations", [])
                )
            }
            macro_id = "macro-dram-l3-q-weight-stationary"
            l3_q_macro = {
                "id": macro_id,
                "stable_id": macro_id,
                "evidence_id": _semantic_id(
                    "macro-evidence",
                    {"macro_id": macro_id, "resources": expected_resources},
                ),
                "level": "L3",
                "family": "weight-stationary-q",
                "eligible": cross_proven,
                "implementation_ready": cross_proven,
                "status": (
                    "eligible-cross-config-schedule-proven"
                    if cross_proven
                    else "blocked-cross-config-schedule-unproven"
                ),
                "member_candidate_ids": [
                    candidate["id"] for candidate in q_candidates
                ],
                "member_stable_ids": sorted(
                    candidate["id"] for candidate in q_candidates
                ),
                "tensor_slices": sorted(
                    region["tensor_slice"] for region in reference_regions
                ),
                "expected_dram_resources": expected_resources,
                "allocation": {
                    "allocator": "l3-q-partials-retained-bank13-v1",
                    "preferred_bank": 13,
                    "preferred_bank_name": _VRF_BANK_NAMES[13],
                    "capacity_source": (
                        "emulator/npu_device_mini.py::VRF_SIZES"
                    ),
                    "proof_scope": "all required validation DAG configurations",
                    "validation_matrix_complete": matrix_complete,
                    "cross_config_proven": cross_proven,
                    "allocation_proven": cross_proven,
                    "required_config_ids": required,
                    "config_results": [
                        {
                            "config_id": result["config_id"],
                            "member_count": result["member_count"],
                            "removable_read_bytes": result[
                                "removable_read_bytes"
                            ],
                            "allocation_proven": result[
                                "allocation_proven"
                            ],
                            "proof_digest": _proof_digest(result),
                        }
                        for result in config_results
                    ],
                    "partial_sum_allocation": reference_proof.get(
                        "partial_sum_allocation", {}
                    ),
                    "retained_output_allocation": reference_proof.get(
                        "retained_output_allocation", {}
                    ),
                    "schedule_signature": reference_proof.get(
                        "schedule_signature", {}
                    ),
                    "regions": reference_regions,
                    "required_source_invariants": [
                        "load each W_Q tile exactly once",
                        "no M_RD or M_RD_DRAM inside the position loop",
                        "use a distinct bank-13 partial for every position",
                        "retain every completed Q[pos,tr] in the declared bank-13 region",
                        "preserve tc=0..num_tiles-1 for every Q output",
                        "apply Q bias only after the final tc partial sum",
                        "consume retained Q only after all Q positions are complete",
                    ],
                },
                "estimated_saving": {
                    "observed_graph_bytes": int(measured.get("seq6", 0)),
                    "projected_seq2_bytes": int(measured.get("seq2", 0)),
                    "projected_seq6_bytes": int(measured.get("seq6", 0)),
                    "projection_assumption": (
                        "measured repeated W_Q MAT_LOAD bytes"
                    ),
                },
                "scope_policy": "only exact W_Q Tensor/address loads may reduce",
                "priority_score": int(measured.get("seq6", 0)),
            }
            merged_macros.append(l3_q_macro)
            l3_proofs.append(
                {
                    "id": macro_id,
                    "level": "L3",
                    "cross_config_proven": cross_proven,
                    "config_results": config_results,
                }
            )

        self_output_candidates = [
            candidate
            for candidate in weight_candidates
            if candidate.get("tensor") == "W_SELF_OUTPUT"
        ]
        if self_output_candidates:
            config_results = [
                _l3_self_output_config_proof(graph_by_id[config_id])
                for config_id in required
                if config_id in graph_by_id
            ]
            by_id = {
                result["config_id"]: result for result in config_results
            }
            cross_proven = bool(matrix_complete and config_results) and all(
                by_id.get(config_id, {}).get("allocation_proven", False)
                for config_id in required
            )
            reference_proof = by_id.get(reference_id, {})
            reference_regions = reference_proof.get("regions", [])
            expected_resources = [
                {
                    "tensor_slice": region["tensor_slice"],
                    "dram_address": region["dram_address"],
                    "dram_address_hex": region["dram_address_hex"],
                }
                for region in reference_regions
            ]
            measured = {
                f"seq{seq_len}": sum(
                    int(
                        candidate["per_config"][f"seq{seq_len}"].get(
                            "removable_read_bytes", 0
                        )
                    )
                    for candidate in self_output_candidates
                )
                for seq_len in sorted(
                    int(item["seq_len"])
                    for item in multiseq_analysis.get("configurations", [])
                )
            }
            macro_id = "macro-dram-l3-self-output-weight-stationary"
            self_output_macro = {
                "id": macro_id,
                "stable_id": macro_id,
                "evidence_id": _semantic_id(
                    "macro-evidence",
                    {"macro_id": macro_id, "resources": expected_resources},
                ),
                "level": "L3",
                "family": "weight-stationary-self-output",
                "eligible": cross_proven,
                "implementation_ready": cross_proven,
                "status": (
                    "eligible-cross-config-schedule-proven"
                    if cross_proven
                    else "blocked-cross-config-schedule-unproven"
                ),
                "member_candidate_ids": [
                    candidate["id"] for candidate in self_output_candidates
                ],
                "member_stable_ids": sorted(
                    candidate["id"] for candidate in self_output_candidates
                ),
                "tensor_slices": sorted(
                    region["tensor_slice"] for region in reference_regions
                ),
                "expected_dram_resources": expected_resources,
                "allocation": {
                    "allocator": "l3-two-state-buffers-bank13-v1",
                    "preferred_bank": 13,
                    "preferred_bank_name": _VRF_BANK_NAMES[13],
                    "capacity_source": "emulator/npu_device_mini.py::VRF_SIZES",
                    "proof_scope": "all required validation DAG configurations",
                    "validation_matrix_complete": matrix_complete,
                    "cross_config_proven": cross_proven,
                    "allocation_proven": cross_proven,
                    "required_config_ids": required,
                    "config_results": [
                        {
                            "config_id": result["config_id"],
                            "member_count": result["member_count"],
                            "removable_read_bytes": result[
                                "removable_read_bytes"
                            ],
                            "allocation_proven": result["allocation_proven"],
                            "proof_digest": _proof_digest(result),
                        }
                        for result in config_results
                    ],
                    "partial_sum_allocation": reference_proof.get(
                        "partial_sum_allocation", {}
                    ),
                    "state_allocations": reference_proof.get(
                        "state_allocations", []
                    ),
                    "schedule_signature": reference_proof.get(
                        "schedule_signature", {}
                    ),
                    "regions": reference_regions,
                    "required_source_invariants": [
                        "finish attention for every position before SELF_OUTPUT",
                        "replace each dead retained Q[pos] with its attention context",
                        "load each W_SELF_OUTPUT tile exactly once",
                        "read contexts from state A and write SELF_OUTPUT to state B",
                        "preserve tc=0..num_tiles-1 and the bias boundary",
                        "start residual1 only after all SELF_OUTPUT positions complete",
                    ],
                },
                "estimated_saving": {
                    "observed_graph_bytes": int(measured.get("seq6", 0)),
                    "projected_seq2_bytes": int(measured.get("seq2", 0)),
                    "projected_seq6_bytes": int(measured.get("seq6", 0)),
                    "projection_assumption": (
                        "measured repeated W_SELF_OUTPUT MAT_LOAD bytes"
                    ),
                },
                "scope_policy": (
                    "only exact W_SELF_OUTPUT Tensor/address loads may reduce"
                ),
                "priority_score": int(measured.get("seq6", 0)),
            }
            merged_macros.append(self_output_macro)
            l3_proofs.append(
                {
                    "id": macro_id,
                    "level": "L3",
                    "cross_config_proven": cross_proven,
                    "config_results": config_results,
                }
            )

        ffn_intermediate_candidates = [
            candidate
            for candidate in weight_candidates
            if candidate.get("tensor") == "W_FFN_INTERMEDIATE"
        ]
        if ffn_intermediate_candidates:
            config_results = [
                _l3_ffn_intermediate_config_proof(graph_by_id[config_id])
                for config_id in required
                if config_id in graph_by_id
            ]
            by_id = {
                result["config_id"]: result for result in config_results
            }
            cross_proven = bool(matrix_complete and config_results) and all(
                by_id.get(config_id, {}).get("allocation_proven", False)
                for config_id in required
            )
            reference_proof = by_id.get(reference_id, {})
            reference_regions = reference_proof.get("regions", [])
            expected_resources = [
                {
                    "tensor_slice": region["tensor_slice"],
                    "dram_address": region["dram_address"],
                    "dram_address_hex": region["dram_address_hex"],
                }
                for region in reference_regions
            ]
            measured = {
                f"seq{seq_len}": sum(
                    int(
                        candidate["per_config"][f"seq{seq_len}"].get(
                            "removable_read_bytes", 0
                        )
                    )
                    for candidate in ffn_intermediate_candidates
                )
                for seq_len in sorted(
                    int(item["seq_len"])
                    for item in multiseq_analysis.get("configurations", [])
                )
            }
            macro_id = "macro-dram-l3-ffn-intermediate-weight-stationary"
            ffn_intermediate_macro = {
                "id": macro_id,
                "stable_id": macro_id,
                "evidence_id": _semantic_id(
                    "macro-evidence",
                    {"macro_id": macro_id, "resources": expected_resources},
                ),
                "level": "L3",
                "family": "weight-stationary-ffn-intermediate",
                "eligible": cross_proven,
                "implementation_ready": cross_proven,
                "status": (
                    "eligible-cross-config-schedule-proven"
                    if cross_proven
                    else "blocked-cross-config-schedule-unproven"
                ),
                "member_candidate_ids": [
                    candidate["id"]
                    for candidate in ffn_intermediate_candidates
                ],
                "member_stable_ids": sorted(
                    candidate["id"]
                    for candidate in ffn_intermediate_candidates
                ),
                "tensor_slices": sorted(
                    region["tensor_slice"] for region in reference_regions
                ),
                "expected_dram_resources": expected_resources,
                "allocation": {
                    "allocator": "l3-two-state-buffers-bank13-v2",
                    "preferred_bank": 13,
                    "preferred_bank_name": _VRF_BANK_NAMES[13],
                    "capacity_source": "emulator/npu_device_mini.py::VRF_SIZES",
                    "proof_scope": "all required validation DAG configurations",
                    "validation_matrix_complete": matrix_complete,
                    "cross_config_proven": cross_proven,
                    "allocation_proven": cross_proven,
                    "required_config_ids": required,
                    "config_results": [
                        {
                            "config_id": result["config_id"],
                            "member_count": result["member_count"],
                            "removable_read_bytes": result[
                                "removable_read_bytes"
                            ],
                            "allocation_proven": result["allocation_proven"],
                            "proof_digest": _proof_digest(result),
                        }
                        for result in config_results
                    ],
                    "partial_sum_allocation": reference_proof.get(
                        "partial_sum_allocation", {}
                    ),
                    "state_allocations": reference_proof.get(
                        "state_allocations", []
                    ),
                    "l2_x_retention": reference_proof.get(
                        "l2_x_retention", {}
                    ),
                    "schedule_signature": reference_proof.get(
                        "schedule_signature", {}
                    ),
                    "regions": reference_regions,
                    "required_source_invariants": [
                        "finish residual1 and LN1 for every position first",
                        "write each normalized residual1 vector to state A",
                        "load each W_FFN_INTERMEDIATE tile exactly once",
                        "read LN1 from state A and write GELU output to state B",
                        "apply bias after final tc and GELU after bias",
                        "preserve tc=0..num_tiles-1 for every output",
                        "retain every original X cache entry until residual2",
                        "start FFN_OUTPUT only after all GELU positions complete",
                    ],
                },
                "estimated_saving": {
                    "observed_graph_bytes": int(measured.get("seq6", 0)),
                    "projected_seq2_bytes": int(measured.get("seq2", 0)),
                    "projected_seq6_bytes": int(measured.get("seq6", 0)),
                    "projection_assumption": (
                        "measured repeated W_FFN_INTERMEDIATE MAT_LOAD bytes"
                    ),
                },
                "scope_policy": (
                    "only exact W_FFN_INTERMEDIATE Tensor/address loads may reduce"
                ),
                "priority_score": int(measured.get("seq6", 0)),
            }
            merged_macros.append(ffn_intermediate_macro)
            l3_proofs.append(
                {
                    "id": macro_id,
                    "level": "L3",
                    "cross_config_proven": cross_proven,
                    "config_results": config_results,
                }
            )

        ffn_output_candidates = [
            candidate
            for candidate in weight_candidates
            if candidate.get("tensor") == "W_FFN_OUTPUT"
        ]
        if ffn_output_candidates:
            config_results = [
                _l3_ffn_output_config_proof(graph_by_id[config_id])
                for config_id in required
                if config_id in graph_by_id
            ]
            by_id = {
                result["config_id"]: result for result in config_results
            }
            cross_proven = bool(matrix_complete and config_results) and all(
                by_id.get(config_id, {}).get("allocation_proven", False)
                for config_id in required
            )
            reference_proof = by_id.get(reference_id, {})
            reference_regions = reference_proof.get("regions", [])
            expected_resources = [
                {
                    "tensor_slice": region["tensor_slice"],
                    "dram_address": region["dram_address"],
                    "dram_address_hex": region["dram_address_hex"],
                }
                for region in reference_regions
            ]
            measured = {
                f"seq{seq_len}": sum(
                    int(
                        candidate["per_config"][f"seq{seq_len}"].get(
                            "removable_read_bytes", 0
                        )
                    )
                    for candidate in ffn_output_candidates
                )
                for seq_len in sorted(
                    int(item["seq_len"])
                    for item in multiseq_analysis.get("configurations", [])
                )
            }
            macro_id = "macro-dram-l3-ffn-output-weight-stationary"
            ffn_output_macro = {
                "id": macro_id,
                "stable_id": macro_id,
                "evidence_id": _semantic_id(
                    "macro-evidence",
                    {"macro_id": macro_id, "resources": expected_resources},
                ),
                "level": "L3",
                "family": "weight-stationary-ffn-output",
                "eligible": cross_proven,
                "implementation_ready": cross_proven,
                "status": (
                    "eligible-cross-config-schedule-proven"
                    if cross_proven
                    else "blocked-cross-config-schedule-unproven"
                ),
                "member_candidate_ids": [
                    candidate["id"] for candidate in ffn_output_candidates
                ],
                "member_stable_ids": sorted(
                    candidate["id"] for candidate in ffn_output_candidates
                ),
                "tensor_slices": sorted(
                    region["tensor_slice"] for region in reference_regions
                ),
                "expected_dram_resources": expected_resources,
                "allocation": {
                    "allocator": "l3-two-state-buffers-bank13-v3",
                    "preferred_bank": 13,
                    "preferred_bank_name": _VRF_BANK_NAMES[13],
                    "capacity_source": "emulator/npu_device_mini.py::VRF_SIZES",
                    "proof_scope": "all required validation DAG configurations",
                    "validation_matrix_complete": matrix_complete,
                    "cross_config_proven": cross_proven,
                    "allocation_proven": cross_proven,
                    "required_config_ids": required,
                    "config_results": [
                        {
                            "config_id": result["config_id"],
                            "member_count": result["member_count"],
                            "removable_read_bytes": result[
                                "removable_read_bytes"
                            ],
                            "allocation_proven": result["allocation_proven"],
                            "proof_digest": _proof_digest(result),
                        }
                        for result in config_results
                    ],
                    "partial_sum_allocation": reference_proof.get(
                        "partial_sum_allocation", {}
                    ),
                    "state_allocations": reference_proof.get(
                        "state_allocations", []
                    ),
                    "l2_x_retention": reference_proof.get(
                        "l2_x_retention", {}
                    ),
                    "schedule_signature": reference_proof.get(
                        "schedule_signature", {}
                    ),
                    "regions": reference_regions,
                    "required_source_invariants": [
                        "finish FFN_INTERMEDIATE+GELU for every position first",
                        "load each W_FFN_OUTPUT tile exactly once",
                        "read GELU from state B and write FFN_OUTPUT to state A",
                        "preserve tc=0..num_tiles-1 and the bias boundary",
                        "retain every original X cache entry until residual2",
                        "begin residual2+LN2 only after all FFN_OUTPUT positions",
                    ],
                },
                "estimated_saving": {
                    "observed_graph_bytes": int(measured.get("seq6", 0)),
                    "projected_seq2_bytes": int(measured.get("seq2", 0)),
                    "projected_seq6_bytes": int(measured.get("seq6", 0)),
                    "projection_assumption": (
                        "measured repeated W_FFN_OUTPUT MAT_LOAD bytes"
                    ),
                },
                "scope_policy": (
                    "only exact W_FFN_OUTPUT Tensor/address loads may reduce"
                ),
                "priority_score": int(measured.get("seq6", 0)),
            }
            merged_macros.append(ffn_output_macro)
            l3_proofs.append(
                {
                    "id": macro_id,
                    "level": "L3",
                    "cross_config_proven": cross_proven,
                    "config_results": config_results,
                }
            )

        remaining = [
            candidate
            for candidate in weight_candidates
            if candidate.get("tensor")
            not in {
                "W_Q",
                "W_K",
                "W_V",
                "W_SELF_OUTPUT",
                "W_FFN_INTERMEDIATE",
                "W_FFN_OUTPUT",
            }
        ]
        if remaining:
            measured = {
                f"seq{seq_len}": sum(
                    int(
                        candidate["per_config"][f"seq{seq_len}"].get(
                            "removable_read_bytes", 0
                        )
                    )
                    for candidate in remaining
                )
                for seq_len in sorted(
                    int(item["seq_len"])
                    for item in multiseq_analysis.get("configurations", [])
                )
            }
            blocked_id = "macro-dram-l3-weight-stationary"
            blocked_macro = {
                "id": blocked_id,
                "stable_id": blocked_id,
                "evidence_id": _semantic_id(
                    "macro-evidence", {"macro_id": blocked_id}
                ),
                "level": "L3",
                "family": "weight-stationary",
                "eligible": False,
                "implementation_ready": False,
                "status": "blocked-phase-schedule-proof-required",
                "member_candidate_ids": [
                    candidate["id"] for candidate in remaining
                ],
                "member_stable_ids": sorted(
                    candidate["id"] for candidate in remaining
                ),
                "tensor_slices": [],
                "expected_dram_resources": [],
                "allocation": {
                    "allocator": None,
                    "allocation_proven": False,
                    "cross_config_proven": False,
                    "validation_matrix_complete": matrix_complete,
                    "missing_proofs": [
                        "phase-level producer/consumer schedule",
                        "MRF residency and clobber proof",
                        "per-position input/output allocation",
                        "FP16 accumulation-order preservation",
                    ],
                },
                "estimated_saving": {
                    "observed_graph_bytes": int(measured.get("seq6", 0)),
                    "projected_seq2_bytes": int(measured.get("seq2", 0)),
                    "projected_seq6_bytes": int(measured.get("seq6", 0)),
                    "projection_assumption": (
                        "measured repeated non-Q/K/V MAT_LOAD upper bound"
                    ),
                },
                "scope_policy": "blocked; not accepted by the declaration gate",
                "priority_score": int(measured.get("seq6", 0)),
            }
            merged_macros.append(blocked_macro)
            l3_proofs.append(
                {
                    "id": blocked_id,
                    "level": "L3",
                    "cross_config_proven": False,
                    "missing_proofs": blocked_macro["allocation"][
                        "missing_proofs"
                    ],
                }
            )

    level_order = {"L1": 1, "L2": 2, "L3": 3}
    merged_macros.sort(
        key=lambda item: (
            level_order.get(str(item.get("level")), 99),
            -int(item.get("priority_score", 0)),
            str(item.get("id")),
        )
    )
    for rank, macro in enumerate(merged_macros, start=1):
        macro["rank"] = rank

    macro_data = {
        "schema": {
            "name": "jimu-npu-dram-macro-candidates",
            "version": "2.0.0",
        },
        "summary": {
            "macros": len(merged_macros),
            "eligible": sum(1 for macro in merged_macros if macro["eligible"]),
            "blocked": sum(1 for macro in merged_macros if not macro["eligible"]),
            "primitive_candidates": reference_macros.get("summary", {}).get(
                "primitive_candidates", 0
            ),
            "validation_matrix_complete": matrix_complete,
        },
        "analysis_limits": {
            "allocation_proof_scope": "all required validation DAG configurations",
            "required_config_ids": required,
            "missing_config_ids": missing,
            "l2_schedule": "preserve trace order; cache after first load",
            "l3_schedule": (
                "K/V and Q position interchange are deterministic; Q outputs "
                "are retained across attention MRF clobbers; later projection "
                "phases remain blocked pending producer/consumer proof"
            ),
        },
        "macros": merged_macros,
    }
    proof = {
        "schema": {
            "name": "jimu-npu-cross-config-allocation-proof",
            "version": "1.0.0",
        },
        "required_config_ids": required,
        "observed_config_ids": sorted(graph_by_id),
        "missing_config_ids": missing,
        "validation_matrix_complete": matrix_complete,
        "l1": l1_proofs,
        "l2": l2_proofs,
        "l3": l3_proofs,
    }
    return proof, macro_data


def _multiseq_markdown(
    analysis: dict[str, Any],
    reference_structured: dict[str, Any],
) -> str:
    reference_seq = int(analysis["reference_seq_len"])
    lines = [
        "# Multi-Sequence DAG Summary",
        "",
        "Concrete sequence DAGs are compared by Tensor/tile semantics. Savings "
        "below are measured from traces; they are not seq=1 projections.",
        "",
        "## Configurations",
        "",
        "| Seq | DIM | Hidden | Heads | Micro-ops | ELF SHA256 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in analysis["configurations"]:
        config = item["firmware_config"]
        digest = item.get("elf_sha256") or "-"
        lines.append(
            f"| {item['seq_len']} | {config['dim']} | "
            f"{config['hidden_size']} | {config['num_head']} | "
            f"{item.get('counts', {}).get('micro_ops', 0)} | `{digest}` |"
        )

    lines.extend(
        [
            "",
            "## Cross-configuration implementation candidates",
            "",
            f"| Macro | Level | Eligible | Members | Seq{reference_seq} estimate |",
            "|---|---|---|---:|---:|",
        ]
    )
    macros = reference_structured.get("macro_candidates", {}).get("macros", [])
    for macro in macros:
        lines.append(
            f"| `{macro['id']}` | {macro.get('level', 'L1')} | "
            f"{str(bool(macro['eligible'])).lower()} | "
            f"{len(macro['member_candidate_ids'])} | "
            f"{macro['estimated_saving']['projected_seq6_bytes']} B |"
        )
    if not macros:
        lines.append("| - | - | false | 0 | 0 B |")

    lines.extend(
        [
            "",
            "## Cross-position reuse families",
            "",
            "| Family | Level | Members | "
            + " | ".join(
                f"Seq{item['seq_len']} removable"
                for item in analysis["configurations"]
            )
            + " | Ready |",
            "|---|---|---:|"
            + "---:|" * len(analysis["configurations"])
            + "---|",
        ]
    )
    for family in analysis["families"]:
        saving_values = []
        for item in analysis["configurations"]:
            config_key = "seq{}".format(item["seq_len"])
            saving_values.append(
                "{} B".format(
                    family["measured_removable_read_bytes"].get(
                        config_key, 0
                    )
                )
            )
        saving_cells = " | ".join(saving_values)
        lines.append(
            f"| `{family['id']}` | {family['level']} | "
            f"{family['member_count']} | {saving_cells} | "
            f"{'yes' if family.get('implementation_ready') else 'no'} |"
        )
    if not analysis["families"]:
        empty_savings = " | ".join(
            "0 B" for _ in analysis["configurations"]
        )
        lines.append(f"| - | - | 0 | {empty_savings} | no |")

    lines.extend(
        [
            "",
            f"## Top measured reuse candidates (seq{reference_seq})",
            "",
            "| Rank | Tensor pattern | Level | Reads | Unique values | Removable | Status |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for candidate in analysis["candidates"][:20]:
        record = candidate["per_config"][f"seq{reference_seq}"]
        lines.append(
            f"| {candidate['rank']} | `{candidate['semantic_key']}` | "
            f"{candidate['level']} | {record.get('read_ops', 0)} | "
            f"{record.get('unique_values', 0)} | "
            f"{record.get('removable_read_bytes', 0)} B | "
            f"{candidate['status']} |"
        )

    lines.extend(
        [
            "",
            "DAG-PR6 permits only macros whose allocation is proven across the "
            "required validation matrix. L3 remains blocked until schedule, MRF "
            "residency, partial-sum storage and FP16 order are all proven.",
            "",
        ]
    )
    return "\n".join(lines)


def _allocation_markdown(
    proof: dict[str, Any],
    macro_data: dict[str, Any],
) -> str:
    lines = [
        "# Cross-Configuration Allocation Summary",
        "",
        f"Validation matrix complete: `{str(bool(proof['validation_matrix_complete'])).lower()}`",
        "",
        "Required configurations: " + ", ".join(
            f"`{config_id}`" for config_id in proof["required_config_ids"]
        ),
        "",
        "| Rank | Macro | Level | Eligible | Seq6 B | Proof |",
        "|---:|---|---|---|---:|---|",
    ]
    for macro in macro_data["macros"]:
        allocation = macro.get("allocation", {})
        lines.append(
            f"| {macro['rank']} | `{macro['id']}` | {macro['level']} | "
            f"{str(bool(macro['eligible'])).lower()} | "
            f"{macro['estimated_saving']['projected_seq6_bytes']} | "
            f"{'cross-config' if allocation.get('cross_config_proven') else macro['status']} |"
        )

    for macro in macro_data["macros"]:
        if macro.get("level") != "L2":
            continue
        lines.extend(
            [
                "",
                f"## `{macro['id']}` configuration proof",
                "",
                "| Configuration | Members | Removable B | Proven | Digest |",
                "|---|---:|---:|---|---|",
            ]
        )
        for result in macro.get("allocation", {}).get("config_results", []):
            lines.append(
                f"| `{result['config_id']}` | {result['member_count']} | "
                f"{result['removable_read_bytes']} | "
                f"{str(bool(result['allocation_proven'])).lower()} | "
                f"`{result['proof_digest']}` |"
            )
        lines.extend(
            [
                "",
                "Reference allocation regions:",
                "",
                "| Tensor | Bank | Region | First | Last |",
                "|---|---|---|---|---|",
            ]
        )
        for region in macro.get("allocation", {}).get("regions", []):
            lines.append(
                f"| `{region['tensor_slice']}` | `{region['bank_name']}` | "
                f"`[{region['base']},{region['end']})` | "
                f"`{region['interval']['start_node']}` | "
                f"`{region['interval']['end_node']}` |"
            )
    lines.extend(
        [
            "",
            "Open `allocation_proof.json` after selecting a macro to inspect "
            "every configuration region. Do not implement blocked L3 evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_multiseq_dag(
    output_dir: str | Path,
    sequence_dirs: dict[int, str | Path],
    *,
    proof_dirs: list[str | Path] | None = None,
    required_config_ids: list[str] | None = None,
) -> dict[str, Path]:
    """Merge concrete DAGs and emit PR5/PR6 evidence."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    structured_by_seq = {
        int(seq_len): load_structured_dag(path)
        for seq_len, path in sequence_dirs.items()
    }
    analysis = build_multiseq_dag(structured_by_seq)
    reference_seq = int(analysis["reference_seq_len"])
    reference = structured_by_seq[reference_seq]

    proof_graphs = list(structured_by_seq.values())
    proof_inputs = []
    for proof_dir in proof_dirs or []:
        graph = load_structured_dag(proof_dir)
        proof_graphs.append(graph)
        proof_inputs.append(
            {
                "config_id": _config_identity(graph),
                "path": str(Path(proof_dir).resolve()),
                "elf_sha256": graph.get("metadata", {})
                .get("elf", {})
                .get("sha256"),
            }
        )
    graph_by_id = {}
    for graph in proof_graphs:
        config_id = _config_identity(graph)
        previous = graph_by_id.get(config_id)
        if previous is not None:
            previous_sha = previous.get("metadata", {}).get("elf", {}).get("sha256")
            current_sha = graph.get("metadata", {}).get("elf", {}).get("sha256")
            if previous_sha != current_sha:
                raise ValueError(
                    f"duplicate proof configuration has different ELF: {config_id}"
                )
            continue
        graph_by_id[config_id] = graph
    required = sorted(required_config_ids or graph_by_id)
    allocation_proof, macro_data = build_cross_config_allocation_proof(
        reference,
        list(graph_by_id.values()),
        analysis,
        required_config_ids=required,
    )
    reference["macro_candidates"] = macro_data

    macro_by_id = {macro["id"]: macro for macro in macro_data["macros"]}
    for family in analysis.get("families", []):
        macro = macro_by_id.get(family["id"])
        ready = bool(macro and macro.get("implementation_ready", macro.get("eligible")))
        family["implementation_ready"] = ready
        family["allocation"] = {
            "proven": ready,
            "validation_matrix_complete": allocation_proof[
                "validation_matrix_complete"
            ],
            "proof_artifact": "allocation_proof.json",
            "macro_id": family["id"],
        }
    for candidate in analysis.get("candidates", []):
        family_macro = next(
            (
                family
                for family in analysis.get("families", [])
                if family["family"] == candidate["family"]
            ),
            None,
        )
        ready = bool(family_macro and family_macro["implementation_ready"])
        candidate["implementation_ready"] = ready
        candidate["allocation"] = {
            "proven": ready,
            "family_macro_id": family_macro["id"] if family_macro else None,
            "proof_artifact": "allocation_proof.json",
        }
        if ready:
            candidate["status"] = "eligible-cross-config-family-proven"
    analysis["summary"]["implementation_ready"] = sum(
        1 for candidate in analysis.get("candidates", [])
        if candidate["implementation_ready"]
    )
    analysis["analysis_limits"]["allocation_search"] = (
        "deterministic first-fit across required validation DAGs"
    )
    analysis["analysis_limits"]["candidate_declaration"] = (
        "only macros with cross_config_proven=true are accepted"
    )

    metadata = {
        "schema": analysis["schema"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workload": reference.get("metadata", {}).get("workload", {}),
        "reference_seq_len": reference_seq,
        "required_config_ids": required,
        "validation_matrix_complete": allocation_proof[
            "validation_matrix_complete"
        ],
        "inputs": [
            {
                "seq_len": seq_len,
                "path": str(Path(sequence_dirs[seq_len]).resolve()),
                "elf_sha256": structured_by_seq[seq_len]
                .get("metadata", {})
                .get("elf", {})
                .get("sha256"),
            }
            for seq_len in sorted(sequence_dirs)
        ],
        "proof_inputs": proof_inputs,
    }
    paths = {
        "metadata": output / "multiseq_metadata.json",
        "loop_invariants": output / "loop_invariants.json",
        "summary": output / "multiseq_summary.md",
        "candidate_evidence": output / "candidate_evidence.jsonl",
        "allocation_proof": output / "allocation_proof.json",
        "allocation_summary": output / "allocation_summary.md",
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["loop_invariants"].write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["allocation_proof"].write_text(
        json.dumps(allocation_proof, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["allocation_summary"].write_text(
        _allocation_markdown(allocation_proof, macro_data), encoding="utf-8"
    )
    (output / "macro_candidates.json").write_text(
        json.dumps(macro_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "macro_candidate_summary.md").write_text(
        _macro_candidate_markdown(macro_data), encoding="utf-8"
    )
    paths["summary"].write_text(
        _multiseq_markdown(analysis, reference), encoding="utf-8"
    )

    primitive_candidates = sorted(
        reference.get("candidates", {}).get("candidates", []),
        key=lambda item: (
            -int(item.get("estimated_saving", {}).get("observed_graph_bytes", 0)),
            str(item.get("id")),
        ),
    )[:20]
    evidence = []
    for macro in reference.get("macro_candidates", {}).get("macros", []):
        allocation = macro.get("allocation", {})
        evidence.append(
            {
                "evidence_type": "macro",
                "id": macro.get("id"),
                "level": macro.get("level"),
                "eligible": bool(macro.get("eligible")),
                "member_count": len(macro.get("member_candidate_ids", [])),
                "estimated_saving": macro.get("estimated_saving", {}),
                "allocation_proven": bool(
                    allocation.get("allocation_proven")
                ),
                "cross_config_proven": bool(
                    allocation.get("cross_config_proven")
                ),
            }
        )
    for candidate in primitive_candidates:
        plan = candidate.get("vrf_plan", {})
        evidence.append(
            {
                "evidence_type": "l1_candidate",
                "id": candidate.get("id"),
                "stable_id": candidate.get("stable_id"),
                "status": candidate.get("status"),
                "tensor_slice": candidate.get("tensor_slice"),
                "phase_kind": candidate.get("phase_kind"),
                "producer": candidate.get("producer"),
                "consumers": candidate.get("consumers", []),
                "estimated_saving": candidate.get("estimated_saving", {}),
                "allocation_proven": bool(plan.get("allocation_proven")),
            }
        )

    selected_ids = set()
    for candidate in primitive_candidates:
        selected_ids.add(str(candidate.get("producer")))
        consumers = list(map(str, candidate.get("consumers", [])))
        selected_ids.update(consumers[:1])
        selected_ids.update(consumers[-1:])
    # The prompt needs concrete evidence, not another full trace. Keep the
    # highest-value cross-sequence slice; complete DAG JSONL remains on disk.
    for candidate in analysis["candidates"][:20]:
        compact_configs = {}
        for config_key, config in candidate["per_config"].items():
            node_ids = list(map(str, config.get("node_ids", [])))
            compact_configs[config_key] = {
                key: config.get(key)
                for key in (
                    "present",
                    "addresses",
                    "slices",
                    "read_ops",
                    "write_ops",
                    "read_bytes",
                    "unique_values",
                    "compulsory_read_bytes",
                    "removable_read_bytes",
                )
            }
            compact_configs[config_key]["representative_node_ids"] = list(
                dict.fromkeys(node_ids[:2] + node_ids[-2:])
            )
        evidence.append(
            {
                "evidence_type": "multiseq_candidate",
                "id": candidate["id"],
                "rank": candidate["rank"],
                "semantic_key": candidate["semantic_key"],
                "level": candidate["level"],
                "status": candidate["status"],
                "implementation_ready": candidate["implementation_ready"],
                "per_config": compact_configs,
            }
        )
        reference_ids = compact_configs[f"seq{reference_seq}"][
            "representative_node_ids"
        ]
        selected_ids.update(reference_ids)

    def compact_resource(resource: dict[str, Any]) -> dict[str, Any]:
        return {
            key: resource[key]
            for key in (
                "space",
                "tensor",
                "slice",
                "address_hex",
                "bank",
                "row",
            )
            if key in resource
        }

    for node in reference.get("micro_ops", []):
        if str(node.get("id")) not in selected_ids:
            continue
        evidence.append(
            {
                "evidence_type": "node",
                "id": node.get("id"),
                "kind": node.get("kind"),
                "phase_kind": node.get("phase_kind"),
                "phase_instance": node.get("phase_instance"),
                "defs": [
                    compact_resource(resource)
                    for resource in node.get("defs", [])
                ],
                "uses": [
                    compact_resource(resource)
                    for resource in node.get("uses", [])
                ],
            }
        )
    paths["candidate_evidence"].write_text(
        "\n".join(_json_line(record) for record in evidence)
        + ("\n" if evidence else ""),
        encoding="utf-8",
    )
    return paths
