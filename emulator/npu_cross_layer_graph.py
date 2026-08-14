"""Unified tensor/command/dependency/timing evidence graph.

Renderers are secondary.  The primary artifact is deterministic JSON that an
optimisation agent can query without guessing semantics from an image.  Every
dynamic command keeps its source provenance and links to declared tensor
regions, def-use predecessors, chain barriers, and optional timing records.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from emulator.workload import TensorRegion, WorkloadManifest


@dataclass
class GraphNode:
    id: str
    layer: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "layer": self.layer, "label": self.label,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    label: str = ""
    attributes: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "target": self.target,
            "kind": self.kind, "label": self.label,
            "attributes": dict(self.attributes),
        }


@dataclass
class CrossLayerGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    opportunities: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "opportunities": self.opportunities,
        }

    def write_json(self, path: str | Path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def to_text(self, max_commands: int = 80,
                max_opportunities: int = 30) -> str:
        commands = [node for node in self.nodes if node.layer == "command"]
        tensors = [node for node in self.nodes if node.layer == "tensor"]
        lines = [
            "Cross-layer firmware evidence graph",
            f"  tensors={len(tensors)} commands={len(commands)} edges={len(self.edges)}",
            "",
            "Tensor regions:",
        ]
        if tensors:
            for node in tensors:
                attrs = node.attributes
                lines.append(
                    f"  {node.label}: [{attrs.get('address')}, {attrs.get('end_address')}) "
                    f"shape={attrs.get('shape') or '?'} role={attrs.get('role')} "
                    f"observable={attrs.get('observable')} frozen={attrs.get('frozen')}"
                )
        else:
            lines.append("  (no manifest tensor metadata; anonymous buffers only)")
        lines.extend(["", "Dynamic commands:"])
        for node in commands[:max_commands]:
            attrs = node.attributes
            source = attrs.get("source") or {}
            where = source.get("file") or "?"
            if source.get("line"):
                where += f":{source['line']}"
            timing = attrs.get("timing") or {}
            cycle_text = (
                f" cycles=[{timing.get('start_cycle')},{timing.get('finish_cycle')})"
                if timing else ""
            )
            lines.append(
                f"  {node.id} chain={attrs.get('chain_id')} {node.label} "
                f"unit={attrs.get('target_unit')} src={where}{cycle_text}"
            )
        if len(commands) > max_commands:
            lines.append(f"  ... {len(commands) - max_commands} commands omitted")
        lines.extend(["", "Evidence-backed opportunities:"])
        for item in self.opportunities[:max_opportunities]:
            lines.append(
                f"  [{item['kind']}] priority={item.get('priority', 'normal')} "
                f"events={item.get('events', [])}: {item['summary']}"
            )
        if not self.opportunities:
            lines.append("  (none detected)")
        return "\n".join(lines)

    def to_dot(self) -> str:
        colors = {"tensor": "lightblue", "command": "lightgoldenrod1"}
        lines = [
            "digraph cross_layer_graph {", "  rankdir=LR;", "  compound=true;",
            '  node [shape=box style="rounded,filled" fontname="Helvetica"];',
        ]
        for node in self.nodes:
            label = _dot_escape(node.label)
            color = colors.get(node.layer, "white")
            lines.append(
                f'  "{node.id}" [label="{label}" fillcolor="{color}"];'
            )
        edge_colors = {
            "tensor_read": "blue", "tensor_write": "darkgreen",
            "RAW": "black", "WAR": "orange", "WAW": "red",
            "chain_barrier": "purple",
        }
        for edge in self.edges:
            color = edge_colors.get(edge.kind, "gray")
            label = _dot_escape(edge.label or edge.kind)
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" '
                f'[label="{label}" color="{color}"];'
            )
        lines.append("}")
        return "\n".join(lines)


def build_cross_layer_graph(
    events: list[dict[str, Any]],
    manifest: WorkloadManifest | None = None,
    schedule: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    profile_name: str | None = None,
) -> CrossLayerGraph:
    timing_by_command = _timing_index(schedule)
    raw_command_schedule = any(
        "sequence" in record and "enqueue_cycle" in record
        for record in timing_by_command.values()
    )
    if raw_command_schedule:
        raw_timing = timing_by_command
        timing_by_command = {}
        for position, event in enumerate(events):
            command_id = int(event.get("command_id", event.get("idx", position)))
            raw_id = int(event.get("raw_instruction_idx", command_id))
            if raw_id in raw_timing:
                timing_by_command[command_id] = {
                    **raw_timing[raw_id], "_raw_command_timing": True,
                }
    nodes: list[GraphNode] = []
    edges: set[GraphEdge] = set()
    manifest_regions = manifest.tensors if manifest else []
    known_tensor_nodes: set[str] = set()
    for region in manifest_regions:
        node_id = f"tensor:{region.name}"
        known_tensor_nodes.add(node_id)
        attrs = region.to_dict()
        attrs["end_address"] = region.end_address
        nodes.append(GraphNode(node_id, "tensor", region.name, attrs))

    last_writer: dict[tuple, int] = {}
    readers: dict[tuple, set[int]] = defaultdict(set)
    last_command_by_chain: dict[int, int] = {}
    command_nodes: dict[int, GraphNode] = {}
    anonymous_regions: dict[tuple[int, int], str] = {}

    for position, event in enumerate(events):
        command_id = int(event.get("command_id", event.get("idx", position)))
        node_id = f"command:{command_id}"
        timing = timing_by_command.get(command_id)
        attrs = {
            key: event.get(key) for key in (
                "opcode", "raw", "raw_instruction_idx", "expanded_idx",
                "chain_id", "target_unit", "source", "cpu_cycle", "memory",
            )
        }
        attrs["timing"] = (
            {key: value for key, value in timing.items()
             if not str(key).startswith("_")}
            if timing else None
        )
        attrs["critical"] = bool(
            timing and (timing.get("critical_path") or timing.get("critical"))
        )
        node = GraphNode(node_id, "command", str(event.get("op", "?")), attrs)
        nodes.append(node)
        command_nodes[command_id] = node

        for resource in event.get("uses", []):
            key = _resource_key(resource)
            if key in last_writer:
                edges.add(GraphEdge(
                    f"command:{last_writer[key]}", node_id, "RAW",
                    _resource_label(key),
                ))
            readers[key].add(command_id)
        for resource in event.get("defs", []):
            key = _resource_key(resource)
            if key in last_writer:
                edges.add(GraphEdge(
                    f"command:{last_writer[key]}", node_id, "WAW",
                    _resource_label(key),
                ))
            for reader in readers[key] - {command_id}:
                edges.add(GraphEdge(
                    f"command:{reader}", node_id, "WAR", _resource_label(key),
                ))
            readers[key].clear()
            last_writer[key] = command_id

        tensor_reads = list(event.get("tensor_reads") or [])
        tensor_writes = list(event.get("tensor_writes") or [])
        memory = event.get("memory")
        if memory and not tensor_reads and not tensor_writes:
            address = int(memory["address"])
            end = int(memory["end_address"])
            key = (address, end)
            name = anonymous_regions.setdefault(
                key, f"buffer_0x{address:x}_0x{end:x}"
            )
            tensor_id = f"tensor:{name}"
            if tensor_id not in known_tensor_nodes:
                known_tensor_nodes.add(tensor_id)
                nodes.append(GraphNode(tensor_id, "tensor", name, {
                    "address": address, "end_address": end,
                    "shape": [end - address], "dtype": "unknown",
                    "role": "anonymous", "observable": False,
                    "frozen": False, "inferred": True,
                }))
            if memory.get("direction") == "read":
                tensor_reads.append(name)
            else:
                tensor_writes.append(name)
        for name in tensor_reads:
            edges.add(GraphEdge(f"tensor:{name}", node_id, "tensor_read", name))
        for name in tensor_writes:
            edges.add(GraphEdge(node_id, f"tensor:{name}", "tensor_write", name))

        chain_id = int(event.get("chain_id", 0))
        previous = last_command_by_chain.get(chain_id)
        if previous is not None and event.get("op") == "INST_ISSUE":
            edges.add(GraphEdge(
                f"command:{previous}", node_id, "chain_barrier", f"chain {chain_id}",
            ))
        last_command_by_chain[chain_id] = command_id

    opportunities = _derive_opportunities(events, manifest_regions,
                                           timing_by_command)
    return CrossLayerGraph(
        nodes=nodes,
        edges=sorted(edges, key=lambda edge: (
            edge.source, edge.target, edge.kind, edge.label
        )),
        opportunities=opportunities,
        metadata={
            "workload": manifest.name if manifest else None,
            "profile": profile_name,
            "event_count": len(events),
            "has_source_provenance": any(
                (event.get("source") or {}).get("file") for event in events
            ),
            "has_timing": bool(timing_by_command),
            "semantic_tensor_count": len(manifest_regions),
        },
    )


def _derive_opportunities(events: list[dict[str, Any]],
                          regions: Iterable[TensorRegion],
                          timing: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    region_by_name = {region.name: region for region in regions}
    reads: dict[str, list[int]] = defaultdict(list)
    writes: dict[str, list[int]] = defaultdict(list)
    source_by_id: dict[int, dict[str, Any]] = {}
    event_by_id: dict[int, dict[str, Any]] = {}
    for position, event in enumerate(events):
        command_id = int(event.get("command_id", event.get("idx", position)))
        event_by_id[command_id] = event
        source_by_id[command_id] = event.get("source") or {}
        for name in event.get("tensor_reads") or []:
            reads[name].append(command_id)
        for name in event.get("tensor_writes") or []:
            writes[name].append(command_id)
    opportunities: list[dict[str, Any]] = []
    for name, event_ids in sorted(reads.items()):
        region = region_by_name.get(name)
        if len(event_ids) > 1 and region and region.frozen:
            opportunities.append({
                "kind": "repeated_frozen_load",
                "priority": "high" if len(event_ids) >= 4 else "normal",
                "tensor": name,
                "events": event_ids[:20],
                "sources": _unique_sources(source_by_id, event_ids),
                "summary": (
                    f"frozen tensor {name!r} is read {len(event_ids)} times; "
                    "consider hoisting, VRF/MRF residency, or prefetch"
                ),
            })
    for name, write_ids in sorted(writes.items()):
        region = region_by_name.get(name)
        read_ids = reads.get(name, [])
        if write_ids and read_ids and region and not region.observable:
            pairs = [(writer, reader) for writer in write_ids for reader in read_ids
                     if writer < reader]
            if pairs:
                selected = pairs[:20]
                ids = sorted({value for pair in selected for value in pair})
                opportunities.append({
                    "kind": "intermediate_materialization",
                    "priority": "high" if len(pairs) >= 2 else "normal",
                    "tensor": name,
                    "events": ids,
                    "sources": _unique_sources(source_by_id, ids),
                    "summary": (
                        f"non-observable tensor {name!r} is stored then reloaded "
                        f"across {len(pairs)} producer-consumer pair(s); test on-chip retention"
                    ),
                })
    for command_id, record in sorted(timing.items()):
        event = event_by_id.get(command_id)
        if (
            record.get("_raw_command_timing")
            and event is not None
            and int(event.get("expanded_idx", 0)) > 0
        ):
            continue
        wait = int(record.get("queue_wait_cycles", 0) or 0)
        enqueue = int(record.get("enqueue_cycle", 0) or 0)
        decoder_wait = max(
            0, int(record.get("decoder_ready_cycle", enqueue) or enqueue) - enqueue
        )
        actionable_wait = max(0, wait - decoder_wait)
        blockers = []
        for reason, field_name in (
            ("fifo_full", "fifo_ready_cycle"),
            ("scoreboard", "dependency_ready_cycle"),
            ("issue", "issue_ready_cycle"),
            ("unit", "unit_ready_cycle"),
            ("dram", "memory_ready_cycle"),
        ):
            if int(record.get(field_name, enqueue) or enqueue) > enqueue + decoder_wait:
                blockers.append(reason)
        if actionable_wait > 0:
            opportunities.append({
                "kind": "scheduled_wait",
                "priority": "high" if record.get("critical_path") else "normal",
                "events": [command_id],
                "sources": _unique_sources(source_by_id, [command_id]),
                "wait_cycles": actionable_wait,
                "blocking_reasons": blockers,
                "summary": (
                    f"command waits {actionable_wait} actionable cycle(s) for "
                    f"{','.join(blockers) or 'dependencies/resources'}; "
                    f"unit={record.get('target_unit') or record.get('resources')}"
                ),
            })
    return sorted(opportunities, key=lambda item: (
        item.get("priority") != "high", item["kind"], item.get("events", [])
    ))


def _timing_index(schedule) -> dict[int, dict[str, Any]]:
    if not schedule:
        return {}
    records = schedule.get("events", []) if isinstance(schedule, dict) else schedule
    result = {}
    for position, record in enumerate(records):
        command_id = record.get("command_id", record.get("idx", position))
        result[int(command_id)] = dict(record)
    return result


def _unique_sources(source_by_id: dict[int, dict[str, Any]],
                    event_ids: Iterable[int]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for event_id in event_ids:
        source = source_by_id.get(event_id) or {}
        key = (source.get("file"), source.get("line"), source.get("function"))
        if key == (None, None, None) or key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique[:10]


def _resource_key(resource: Any) -> tuple:
    if isinstance(resource, tuple):
        return resource
    if isinstance(resource, list):
        return tuple(resource)
    return (str(resource),)


def _resource_label(resource: tuple) -> str:
    return ":".join(str(value) for value in resource)


def _dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
