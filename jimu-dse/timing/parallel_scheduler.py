"""Deterministic dependency/resource scheduler for NPU instruction traces.

This module is deliberately separate from the functional emulator.  It consumes
the dynamic EventTracer stream, per-event latencies, and a versioned hardware
profile, then produces a cycle-like resource schedule.  The model is intended
for optimization ranking and auditability; it is not an RTL replacement.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class TimingScheduleError(RuntimeError):
    pass


SUPPORTED_RESOURCES = {"dram_bus", "vmm", "mmm", "mvu", "spu"}
VECTOR_DRAM_OPS = {"V_RD_DRAM", "V_WR_DRAM"}
MATRIX_DRAM_OPS = {"M_RD_DRAM", "M_WR_DRAM"}
MATRIX_MOVE_OPS = {"M_RD", "M_WR"}
SCALAR_OPS = {"S_WR", "S_RD", "S_RECIP", "S_SQRT", "SS_MUL", "SS_ADD"}


def resources_for_event(event: dict[str, Any]) -> list[str]:
    """Map one dynamic instruction to the units occupied for its duration."""
    op = str(event.get("op", ""))
    if op in VECTOR_DRAM_OPS:
        return ["dram_bus", "vmm"]
    if op in MATRIX_DRAM_OPS:
        return ["dram_bus", "mmm"]
    if op == "MV_MUL":
        return ["mvu"]
    if op in MATRIX_MOVE_OPS:
        return ["mmm"]
    if op in SCALAR_OPS or op.startswith("S_") or op.startswith("SS_"):
        return ["spu"]
    if op == "INST_ISSUE":
        return []
    return ["vmm"]


def _resource_key(value: Any) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (str(value),)


def _memory_interval(event: dict[str, Any]) -> tuple[int, int, str] | None:
    memory = event.get("memory")
    if not isinstance(memory, dict):
        return None
    try:
        start = int(memory["address"])
        end = int(memory["end_address"])
        direction = str(memory["direction"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start or direction not in {"read", "write"}:
        return None
    return start, end, direction


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def build_dependency_details(
    events: list[dict[str, Any]],
) -> dict[int, dict[int, set[str]]]:
    """Build dependency edges and retain the reason for every predecessor."""
    details: dict[int, dict[int, set[str]]] = {
        int(event["idx"]): {} for event in events
    }

    def add(current: int, predecessor: int, reason: str) -> None:
        if current != predecessor:
            details[current].setdefault(predecessor, set()).add(reason)

    last_writer: dict[tuple, int] = {}
    readers: dict[tuple, set[int]] = defaultdict(set)
    memory_accesses: list[tuple[int, int, str, int]] = []
    prior: list[int] = []
    latest_fence: int | None = None

    for event in events:
        idx = int(event["idx"])
        op = str(event.get("op", ""))
        if latest_fence is not None and idx != latest_fence:
            add(idx, latest_fence, "CONFIG_FENCE")

        if op == "S_WR":
            for predecessor in prior:
                add(idx, predecessor, "CONFIG_FENCE")
            latest_fence = idx

        uses = {
            _resource_key(resource)
            for resource in event.get("uses", [])
            if resource and _resource_key(resource)[0] != "DRAM"
        }
        defs = {
            _resource_key(resource)
            for resource in event.get("defs", [])
            if resource and _resource_key(resource)[0] != "DRAM"
        }

        for resource in uses:
            if resource in last_writer:
                add(idx, last_writer[resource], "RAW")
            readers[resource].add(idx)
        for resource in defs:
            if resource in last_writer:
                add(idx, last_writer[resource], "WAW")
            for predecessor in readers[resource] - {idx}:
                add(idx, predecessor, "WAR")
            readers[resource].clear()
            last_writer[resource] = idx

        access = _memory_interval(event)
        if access is not None:
            start, end, direction = access
            for old_start, old_end, old_direction, old_idx in memory_accesses:
                if not _overlaps((start, end), (old_start, old_end)):
                    continue
                if direction == "write" or old_direction == "write":
                    if old_direction == "write" and direction == "read":
                        reason = "DRAM_RAW"
                    elif old_direction == "read" and direction == "write":
                        reason = "DRAM_WAR"
                    else:
                        reason = "DRAM_WAW"
                    add(idx, old_idx, reason)
            memory_accesses.append((start, end, direction, idx))

        prior.append(idx)

    return details


def build_dependencies(events: list[dict[str, Any]]) -> dict[int, set[int]]:
    """Build RAW, WAR, WAW, DRAM-range, and configuration-fence edges."""
    return {
        idx: set(predecessors)
        for idx, predecessors in build_dependency_details(events).items()
    }


def _partition_chains(
    events: list[dict[str, Any]],
) -> tuple[list[tuple[list[dict[str, Any]], dict[str, Any] | None]], bool]:
    groups: list[tuple[list[dict[str, Any]], dict[str, Any] | None]] = []
    current: list[dict[str, Any]] = []
    has_marker = False
    for event in events:
        if event.get("op") == "INST_ISSUE":
            groups.append((current, event))
            current = []
            has_marker = True
        else:
            current.append(event)
    if current or not groups:
        groups.append((current, None))
    return groups, not has_marker


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start >= merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _intersection_length(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        total += max(0, end - start)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _max_concurrency(records: list[dict[str, Any]]) -> int:
    points: list[tuple[int, int]] = []
    for record in records:
        start = int(record["start_cycle"])
        end = int(record["end_cycle"])
        if end > start:
            points.extend(((start, 1), (end, -1)))
    active = maximum = 0
    for _cycle, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _mark_critical_path(records: list[dict[str, Any]], makespan: int) -> None:
    by_idx = {int(record["idx"]): record for record in records}
    stack = [
        int(record["idx"])
        for record in records
        if int(record["end_cycle"]) == makespan
    ]
    critical: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in critical or idx not in by_idx:
            continue
        critical.add(idx)
        record = by_idx[idx]
        start = int(record["start_cycle"])
        for predecessor in record.get("predecessors", []):
            pred = by_idx.get(int(predecessor))
            if pred is not None and int(pred["end_cycle"]) == start:
                stack.append(int(predecessor))
        dispatch_predecessor = record.get("dispatch_predecessor")
        if dispatch_predecessor is not None:
            pred = by_idx.get(int(dispatch_predecessor))
            if pred is not None and int(pred["start_cycle"]) + 1 == start:
                stack.append(int(dispatch_predecessor))
    for record in records:
        record["critical"] = int(record["idx"]) in critical


def _optimization_diagnostics(
    records: list[dict[str, Any]], utilization: dict[str, float], limit: int = 12,
) -> dict[str, Any]:
    """Create a bounded, agent-facing explanation without dropping raw events."""
    critical = [record for record in records if record.get("critical")]
    top_events = sorted(
        critical,
        key=lambda record: (
            -int(record["duration_cycles"]), int(record["start_cycle"]),
            int(record["idx"]),
        ),
    )[:limit]
    event_fields = (
        "idx", "raw_instruction_idx", "expanded_idx", "chain_id", "op",
        "start_cycle", "end_cycle", "duration_cycles", "resources",
        "blocking_reasons", "dependency_reasons",
    )
    wait_cycles: dict[str, int] = defaultdict(int)
    wait_events: dict[str, int] = defaultdict(int)
    blockers = []
    for record in critical:
        reasons = list(record.get("blocking_reasons", [])) or ["ready"]
        dominant = reasons[0]
        wait = int(record.get("queue_wait_cycles", 0))
        wait_cycles[dominant] += wait
        wait_events[dominant] += 1
        if wait:
            blockers.append({
                "idx": int(record["idx"]),
                "op": record["op"],
                "wait_cycles": wait,
                "blocking_reasons": reasons,
                "dependency_predecessors": record.get(
                    "dependency_predecessors", []
                ),
            })
    blockers.sort(key=lambda item: (-item["wait_cycles"], item["idx"]))
    bottlenecks = [
        {"resource": resource, "utilization": value}
        for resource, value in sorted(
            utilization.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "critical_path_event_count": len(critical),
        "critical_path_top_events": [
            {key: record.get(key) for key in event_fields} for record in top_events
        ],
        "critical_path_top_blockers": blockers[:limit],
        "critical_event_wait_cycles_by_reason": dict(
            sorted(wait_cycles.items(), key=lambda item: (-item[1], item[0]))
        ),
        "critical_event_counts_by_reason": dict(
            sorted(wait_events.items(), key=lambda item: (-item[1], item[0]))
        ),
        "resource_bottlenecks": bottlenecks,
        "source_mapping_note": (
            "raw_instruction_idx and expanded_idx identify dynamic trace events; "
            "direct C source-line mapping is not available"
        ),
    }


def schedule_trace(
    events: list[dict[str, Any]],
    durations: dict[int, int],
    profile: dict[str, Any],
    serial_cycles: int,
) -> dict[str, Any]:
    """Schedule a trace and return aggregate metrics plus an auditable timeline."""
    scheduler = profile.get("scheduler", {})
    issue_width = int(scheduler.get("issue_width", 1))
    queue_depth = int(scheduler.get("queue_depth", 2))
    commit_cycles = int(scheduler.get("chain_commit_cycles", 1))
    resource_counts = {
        name: int(count)
        for name, count in scheduler.get("resources", {}).items()
    }
    missing = SUPPORTED_RESOURCES - set(resource_counts)
    unknown = set(resource_counts) - SUPPORTED_RESOURCES
    if missing or unknown:
        raise TimingScheduleError(
            "scheduler resources mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if issue_width < 1 or queue_depth < 1 or commit_cycles < 0:
        raise TimingScheduleError("invalid scheduler issue/queue/commit configuration")
    if any(count < 1 for count in resource_counts.values()):
        raise TimingScheduleError("scheduler resource counts must be positive")

    groups, implicit_chain = _partition_chains(events)
    resource_available = {
        name: [0] * count for name, count in resource_counts.items()
    }
    resource_last: dict[tuple[str, int], int] = {}
    scheduled: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    group_start = 0
    barrier_idx: int | None = None
    last_dispatch_idx: int | None = None
    last_dispatch_cycle: int | None = None

    for group_events, marker in groups:
        dependency_details = build_dependency_details(group_events)
        dependencies = {
            idx: set(predecessors)
            for idx, predecessors in dependency_details.items()
        }
        operations = []
        for event in group_events:
            idx = int(event["idx"])
            duration = int(durations.get(idx, 0))
            if duration < 0:
                raise TimingScheduleError(f"negative duration for event {idx}")
            operations.append({
                "event": event,
                "idx": idx,
                "duration": duration,
                "resources": resources_for_event(event),
                "dependencies": set(dependencies.get(idx, set())),
                "dependency_reasons": dependency_details.get(idx, {}),
            })

        queue: list[dict[str, Any]] = []
        cursor = 0
        now = group_start

        def fill_queue() -> None:
            nonlocal cursor
            while len(queue) < queue_depth and cursor < len(operations):
                operation = operations[cursor]
                operation.setdefault("queue_entry_cycle", now)
                queue.append(operation)
                cursor += 1

        fill_queue()
        while queue or cursor < len(operations):
            fill_queue()
            issued = 0
            while issued < issue_width:
                chosen_pos = None
                chosen_slots: dict[str, int] = {}
                for pos, operation in enumerate(queue):
                    if any(
                        predecessor not in scheduled
                        or int(scheduled[predecessor]["end_cycle"]) > now
                        for predecessor in operation["dependencies"]
                    ):
                        continue
                    slots: dict[str, int] = {}
                    available = True
                    for resource in operation["resources"]:
                        candidates = [
                            (cycle, slot)
                            for slot, cycle in enumerate(resource_available[resource])
                            if cycle <= now
                        ]
                        if not candidates:
                            available = False
                            break
                        slots[resource] = min(candidates)[1]
                    if available:
                        chosen_pos = pos
                        chosen_slots = slots
                        break
                if chosen_pos is None:
                    break

                operation = queue.pop(chosen_pos)
                event = operation["event"]
                idx = operation["idx"]
                end = now + operation["duration"]
                dependency_predecessors = set(operation["dependencies"])
                dependency_reasons = {
                    str(predecessor): sorted(reasons)
                    for predecessor, reasons in operation[
                        "dependency_reasons"
                    ].items()
                }
                if barrier_idx is not None:
                    dependency_predecessors.add(barrier_idx)
                    dependency_reasons[str(barrier_idx)] = ["CHAIN_BARRIER"]
                resource_predecessors: set[int] = set()
                resource_blockers: dict[str, int] = {}
                for resource, slot in chosen_slots.items():
                    previous = resource_last.get((resource, slot))
                    if previous is not None:
                        resource_predecessors.add(previous)
                        resource_blockers[resource] = previous
                    resource_available[resource][slot] = end
                    resource_last[(resource, slot)] = idx
                predecessors = dependency_predecessors | resource_predecessors
                dispatch_predecessor = (
                    last_dispatch_idx
                    if last_dispatch_cycle is not None
                    and now == last_dispatch_cycle + 1
                    else None
                )
                blocking_reasons: list[str] = []
                for predecessor in sorted(dependency_predecessors):
                    prior_record = scheduled.get(predecessor)
                    if (
                        prior_record is not None
                        and int(prior_record["end_cycle"]) == now
                    ):
                        for reason in dependency_reasons.get(
                            str(predecessor), ["DEPENDENCY"]
                        ):
                            label = f"dependency:{reason}"
                            if label not in blocking_reasons:
                                blocking_reasons.append(label)
                for resource, predecessor in sorted(resource_blockers.items()):
                    prior_record = scheduled.get(predecessor)
                    if (
                        prior_record is not None
                        and int(prior_record["end_cycle"]) == now
                    ):
                        blocking_reasons.append(f"resource:{resource}")
                queue_entry_cycle = int(operation["queue_entry_cycle"])
                if queue_entry_cycle == now and now > group_start:
                    blocking_reasons.append("queue_window")
                if dispatch_predecessor is not None:
                    blocking_reasons.append("issue_width")
                record = {
                    "idx": idx,
                    "raw_instruction_idx": event.get("raw_instruction_idx"),
                    "expanded_idx": event.get("expanded_idx"),
                    "chain_id": event.get("chain_id", 0),
                    "op": event.get("op", ""),
                    "start_cycle": now,
                    "end_cycle": end,
                    "duration_cycles": operation["duration"],
                    "resources": operation["resources"],
                    "resource_slots": chosen_slots,
                    "dependency_predecessors": sorted(dependency_predecessors),
                    "dependency_reasons": dependency_reasons,
                    "resource_predecessors": sorted(resource_predecessors),
                    "predecessors": sorted(predecessors),
                    "dispatch_predecessor": dispatch_predecessor,
                    "queue_entry_cycle": queue_entry_cycle,
                    "queue_wait_cycles": max(0, now - queue_entry_cycle),
                    "blocking_reasons": blocking_reasons or ["ready"],
                    "memory": event.get("memory"),
                    "critical": False,
                }
                records.append(record)
                scheduled[idx] = record
                last_dispatch_idx = idx
                last_dispatch_cycle = now
                issued += 1
                fill_queue()

            if issued:
                now += 1
                continue

            future: list[int] = []
            for operation in queue:
                for predecessor in operation["dependencies"]:
                    if predecessor in scheduled:
                        end = int(scheduled[predecessor]["end_cycle"])
                        if end > now:
                            future.append(end)
                for resource in operation["resources"]:
                    future.extend(
                        cycle for cycle in resource_available[resource] if cycle > now
                    )
            if not future:
                waiting = [operation["idx"] for operation in queue]
                raise TimingScheduleError(
                    f"scheduler deadlock while waiting for events {waiting}"
                )
            now = min(future)

        group_records = [scheduled[int(event["idx"])] for event in group_events]
        group_end = max(
            [group_start, *(int(record["end_cycle"]) for record in group_records)]
        )
        if marker is not None:
            marker_idx = int(marker["idx"])
            end_predecessors = sorted(
                int(record["idx"])
                for record in group_records
                if int(record["end_cycle"]) == group_end
            )
            marker_end = group_end + commit_cycles
            marker_record = {
                "idx": marker_idx,
                "raw_instruction_idx": marker.get("raw_instruction_idx"),
                "expanded_idx": marker.get("expanded_idx"),
                "chain_id": marker.get("chain_id", 0),
                "op": "INST_ISSUE",
                "start_cycle": group_end,
                "end_cycle": marker_end,
                "duration_cycles": commit_cycles,
                "resources": ["spu"],
                "resource_slots": {"spu": 0},
                "dependency_predecessors": end_predecessors,
                "dependency_reasons": {
                    str(idx): ["CHAIN_COMMIT"] for idx in end_predecessors
                },
                "resource_predecessors": [],
                "predecessors": end_predecessors,
                "dispatch_predecessor": None,
                "queue_entry_cycle": group_end,
                "queue_wait_cycles": 0,
                "blocking_reasons": ["chain_commit"],
                "memory": None,
                "critical": False,
            }
            resource_available["spu"][0] = marker_end
            resource_last[("spu", 0)] = marker_idx
            records.append(marker_record)
            scheduled[marker_idx] = marker_record
            barrier_idx = marker_idx
            group_start = marker_end
        else:
            group_start = group_end

    makespan = max((int(record["end_cycle"]) for record in records), default=0)
    _mark_critical_path(records, makespan)
    records.sort(key=lambda item: (int(item["start_cycle"]), int(item["idx"])))

    busy_cycles = {name: 0 for name in SUPPORTED_RESOURCES}
    memory_intervals: list[tuple[int, int]] = []
    compute_intervals: list[tuple[int, int]] = []
    for record in records:
        duration = int(record["duration_cycles"])
        for resource in record["resources"]:
            busy_cycles[resource] += duration
        interval = (int(record["start_cycle"]), int(record["end_cycle"]))
        if "dram_bus" in record["resources"]:
            memory_intervals.append(interval)
        elif record["op"] != "INST_ISSUE":
            compute_intervals.append(interval)

    utilization = {
        name: (
            busy_cycles[name] / (makespan * resource_counts[name])
            if makespan else 0.0
        )
        for name in SUPPORTED_RESOURCES
    }
    metrics = {
        "parallel_predicted_npu_cycles": makespan,
        "overlap_saved_cycles": int(serial_cycles) - makespan,
        "memory_compute_overlap_cycles": _intersection_length(
            memory_intervals, compute_intervals
        ),
        "max_concurrent_ops": _max_concurrency(records),
        "dram_bus_utilization": utilization["dram_bus"],
        "mvu_utilization": utilization["mvu"],
        "vmm_utilization": utilization["vmm"],
        "mmm_utilization": utilization["mmm"],
        "spu_utilization": utilization["spu"],
        "schedule_chain_count": len(groups),
    }
    diagnostics = _optimization_diagnostics(records, utilization)
    return {
        "schema_version": 1,
        "model": profile.get("name", "scalesim-parallel"),
        "implicit_chain": implicit_chain,
        "scheduler": scheduler,
        "serial_cycles": int(serial_cycles),
        "metrics": metrics,
        "resource_busy_cycles": busy_cycles,
        "optimization_diagnostics": diagnostics,
        "events": records,
    }
