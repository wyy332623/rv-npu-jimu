#!/usr/bin/env python3
"""Generate presentation-ready RTL parallelism figures from closed-loop runs.

This is an intentionally standalone reporting tool.  It does not participate in
the optimisation loop and never builds firmware or mutates run artifacts.

By default it discovers the latest completed Verilator RTL runs, compares each
run's baseline schedule with its promoted best iteration, and writes 16:9 SVG
figures suitable for direct insertion into PowerPoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import datetime as dt
from html import escape
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "jimu-dse" / "results"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_out" / "rtl-parallelism"

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

UNIT_ORDER = ("load", "store", "mvu", "vector", "control")
UNIT_LABELS = {
    "load": "Load / DRAM Read",
    "store": "Store / DRAM Write",
    "mvu": "MVU",
    "vector": "Vector",
    "control": "Control",
}
UNIT_COLORS = {
    "load": "#2563EB",
    "store": "#EA580C",
    "mvu": "#7C3AED",
    "vector": "#059669",
    "control": "#64748B",
}

COLOR = {
    "background": "#FFFFFF",
    "foreground": "#0F172A",
    "muted": "#64748B",
    "grid": "#CBD5E1",
    "panel": "#F8FAFC",
    "panel_alt": "#F1F5F9",
    "critical": "#DC2626",
    "best": "#0F766E",
    "baseline": "#475569",
    "concurrency_1": "#CBD5E1",
    "concurrency_2": "#60A5FA",
    "concurrency_3": "#1D4ED8",
    "parallel_window": "#DBEAFE",
}


class VisualizationError(RuntimeError):
    """Raised for invalid or incomplete run artifacts."""


@dataclass(frozen=True)
class Schedule:
    path: Path
    events: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    makespan: int
    serial_cycles: int


@dataclass(frozen=True)
class RunSelection:
    run_dir: Path
    summary: dict[str, Any]
    baseline: Schedule
    best: Schedule
    best_iteration: int | None
    selection_kind: str = "best"

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def goal(self) -> str:
        return str(self.summary.get("goal") or "unknown-goal")

    @property
    def status(self) -> str:
        return str(self.summary.get("status") or "unknown")

    @property
    def improvement(self) -> float:
        if self.baseline.makespan <= 0:
            return 0.0
        return (
            (self.baseline.makespan - self.best.makespan)
            / self.baseline.makespan
        )

    @property
    def best_label(self) -> str:
        if self.selection_kind == "candidate":
            return f"Candidate（Iteration {self.best_iteration} · 未晋升）"
        if self.best_iteration is None:
            return "Best（无晋升候选）"
        return f"Best（Iteration {self.best_iteration}）"

    @property
    def comparison_name(self) -> str:
        return "Candidate" if self.selection_kind == "candidate" else "Best"


class SvgDocument:
    """Small deterministic SVG writer with presentation-oriented defaults."""

    def __init__(self, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img">',
            "<title>NPU RTL parallelism visualization</title>",
            "<desc>Aligned controller lanes show when NPU resources work in parallel.</desc>",
            "<style>",
            "text{font-family:'Microsoft YaHei','Noto Sans CJK SC','Arial',sans-serif;}",
            ".title{font-size:42px;font-weight:600;}",
            ".subtitle{font-size:22px;font-weight:400;}",
            ".metric{font-size:34px;font-weight:600;}",
            ".metric-label{font-size:18px;font-weight:400;}",
            ".panel-title{font-size:25px;font-weight:600;}",
            ".lane{font-size:18px;font-weight:500;}",
            ".event-label{font-size:13px;font-weight:600;}",
            ".event-label-secondary{font-size:11px;font-weight:400;}",
            ".axis{font-size:15px;font-weight:400;}",
            ".small{font-size:14px;font-weight:400;}",
            "</style>",
            f'<rect x="0" y="0" width="{width}" height="{height}" '
            f'fill="{COLOR["background"]}"/>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def text(
        self,
        x: float,
        y: float,
        value: object,
        *,
        css_class: str = "",
        fill: str | None = None,
        anchor: str = "start",
    ) -> None:
        attrs = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'text-anchor="{anchor}"']
        if css_class:
            attrs.append(f'class="{escape(css_class)}"')
        attrs.append(f'fill="{fill or COLOR["foreground"]}"')
        self.add(f'<text {" ".join(attrs)}>{escape(str(value))}</text>')

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        opacity: float = 1.0,
        rx: float = 0,
        stroke: str | None = None,
        stroke_width: float = 1,
    ) -> None:
        attrs = [
            f'x="{x:.2f}"', f'y="{y:.2f}"',
            f'width="{max(width, 0):.2f}"', f'height="{max(height, 0):.2f}"',
            f'fill="{fill}"', f'fill-opacity="{opacity:.3f}"', f'rx="{rx:.2f}"',
        ]
        if stroke:
            attrs.extend([f'stroke="{stroke}"', f'stroke-width="{stroke_width:.2f}"'])
        self.add(f'<rect {" ".join(attrs)}/>')

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        width: float = 1,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}"{dash_attr}/>'
        )

    def polygon(self, points: Sequence[tuple[float, float]], *, fill: str) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polygon points="{encoded}" fill="{fill}"/>')

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualizationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VisualizationError(f"JSON artifact must contain an object: {path}")
    return value


def load_schedule(path: Path) -> Schedule:
    data = _read_json(path)
    events = data.get("events")
    metrics = data.get("metrics")
    if not isinstance(events, list) or not isinstance(metrics, dict):
        raise VisualizationError(f"schedule lacks events or metrics: {path}")
    makespan = int(
        metrics.get("rtl_predicted_npu_cycles")
        or metrics.get("rtl_completion_makespan_cycles")
        or max(
            (int(item.get("finish_cycle", item.get("end_cycle", 0))) for item in events),
            default=0,
        )
    )
    serial_cycles = int(metrics.get("serial_command_cycles") or data.get("serial_cycles") or 0)
    if makespan <= 0:
        raise VisualizationError(f"schedule has no positive makespan: {path}")
    backend = data.get("backend")
    if backend != "verilator-rtl":
        raise VisualizationError(f"schedule is not a Verilator RTL schedule: {path}")
    return Schedule(
        path=path,
        events=tuple(item for item in events if isinstance(item, dict)),
        metrics=metrics,
        makespan=makespan,
        serial_cycles=serial_cycles,
    )


def _augment_schedule_metrics(
    schedule: Schedule,
    extra_metrics: object,
) -> Schedule:
    if not isinstance(extra_metrics, dict):
        return schedule
    return Schedule(
        path=schedule.path,
        events=schedule.events,
        metrics={**extra_metrics, **schedule.metrics},
        makespan=schedule.makespan,
        serial_cycles=schedule.serial_cycles,
    )


def load_run(
    run_dir: Path,
    *,
    candidate_iteration: int | None = None,
) -> RunSelection:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run-summary.json"
    baseline_path = run_dir / "baseline" / "timing-schedule.json"
    if not summary_path.is_file() or not baseline_path.is_file():
        raise VisualizationError(f"run lacks summary or baseline schedule: {run_dir}")
    summary = _read_json(summary_path)
    baseline = _augment_schedule_metrics(
        load_schedule(baseline_path),
        summary.get("baseline_metrics"),
    )

    if candidate_iteration is not None:
        if candidate_iteration < 1:
            raise VisualizationError("--candidate must be a positive iteration number")
        best_iteration = candidate_iteration
        best_path = run_dir / f"iteration-{best_iteration}" / "timing-schedule.json"
        if not best_path.is_file():
            raise VisualizationError(
                f"candidate iteration lacks timing schedule: {best_path}"
            )
        selection_kind = "candidate"
    else:
        raw_best_iteration = summary.get("best_iteration")
        best_iteration = (
            int(raw_best_iteration) if raw_best_iteration is not None else None
        )
        best_path = (
            run_dir / f"iteration-{best_iteration}" / "timing-schedule.json"
            if best_iteration is not None
            else baseline_path
        )
        if not best_path.is_file():
            best_path = baseline_path
            best_iteration = None
        selection_kind = "best"
    best = load_schedule(best_path)
    if selection_kind == "candidate":
        iteration_report_path = run_dir / f"iteration-{best_iteration}.json"
        if iteration_report_path.is_file():
            iteration_report = _read_json(iteration_report_path)
            probe = iteration_report.get("probe")
            probe_metrics = probe.get("metrics") if isinstance(probe, dict) else None
            best = _augment_schedule_metrics(
                best,
                iteration_report.get("metrics") or probe_metrics,
            )
    else:
        best = _augment_schedule_metrics(best, summary.get("best_metrics"))
    return RunSelection(
        run_dir=run_dir,
        summary=summary,
        baseline=baseline,
        best=best,
        best_iteration=best_iteration,
        selection_kind=selection_kind,
    )


def discover_recent_runs(
    results_root: Path,
    limit: int,
    *,
    include_incomplete: bool = False,
) -> list[RunSelection]:
    if limit < 1:
        raise VisualizationError("--latest must be positive")
    if not results_root.is_dir():
        raise VisualizationError(f"results root not found: {results_root}")
    candidates = sorted(
        (path for path in results_root.iterdir() if path.is_dir()),
        key=lambda path: (path / "run-summary.json").stat().st_mtime
        if (path / "run-summary.json").is_file()
        else path.stat().st_mtime,
        reverse=True,
    )
    selected: list[RunSelection] = []
    for candidate in candidates:
        try:
            run = load_run(candidate)
        except VisualizationError:
            continue
        if not include_incomplete and run.status != "completed":
            continue
        selected.append(run)
        if len(selected) >= limit:
            break
    if not selected:
        raise VisualizationError("no matching Verilator RTL closed-loop runs found")
    return selected


def resolve_requested_runs(
    paths: Sequence[str],
    results_root: Path,
    *,
    candidate_iteration: int | None = None,
) -> list[RunSelection]:
    runs: list[RunSelection] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            direct = (REPO_ROOT / path).resolve()
            path = direct if direct.is_dir() else (results_root / path).resolve()
        else:
            path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        runs.append(load_run(path, candidate_iteration=candidate_iteration))
    return runs


def _event_interval(event: dict[str, Any]) -> tuple[int, int] | None:
    try:
        start = int(event.get("start_cycle", event.get("start", 0)))
        finish = int(
            event.get("finish_cycle", event.get("end_cycle", event.get("finish", start)))
        )
    except (TypeError, ValueError):
        return None
    if finish <= start:
        return None
    return max(start, 0), finish


def intervals_by_unit(schedule: Schedule) -> dict[str, list[tuple[int, int, bool]]]:
    result = {unit: [] for unit in UNIT_ORDER}
    for event in schedule.events:
        unit = str(event.get("target_unit") or "")
        if unit not in result:
            continue
        interval = _event_interval(event)
        if interval is None:
            continue
        result[unit].append((interval[0], interval[1], bool(event.get("critical"))))
    return result


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, finish in ordered[1:]:
        prev_start, prev_finish = merged[-1]
        if start <= prev_finish:
            merged[-1] = (prev_start, max(prev_finish, finish))
        else:
            merged.append((start, finish))
    return merged


def utilization_by_unit(schedule: Schedule) -> dict[str, float]:
    units = intervals_by_unit(schedule)
    utilization: dict[str, float] = {}
    for unit, records in units.items():
        merged = _merge_intervals((start, finish) for start, finish, _ in records)
        busy = sum(finish - start for start, finish in merged)
        utilization[unit] = busy / schedule.makespan if schedule.makespan else 0.0
    return utilization


def _mark_bins(
    records: Iterable[tuple[int, int, bool]],
    domain_cycles: int,
    bins: int,
) -> tuple[bytearray, bytearray]:
    occupied = bytearray(bins)
    critical = bytearray(bins)
    if domain_cycles <= 0:
        return occupied, critical
    for start, finish, is_critical in records:
        lo = max(0, min(bins - 1, math.floor(start * bins / domain_cycles)))
        hi = max(lo + 1, min(bins, math.ceil(finish * bins / domain_cycles)))
        for idx in range(lo, hi):
            occupied[idx] = 1
            if is_critical:
                critical[idx] = 1
    return occupied, critical


def _runs(values: Sequence[int]) -> list[tuple[int, int, int]]:
    if not values:
        return []
    result: list[tuple[int, int, int]] = []
    start = 0
    current = int(values[0])
    for idx in range(1, len(values)):
        value = int(values[idx])
        if value == current:
            continue
        result.append((start, idx, current))
        start = idx
        current = value
    result.append((start, len(values), current))
    return result


def _format_cycles(value: int | float) -> str:
    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs(number) >= 1000:
        text = f"{number / 1000:.1f}k"
        return text.replace(".0k", "k")
    return str(int(round(number)))


def _nice_ticks(maximum: int, count: int = 5) -> list[int]:
    if maximum <= 0:
        return [0]
    raw = maximum / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    ticks = list(range(0, maximum + 1, int(step)))
    if not ticks or ticks[-1] != maximum:
        ticks.append(maximum)
    return ticks


def _draw_header(doc: SvgDocument, run: RunSelection, *, detail: str | None = None) -> None:
    title = "NPU RTL 并行工作时间轴"
    if detail:
        title = f"{title} · {detail}"
    doc.text(78, 72, title, css_class="title")
    doc.text(
        80,
        111,
        f"{run.goal}  ·  {run.name}  ·  status={run.status}",
        css_class="subtitle",
        fill=COLOR["muted"],
    )


def _draw_metric_strip(doc: SvgDocument, run: RunSelection, *, y: float = 152) -> None:
    improvement = run.improvement * 100
    speedup = run.baseline.makespan / run.best.makespan if run.best.makespan else 0
    if improvement >= 0:
        outcome_label = "Improvement"
        outcome_value = f"{improvement:.2f}%"
        outcome_context = f"{speedup:.2f}× speedup"
        outcome_color = COLOR["best"]
    else:
        extra_cycles = run.best.makespan - run.baseline.makespan
        outcome_label = "Regression"
        outcome_value = f"+{abs(improvement):.2f}% slower"
        outcome_context = f"+{extra_cycles:,} cycles"
        outcome_color = COLOR["critical"]
    metrics = [
        ("Baseline", f"{run.baseline.makespan:,}", "cycles", COLOR["baseline"]),
        (
            run.comparison_name,
            f"{run.best.makespan:,}",
            "cycles",
            outcome_color,
        ),
        (outcome_label, outcome_value, outcome_context, outcome_color),
    ]
    starts = (80, 555, 1030)
    for x, (label, value, context, color) in zip(starts, metrics):
        doc.text(x, y, label, css_class="metric-label", fill=COLOR["muted"])
        doc.text(x, y + 42, value, css_class="metric", fill=color)
        context_offset = 350 if label == "Regression" else 215
        doc.text(
            x + context_offset,
            y + 40,
            context,
            css_class="metric-label",
            fill=COLOR["muted"],
        )


def _draw_axis(
    doc: SvgDocument,
    *,
    plot_x: float,
    plot_y: float,
    plot_width: float,
    plot_height: float,
    domain_cycles: int,
    label_y: float,
) -> None:
    for tick in _nice_ticks(domain_cycles):
        x = plot_x + tick / domain_cycles * plot_width
        doc.line(x, plot_y, x, plot_y + plot_height, stroke=COLOR["grid"], width=1)
        anchor = "middle"
        if tick == 0:
            anchor = "start"
        elif tick == domain_cycles:
            anchor = "end"
        doc.text(
            x,
            label_y,
            _format_cycles(tick),
            css_class="axis",
            fill=COLOR["muted"],
            anchor=anchor,
        )
    doc.text(
        plot_x + plot_width / 2,
        label_y + 31,
        "RTL cycles",
        css_class="axis",
        fill=COLOR["muted"],
        anchor="middle",
    )


def _draw_schedule_panel(
    doc: SvgDocument,
    schedule: Schedule,
    *,
    label: str,
    y: float,
    domain_cycles: int,
    plot_x: float = 285,
    plot_width: float = 1520,
    lane_height: float = 29,
    lane_gap: float = 12,
    bins: int = 600,
    show_axis: bool = True,
) -> float:
    units = intervals_by_unit(schedule)
    utilization = utilization_by_unit(schedule)
    masks: dict[str, bytearray] = {}
    critical_masks: dict[str, bytearray] = {}
    critical_event_count = 0
    for unit in UNIT_ORDER:
        occupied, critical = _mark_bins(units[unit], domain_cycles, bins)
        masks[unit] = occupied
        critical_masks[unit] = critical
        critical_event_count += sum(1 for _, _, is_critical in units[unit] if is_critical)

    counts = [sum(mask[idx] for mask in masks.values()) for idx in range(bins)]
    critical_any = bytearray(
        1 if any(mask[idx] for mask in critical_masks.values()) else 0
        for idx in range(bins)
    )
    critical_y = y + len(UNIT_ORDER) * (lane_height + lane_gap)
    concurrency_y = critical_y + lane_height + lane_gap
    panel_bottom = concurrency_y + 72
    doc.rect(
        62,
        y - 43,
        1796,
        panel_bottom - (y - 43),
        fill=COLOR["panel"],
        rx=12,
    )
    doc.text(82, y - 10, label, css_class="panel-title")
    doc.text(
        1805,
        y - 10,
        f"makespan {schedule.makespan:,} cycles",
        css_class="metric-label",
        fill=COLOR["muted"],
        anchor="end",
    )

    # Draw lane tracks first, then lightly shade periods where two or more
    # resources are active.  This provides a large-area parallelism cue that
    # remains visible when individual commands are only a few cycles long.
    for lane_index, unit in enumerate(UNIT_ORDER):
        lane_y = y + lane_index * (lane_height + lane_gap)
        doc.text(
            plot_x - 24,
            lane_y + lane_height * 0.72,
            f"{UNIT_LABELS[unit]}  {utilization[unit] * 100:.1f}%",
            css_class="lane",
            anchor="end",
        )
        doc.rect(plot_x, lane_y, plot_width, lane_height, fill=COLOR["panel_alt"], rx=3)

    resource_height = (
        (len(UNIT_ORDER) - 1) * (lane_height + lane_gap) + lane_height
    )
    parallel_mask = bytearray(1 if count >= 2 else 0 for count in counts)
    for start, finish, value in _runs(parallel_mask):
        if not value:
            continue
        x = plot_x + start / bins * plot_width
        width = max((finish - start) / bins * plot_width, 1.5)
        doc.rect(
            x,
            y,
            width,
            resource_height,
            fill=COLOR["parallel_window"],
            opacity=0.72,
        )

    for lane_index, unit in enumerate(UNIT_ORDER):
        lane_y = y + lane_index * (lane_height + lane_gap)
        for start, finish, value in _runs(masks[unit]):
            if not value:
                continue
            x = plot_x + start / bins * plot_width
            width = max((finish - start) / bins * plot_width, 1.5)
            doc.rect(x, lane_y, width, lane_height, fill=UNIT_COLORS[unit], rx=1.5)
        for start, finish, value in _runs(critical_masks[unit]):
            if not value:
                continue
            x = plot_x + start / bins * plot_width
            width = max((finish - start) / bins * plot_width, 1.5)
            marker_height = max(6.0, lane_height * 0.18)
            doc.rect(
                x,
                lane_y,
                width,
                marker_height,
                fill=COLOR["critical"],
                rx=1.5,
            )

    doc.text(
        plot_x - 24,
        critical_y + lane_height * 0.72,
        f"Critical path  {critical_event_count} events",
        css_class="lane",
        fill=COLOR["critical"],
        anchor="end",
    )
    doc.rect(plot_x, critical_y, plot_width, lane_height, fill="#FEE2E2", rx=3)
    for start, finish, value in _runs(critical_any):
        if not value:
            continue
        x = plot_x + start / bins * plot_width
        width = max((finish - start) / bins * plot_width, 1.5)
        doc.rect(x, critical_y, width, lane_height, fill=COLOR["critical"], rx=1.5)

    doc.text(
        plot_x - 24,
        concurrency_y + 21,
        "Parallel units  1 / 2 / 3+",
        css_class="lane",
        anchor="end",
    )
    concurrency_height = max(24.0, lane_height * 0.65)
    doc.rect(
        plot_x,
        concurrency_y,
        plot_width,
        concurrency_height,
        fill=COLOR["panel_alt"],
        rx=2,
    )
    for start, finish, count in _runs(counts):
        if count <= 0:
            continue
        fill = (
            COLOR["concurrency_1"]
            if count == 1
            else COLOR["concurrency_2"]
            if count == 2
            else COLOR["concurrency_3"]
        )
        x = plot_x + start / bins * plot_width
        width = max((finish - start) / bins * plot_width, 1.5)
        doc.rect(x, concurrency_y, width, concurrency_height, fill=fill, rx=1)

    if show_axis:
        _draw_axis(
            doc,
            plot_x=plot_x,
            plot_y=y,
            plot_width=plot_width,
            plot_height=concurrency_y + concurrency_height - y,
            domain_cycles=domain_cycles,
            label_y=concurrency_y + concurrency_height + 28,
        )
    return panel_bottom


def _draw_legend(doc: SvgDocument, *, y: float) -> None:
    x = 80.0
    for unit in UNIT_ORDER:
        doc.rect(x, y - 13, 25, 12, fill=UNIT_COLORS[unit], rx=2)
        doc.text(x + 34, y - 2, UNIT_LABELS[unit], css_class="small", fill=COLOR["muted"])
        x += 245
    doc.rect(x, y - 16, 25, 16, fill=COLOR["critical"], rx=2)
    doc.text(x + 34, y - 2, "Critical event / path", css_class="small", fill=COLOR["muted"])


def render_run_comparison(run: RunSelection, output: Path) -> None:
    doc = SvgDocument()
    _draw_header(doc, run)
    _draw_metric_strip(doc, run)
    _draw_schedule_panel(
        doc,
        run.baseline,
        label="Baseline · full-width timeline",
        y=275,
        domain_cycles=run.baseline.makespan,
        lane_height=25,
        lane_gap=8,
    )
    _draw_schedule_panel(
        doc,
        run.best,
        label=f"{run.best_label} · full-width timeline",
        y=650,
        domain_cycles=run.best.makespan,
        lane_height=25,
        lane_gap=8,
    )
    _draw_legend(doc, y=1048)
    output.write_text(doc.finish(), encoding="utf-8")


def _schedule_metric(schedule: Schedule, name: str, fallback: int = 0) -> int:
    try:
        return int(schedule.metrics.get(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _percent_change(before: int, after: int) -> float:
    if before == 0:
        return 0.0
    return (after - before) / before * 100


def render_parallelism_regression(run: RunSelection, output: Path) -> None:
    """Explain a rejected candidate whose reduced work lost more overlap."""

    baseline = run.baseline
    candidate = run.best
    baseline_serial = baseline.serial_cycles
    candidate_serial = candidate.serial_cycles
    baseline_saving = _schedule_metric(
        baseline,
        "overlap_saved_cycles",
        max(baseline_serial - baseline.makespan, 0),
    )
    candidate_saving = _schedule_metric(
        candidate,
        "overlap_saved_cycles",
        max(candidate_serial - candidate.makespan, 0),
    )
    baseline_bytes = _schedule_metric(baseline, "total_bytes")
    candidate_bytes = _schedule_metric(candidate, "total_bytes")
    work_delta = candidate_serial - baseline_serial
    overlap_delta = baseline_saving - candidate_saving
    makespan_delta = candidate.makespan - baseline.makespan

    doc = SvgDocument()
    doc.text(78, 72, "负优化归因：减少工作量，不等于缩短完成时间", css_class="title")
    doc.text(
        80,
        111,
        f"{run.name} · Candidate Iteration {run.best_iteration} · 未晋升",
        css_class="subtitle",
        fill=COLOR["muted"],
    )

    dram_change = _percent_change(baseline_bytes, candidate_bytes)
    facts = [
        (
            "DRAM traffic",
            f"{baseline_bytes:,} → {candidate_bytes:,} B",
            f"{dram_change:.1f}%",
            COLOR["best"],
        ),
        (
            "Serial work",
            f"{baseline_serial:,} → {candidate_serial:,}",
            f"{work_delta:+,} cycles",
            COLOR["best"],
        ),
        (
            "Overlap saving",
            f"{baseline_saving:,} → {candidate_saving:,}",
            f"{-overlap_delta:+,} cycles",
            COLOR["critical"],
        ),
        (
            "RTL makespan",
            f"{baseline.makespan:,} → {candidate.makespan:,}",
            f"{makespan_delta:+,} cycles",
            COLOR["critical"],
        ),
    ]
    for x, (name, values, delta, color) in zip((80, 525, 970, 1415), facts):
        doc.text(x, 167, name, css_class="metric-label", fill=COLOR["muted"])
        doc.text(x, 207, values, css_class="panel-title", fill=COLOR["foreground"])
        doc.text(x, 240, delta, css_class="metric-label", fill=color)

    doc.text(
        80,
        307,
        "Serial work = RTL makespan + cycles hidden by parallel execution",
        css_class="panel-title",
    )
    plot_x = 390.0
    plot_width = 1410.0
    domain = max(baseline_serial, candidate_serial, 1)
    rows = (
        ("Baseline", baseline, baseline_saving, COLOR["baseline"], 395.0),
        ("Candidate", candidate, candidate_saving, COLOR["critical"], 555.0),
    )
    for label, schedule, saving, main_color, y in rows:
        serial = schedule.serial_cycles
        makespan_width = schedule.makespan / domain * plot_width
        saving_width = saving / domain * plot_width
        doc.text(plot_x - 28, y + 42, label, css_class="panel-title", anchor="end")
        doc.rect(plot_x, y, plot_width, 66, fill=COLOR["panel_alt"], rx=5)
        doc.rect(plot_x, y, makespan_width, 66, fill=main_color, rx=5)
        doc.rect(
            plot_x + makespan_width,
            y,
            saving_width,
            66,
            fill=COLOR["concurrency_2"],
            rx=3,
        )
        doc.text(
            plot_x + 18,
            y + 42,
            f"visible makespan  {schedule.makespan:,}",
            css_class="panel-title",
            fill="#FFFFFF",
        )
        doc.text(
            plot_x + makespan_width + saving_width / 2,
            y + 42,
            f"overlap  {saving:,}",
            css_class="lane",
            fill=COLOR["foreground"],
            anchor="middle",
        )
        doc.text(
            plot_x + serial / domain * plot_width,
            y + 94,
            f"serial {serial:,}",
            css_class="axis",
            fill=COLOR["muted"],
            anchor="end",
        )

    doc.line(plot_x, 700, plot_x + plot_width, 700, stroke=COLOR["grid"], width=2)
    for tick in _nice_ticks(domain):
        x = plot_x + tick / domain * plot_width
        doc.line(x, 693, x, 707, stroke=COLOR["grid"], width=2)
        anchor = "middle" if 0 < tick < domain else "start" if tick == 0 else "end"
        doc.text(
            x,
            733,
            _format_cycles(tick),
            css_class="axis",
            fill=COLOR["muted"],
            anchor=anchor,
        )
    doc.text(
        plot_x + plot_width / 2,
        765,
        "cycles",
        css_class="axis",
        fill=COLOR["muted"],
        anchor="middle",
    )

    equations = [
        (170, "Work removed", f"{work_delta:+,}", "cycles", COLOR["best"]),
        (720, "Overlap lost", f"+{overlap_delta:,}", "cycles", COLOR["critical"]),
        (1270, "Net result", f"+{makespan_delta:,}", "cycles slower", COLOR["critical"]),
    ]
    for x, label, value, unit, color in equations:
        doc.text(x, 842, label, css_class="metric-label", fill=COLOR["muted"])
        doc.text(x, 892, value, css_class="metric", fill=color)
        doc.text(x + 130, 890, unit, css_class="metric-label", fill=COLOR["muted"])
    doc.text(620, 887, "+", css_class="metric", fill=COLOR["muted"], anchor="middle")
    doc.text(1170, 887, "=", css_class="metric", fill=COLOR["muted"], anchor="middle")

    doc.rect(80, 950, 1760, 66, fill="#FEF2F2", rx=8)
    doc.text(
        960,
        992,
        (
            f"少做 {abs(work_delta):,} cycles，但少隐藏 {overlap_delta:,} cycles；"
            f"并行损失超过工作量收益，最终慢 {makespan_delta:,} cycles。"
        ),
        css_class="panel-title",
        fill=COLOR["critical"],
        anchor="middle",
    )
    output.write_text(doc.finish(), encoding="utf-8")


STRUCTURAL_EVENT_FIELDS = (
    "op",
    "target_unit",
    "duration_cycles",
    "memory",
    "resources",
    "tensor_reads",
    "tensor_writes",
)


def changed_event_pairs(
    baseline: Schedule,
    candidate: Schedule,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return aligned commands whose semantics/resource mapping changed.

    Timing fields are deliberately excluded: one early duration change shifts
    thousands of later timestamps without changing those later commands.
    """

    candidate_by_key = {
        (event.get("raw_instruction_idx"), event.get("expanded_idx", 0)): event
        for event in candidate.events
        if event.get("raw_instruction_idx") is not None
    }
    use_index_alignment = (
        len(candidate_by_key) != len(candidate.events)
        or any(
            event.get("raw_instruction_idx") is None
            for event in baseline.events
        )
    )
    changed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, baseline_event in enumerate(baseline.events):
        if use_index_alignment:
            candidate_event = (
                candidate.events[index] if index < len(candidate.events) else None
            )
        else:
            key = (
                baseline_event.get("raw_instruction_idx"),
                baseline_event.get("expanded_idx", 0),
            )
            candidate_event = candidate_by_key.get(key)
        if candidate_event is None:
            continue
        if any(
            baseline_event.get(field) != candidate_event.get(field)
            for field in STRUCTURAL_EVENT_FIELDS
        ):
            changed.append((baseline_event, candidate_event))
    return changed


def concurrency_distribution(schedule: Schedule) -> dict[int, int]:
    """Count cycles by number of simultaneously active target units."""

    changes: dict[int, int] = {}
    unit_intervals = intervals_by_unit(schedule)
    for records in unit_intervals.values():
        merged = _merge_intervals((start, finish) for start, finish, _ in records)
        for start, finish in merged:
            changes[start] = changes.get(start, 0) + 1
            changes[finish] = changes.get(finish, 0) - 1
    changes.setdefault(0, 0)
    changes.setdefault(schedule.makespan, 0)

    distribution: dict[int, int] = {}
    active = 0
    previous = 0
    for cycle in sorted(changes):
        if cycle > previous:
            distribution[active] = distribution.get(active, 0) + cycle - previous
        active += changes[cycle]
        previous = cycle
    return distribution


def _event_change_label(event: dict[str, Any]) -> str:
    op = str(event.get("op") or "unknown")
    unit = str(event.get("target_unit") or "unknown").capitalize()
    duration = int(event.get("duration_cycles") or 0)
    return f"{op} · {unit} · {duration} cycles"


def _estimated_label_width(
    value: str,
    *,
    font_size: float = 13.0,
    horizontal_padding: float = 16.0,
) -> float:
    """Conservatively estimate label width without relying on host fonts."""

    latin_width = font_size * 0.65
    glyph_width = sum(font_size if ord(char) > 127 else latin_width for char in value)
    return glyph_width + horizontal_padding


def _task_tensor_label(value: object) -> str:
    """Use the compact, report-friendly tensor notation used inside task boxes."""

    tensor = str(value)
    match = re.fullmatch(r"([A-Za-z0-9]+)_pos(\d+)_tile(\d+)", tensor)
    if match:
        return f"{match.group(1).upper()} p{match.group(2)}t{match.group(3)}"
    return tensor if len(tensor) <= 11 else tensor[:9] + "…"


def _event_task_name(event: dict[str, Any]) -> str:
    """Derive a semantic task title from fields present in the RTL schedule."""

    op = str(event.get("op") or "")
    writes = list(event.get("tensor_writes") or [])
    reads = list(event.get("tensor_reads") or [])
    tensors = writes or reads
    tensor = _task_tensor_label(tensors[0]) if tensors else ""
    action_by_op = {
        "V_WR_DRAM": "Save",
        "V_RD_DRAM": "Load",
        "V_WR": "Cache",
        "V_RD": "Read",
        "M_WR_DRAM": "Save matrix",
        "M_RD_DRAM": "Load matrix",
        "M_WR": "Matrix write",
        "M_RD": "Matrix read",
        "MV_MUL": "MV multiply",
        "VV_ADD": "Vector add",
        "VV_MUL": "Vector multiply",
        "V_GELU": "GELU",
        "V_FUNC": "Vector function",
    }
    task = action_by_op.get(op, op.replace("_", " ").title())
    if tensor and op in {"V_WR_DRAM", "V_RD_DRAM", "V_WR", "V_RD"}:
        return f"{task} {tensor}"
    return task


def _event_box_label_for_width(
    event: dict[str, Any],
    available_width: float,
) -> tuple[str, str | None] | None:
    """Choose a two-level task label, degrading cleanly for narrow blocks."""

    op = str(event.get("op") or "")
    if not op:
        return None
    task = _event_task_name(event)
    duration = int(event.get("duration_cycles") or 0)
    detail = f"{op}  ·  {duration} cy"
    duration_detail = f"{duration} cy"
    if (
        _estimated_label_width(task) <= available_width
        and _estimated_label_width(detail, font_size=11.0) <= available_width
    ):
        return task, detail
    if (
        _estimated_label_width(task) <= available_width
        and _estimated_label_width(duration_detail, font_size=11.0)
        <= available_width
    ):
        return task, duration_detail

    compact_aliases = {
        "V_RD_DRAM": "DRAM RD",
        "V_WR_DRAM": "DRAM WR",
        "M_RD_DRAM": "M DRAM RD",
        "M_WR_DRAM": "M DRAM WR",
    }
    for candidate in (op, compact_aliases.get(op, "")):
        if (
            candidate
            and _estimated_label_width(candidate) <= available_width
            and _estimated_label_width(duration_detail, font_size=11.0)
            <= available_width
        ):
            return candidate, duration_detail
    if _estimated_label_width(task) <= available_width:
        return task, None
    for candidate in (op, compact_aliases.get(op, "")):
        if candidate and _estimated_label_width(candidate) <= available_width:
            return candidate, None
    return None


def group_changed_event_pairs(
    changed: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    max_instruction_gap: int = 40,
) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Group nearby changed instructions into report-sized local windows."""

    def sequence_index(pair: tuple[dict[str, Any], dict[str, Any]]) -> int:
        event = pair[0]
        value = event.get("raw_instruction_idx", event.get("idx", 0))
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    ordered = sorted(changed, key=sequence_index)
    if not ordered:
        return []
    groups = [[ordered[0]]]
    previous = sequence_index(ordered[0])
    for pair in ordered[1:]:
        current = sequence_index(pair)
        if current - previous <= max_instruction_gap:
            groups[-1].append(pair)
        else:
            groups.append([pair])
        previous = current
    return groups


def _changed_window_label(
    group: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    ordinal: int,
    category_ordinal: int | None = None,
) -> str:
    baseline_events = [pair[0] for pair in group]
    ops = {str(event.get("op") or "") for event in baseline_events}
    tensors = [
        str(tensor)
        for event in baseline_events
        for tensor in (
            list(event.get("tensor_reads") or [])
            + list(event.get("tensor_writes") or [])
        )
    ]
    if ops == {"V_WR_DRAM"} and tensors:
        match = re.search(r"\b([kv])_pos(\d+)", tensors[0])
        if match:
            tensor_name = match.group(1).upper()
            return f"{tensor_name} cache write · position {match.group(2)}"
    if ops == {"V_RD_DRAM"}:
        return f"Attention cache reads · block {category_ordinal or ordinal}"
    return f"Changed command window {ordinal}"


def _draw_local_schedule_panel(
    doc: SvgDocument,
    schedule: Schedule,
    *,
    label: str,
    window_start: int,
    window_finish: int,
    changed_event_ids: set[int],
    y: float,
    plot_x: float = 300,
    plot_width: float = 1500,
    axis_origin: int | None = None,
) -> None:
    # Taller event boxes make room for the same two-level information hierarchy
    # as dram_clusters.svg: bold task name, then compact quantitative attributes.
    lane_height = 42.0
    lane_gap = 6.0
    bins = 750
    domain = max(window_finish - window_start, 1)
    events = [
        event
        for event in schedule.events
        if (_event_interval(event) or (0, 0))[1] > window_start
        and (_event_interval(event) or (0, 0))[0] < window_finish
    ]
    records: dict[str, list[tuple[int, int, bool]]] = {
        unit: [] for unit in UNIT_ORDER
    }
    label_candidates: list[
        tuple[int, float, float, float, str, str | None]
    ] = []
    for event in events:
        unit = str(event.get("target_unit") or "")
        interval = _event_interval(event)
        if unit not in records or interval is None:
            continue
        records[unit].append(
            (
                max(interval[0], window_start) - window_start,
                min(interval[1], window_finish) - window_start,
                bool(event.get("critical")),
            )
        )

    masks = {
        unit: _mark_bins(unit_records, domain, bins)[0]
        for unit, unit_records in records.items()
    }
    counts = [sum(mask[index] for mask in masks.values()) for index in range(bins)]
    concurrency_y = y + len(UNIT_ORDER) * (lane_height + lane_gap)
    panel_bottom = concurrency_y + 77
    doc.rect(
        62,
        y - 43,
        1796,
        panel_bottom - (y - 43),
        fill=COLOR["panel"],
        rx=12,
    )
    doc.text(82, y - 10, label, css_class="panel-title")
    if axis_origin is None:
        range_label = f"cycles [{window_start:,}, {window_finish:,})"
    else:
        relative_start = window_start - axis_origin
        relative_finish = window_finish - axis_origin
        range_label = (
            f"absolute [{window_start:,}, {window_finish:,}) · "
            f"t=[{relative_start:+,}, {relative_finish:+,})"
        )
    doc.text(
        1800,
        y - 10,
        range_label,
        css_class="metric-label",
        fill=COLOR["muted"],
        anchor="end",
    )

    for lane_index, unit in enumerate(UNIT_ORDER):
        lane_y = y + lane_index * (lane_height + lane_gap)
        doc.text(
            plot_x - 24,
            lane_y + 22,
            UNIT_LABELS[unit],
            css_class="lane",
            anchor="end",
        )
        doc.rect(plot_x, lane_y, plot_width, lane_height, fill=COLOR["panel_alt"], rx=3)

    # Mark changed command spans across all lanes before drawing the events.
    for event in events:
        if id(event) not in changed_event_ids:
            continue
        interval = _event_interval(event)
        if interval is None:
            continue
        clipped_start = max(interval[0], window_start)
        clipped_finish = min(interval[1], window_finish)
        x = plot_x + (clipped_start - window_start) / domain * plot_width
        width = max((clipped_finish - clipped_start) / domain * plot_width, 4)
        doc.rect(
            x - 3,
            y - 4,
            width + 6,
            len(UNIT_ORDER) * (lane_height + lane_gap) - lane_gap + 8,
            fill="#FEE2E2",
            opacity=0.72,
            rx=2,
        )

    for event in events:
        unit = str(event.get("target_unit") or "")
        if unit not in UNIT_ORDER:
            continue
        interval = _event_interval(event)
        if interval is None:
            continue
        clipped_start = max(interval[0], window_start)
        clipped_finish = min(interval[1], window_finish)
        x = plot_x + (clipped_start - window_start) / domain * plot_width
        width = max((clipped_finish - clipped_start) / domain * plot_width, 3)
        lane_index = UNIT_ORDER.index(unit)
        lane_y = y + lane_index * (lane_height + lane_gap)
        changed_event = id(event) in changed_event_ids
        doc.rect(
            x,
            lane_y,
            width,
            lane_height,
            fill=UNIT_COLORS[unit],
            rx=2,
            stroke=COLOR["critical"] if changed_event else None,
            stroke_width=4 if changed_event else 1,
        )
        event_label = _event_box_label_for_width(event, width)
        if event_label:
            label_candidates.append(
                (lane_index, x, width, lane_y, event_label[0], event_label[1])
            )
        if bool(event.get("critical")) and not changed_event:
            doc.rect(x, lane_y, width, 4, fill=COLOR["critical"], rx=1)

    # Labels are drawn after all marks so later pipelined events cannot paint
    # over their text.  Suppress labels whose measured ranges would collide
    # with another label on the same resource lane.
    last_label_end: dict[int, float] = {}
    for lane_index, x, width, lane_y, task_label, detail_label in sorted(
        label_candidates
    ):
        task_width = _estimated_label_width(task_label) - 16
        detail_width = (
            _estimated_label_width(detail_label, font_size=11.0) - 16
            if detail_label
            else 0
        )
        text_width = max(task_width, detail_width)
        text_x = x + width / 2
        text_left = text_x - text_width / 2
        text_right = text_x + text_width / 2
        if text_left < last_label_end.get(lane_index, -math.inf) + 8:
            continue
        if text_left < x + 5 or text_right > x + width - 5:
            continue
        doc.text(
            text_x,
            lane_y + (17 if detail_label else 27),
            task_label,
            css_class="event-label",
            fill="#FFFFFF",
            anchor="middle",
        )
        if detail_label:
            doc.text(
                text_x,
                lane_y + 33,
                detail_label,
                css_class="event-label-secondary",
                fill="#FFFFFF",
                anchor="middle",
            )
        last_label_end[lane_index] = text_right

    doc.text(
        plot_x - 24,
        concurrency_y + 21,
        "Parallel units",
        css_class="lane",
        anchor="end",
    )
    doc.rect(plot_x, concurrency_y, plot_width, 25, fill=COLOR["panel_alt"], rx=2)
    for start, finish, count in _runs(counts):
        if count <= 0:
            continue
        color = (
            COLOR["concurrency_1"]
            if count == 1
            else COLOR["concurrency_2"]
            if count == 2
            else COLOR["concurrency_3"]
        )
        x = plot_x + start / bins * plot_width
        width = max((finish - start) / bins * plot_width, 2)
        doc.rect(x, concurrency_y, width, 25, fill=color, rx=1)

    for tick in _nice_ticks(domain, count=6):
        actual_cycle = window_start + tick
        x = plot_x + tick / domain * plot_width
        doc.line(x, y, x, concurrency_y + 25, stroke=COLOR["grid"], width=1)
        anchor = "middle" if 0 < tick < domain else "start" if tick == 0 else "end"
        doc.text(
            x,
            concurrency_y + 52,
            (
                f"{actual_cycle - axis_origin:+,}"
                if axis_origin is not None and actual_cycle != axis_origin
                else "0"
                if axis_origin is not None
                else f"{actual_cycle:,}"
            ),
            css_class="axis",
            fill=COLOR["muted"],
            anchor=anchor,
        )
    doc.text(
        plot_x + plot_width / 2,
        concurrency_y + 76,
        "Cycles from Attention start (t=0)"
        if axis_origin is not None
        else "RTL cycles",
        css_class="axis",
        fill=COLOR["muted"],
        anchor="middle",
    )
    if axis_origin is not None and window_start <= axis_origin <= window_finish:
        marker_x = plot_x + (axis_origin - window_start) / domain * plot_width
        doc.line(
            marker_x,
            y - 2,
            marker_x,
            concurrency_y + 25,
            stroke=COLOR["critical"],
            width=3,
            dash="8 5",
        )
        doc.polygon(
            [
                (marker_x, y - 2),
                (marker_x - 7, y - 13),
                (marker_x + 7, y - 13),
            ],
            fill=COLOR["critical"],
        )
        doc.text(
            marker_x,
            concurrency_y + 52,
            "0",
            css_class="axis",
            fill=COLOR["critical"],
            anchor="middle",
        )


def _find_attention_start_event(
    schedule: Schedule,
    group_events: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the Q-projection setup that starts the enclosing Attention call.

    ``mvm_tiled_q`` begins with three consecutive ``S_WR`` configuration
    commands.  The nearest such triplet before the changed K/V-read group is a
    schedule-derived phase marker and remains stable across the compared
    firmware variants because their instruction indices are aligned.
    """

    first_raw = min(
        int(event.get("raw_instruction_idx", event.get("idx", 0)) or 0)
        for event in group_events
    )
    event_by_raw: dict[int, dict[str, Any]] = {}
    for event in schedule.events:
        if int(event.get("expanded_idx", 0) or 0) != 0:
            continue
        raw = int(event.get("raw_instruction_idx", event.get("idx", 0)) or 0)
        event_by_raw.setdefault(raw, event)
    for raw in range(first_raw - 1, max(first_raw - 256, -1), -1):
        triplet = [event_by_raw.get(raw + offset) for offset in range(3)]
        if all(event and event.get("op") == "S_WR" for event in triplet):
            return triplet[0]
    return None


def render_changed_window(
    run: RunSelection,
    group: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    ordinal: int,
    total_windows: int,
    context_cycles: int,
    output: Path,
    window_label: str | None = None,
    alignment: str = "absolute",
) -> dict[str, Any]:
    baseline_events = [pair[0] for pair in group]
    candidate_events = [pair[1] for pair in group]
    baseline_start = max(
        min(int(event.get("start_cycle", 0)) for event in baseline_events)
        - context_cycles,
        0,
    )
    baseline_finish = min(
        max(int(event.get("finish_cycle", 0)) for event in baseline_events)
        + context_cycles,
        run.baseline.makespan,
    )
    candidate_start = max(
        min(int(event.get("start_cycle", 0)) for event in candidate_events)
        - context_cycles,
        0,
    )
    candidate_finish = min(
        max(int(event.get("finish_cycle", 0)) for event in candidate_events)
        + context_cycles,
        run.best.makespan,
    )
    label = window_label or _changed_window_label(group, ordinal)
    baseline_origin: int | None = None
    candidate_origin: int | None = None
    relative_cycles: list[int] | None = None
    attention_aligned = alignment == "attention-start" and label.startswith(
        "Attention"
    )
    if attention_aligned:
        baseline_start_event = _find_attention_start_event(
            run.baseline, baseline_events
        )
        candidate_start_event = _find_attention_start_event(
            run.best, candidate_events
        )
        if baseline_start_event is None or candidate_start_event is None:
            raise VisualizationError(
                "cannot locate Attention start before changed-command group"
            )
        baseline_origin = int(baseline_start_event.get("start_cycle", 0))
        candidate_origin = int(candidate_start_event.get("start_cycle", 0))
        baseline_relative_start = baseline_start - baseline_origin
        baseline_relative_finish = baseline_finish - baseline_origin
        candidate_relative_start = candidate_start - candidate_origin
        candidate_relative_finish = candidate_finish - candidate_origin
        pre_start_context = min(context_cycles, 12)
        shared_relative_start = min(
            -pre_start_context,
            baseline_relative_start,
            candidate_relative_start,
        )
        shared_relative_finish = max(
            baseline_relative_finish,
            candidate_relative_finish,
        )
        baseline_panel_start = baseline_origin + shared_relative_start
        baseline_panel_finish = baseline_origin + shared_relative_finish
        candidate_panel_start = candidate_origin + shared_relative_start
        candidate_panel_finish = candidate_origin + shared_relative_finish
        relative_cycles = [shared_relative_start, shared_relative_finish]
        alignment_note = "以 Attention start (t=0) 对齐"
    else:
        # Both panels use the same absolute-cycle coordinate system.  Taking
        # the union ensures a given cycle maps to the exact same x position.
        shared_start = min(baseline_start, candidate_start)
        shared_finish = max(baseline_finish, candidate_finish)
        baseline_panel_start = candidate_panel_start = shared_start
        baseline_panel_finish = candidate_panel_finish = shared_finish
        alignment_note = "上下共享绝对 cycle 刻度"
    raw_indices = [
        int(event.get("raw_instruction_idx", event.get("idx", 0)) or 0)
        for event in baseline_events
    ]
    before_ops = Counter(str(event.get("op") or "unknown") for event in baseline_events)
    after_ops = Counter(str(event.get("op") or "unknown") for event in candidate_events)

    doc = SvgDocument()
    doc.text(78, 72, f"修改指令附近的时间轴 · {label}", css_class="title")
    doc.text(
        80,
        111,
        (
            f"Window {ordinal}/{total_windows} · raw instructions "
            f"{min(raw_indices)}–{max(raw_indices)} · 前后各保留 {context_cycles} cycles"
            f" · {alignment_note}"
        ),
        css_class="subtitle",
        fill=COLOR["muted"],
    )
    before_summary = ", ".join(
        f"{count}× {op}" for op, count in before_ops.most_common()
    )
    after_summary = ", ".join(
        f"{count}× {op}" for op, count in after_ops.most_common()
    )
    doc.text(80, 157, "Changed commands", css_class="metric-label", fill=COLOR["muted"])
    doc.text(
        80,
        194,
        f"Baseline: {before_summary}  →  Candidate: {after_summary}",
        css_class="panel-title",
    )
    doc.rect(1570, 172, 30, 18, fill="#FEE2E2", stroke=COLOR["critical"], stroke_width=3, rx=2)
    doc.text(1614, 188, "changed event", css_class="small", fill=COLOR["muted"])
    if attention_aligned:
        doc.line(1320, 171, 1320, 192, stroke=COLOR["critical"], width=3, dash="7 4")
        doc.text(
            1335,
            188,
            "Attention start (t=0)",
            css_class="small",
            fill=COLOR["muted"],
        )

    _draw_local_schedule_panel(
        doc,
        run.baseline,
        label="Baseline · local context",
        window_start=baseline_panel_start,
        window_finish=baseline_panel_finish,
        changed_event_ids={id(event) for event in baseline_events},
        y=270,
        axis_origin=baseline_origin,
    )
    _draw_local_schedule_panel(
        doc,
        run.best,
        label="Candidate · local context",
        window_start=candidate_panel_start,
        window_finish=candidate_panel_finish,
        changed_event_ids={id(event) for event in candidate_events},
        y=665,
        axis_origin=candidate_origin,
    )
    output.write_text(doc.finish(), encoding="utf-8")
    return {
        "window": ordinal,
        "label": label,
        "raw_instruction_start": min(raw_indices),
        "raw_instruction_finish": max(raw_indices),
        "changed_event_count": len(group),
        "alignment": "attention-start" if attention_aligned else "absolute",
        "baseline_cycles": [baseline_panel_start, baseline_panel_finish],
        "candidate_cycles": [candidate_panel_start, candidate_panel_finish],
        "shared_cycles": relative_cycles
        if relative_cycles is not None
        else [baseline_panel_start, baseline_panel_finish],
        "baseline_attention_start": baseline_origin,
        "candidate_attention_start": candidate_origin,
        "file": str(output.resolve()),
    }


def render_changed_windows(
    run: RunSelection,
    output_dir: Path,
    *,
    context_cycles: int = 24,
    alignment: str = "absolute",
) -> tuple[list[Path], Path]:
    if context_cycles < 1:
        raise VisualizationError("--window-context-cycles must be positive")
    groups = group_changed_event_pairs(changed_event_pairs(run.baseline, run.best))
    if not groups:
        raise VisualizationError("candidate has no aligned structural event changes")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    windows: list[dict[str, Any]] = []
    category_counts: Counter[tuple[str, ...]] = Counter()
    for ordinal, group in enumerate(groups, 1):
        category = tuple(
            sorted({str(pair[0].get("op") or "unknown") for pair in group})
        )
        category_counts[category] += 1
        window_label = _changed_window_label(
            group,
            ordinal,
            category_counts[category],
        )
        path = output_dir / f"changed-window-{ordinal:02d}.svg"
        windows.append(
            render_changed_window(
                run,
                group,
                ordinal=ordinal,
                total_windows=len(groups),
                context_cycles=context_cycles,
                output=path,
                window_label=window_label,
                alignment=alignment,
            )
        )
        generated.append(path)
    manifest_path = output_dir / "changed-windows-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run": run.name,
                "candidate_iteration": run.best_iteration,
                "context_cycles": context_cycles,
                "alignment": alignment,
                "window_count": len(windows),
                "windows": windows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return generated, manifest_path


def render_candidate_diff(run: RunSelection, output: Path) -> None:
    """Render only structural command changes and their parallelism impact."""

    changed = changed_event_pairs(run.baseline, run.best)
    transformations = Counter(
        (_event_change_label(before), _event_change_label(after))
        for before, after in changed
    )
    baseline_concurrency = concurrency_distribution(run.baseline)
    candidate_concurrency = concurrency_distribution(run.best)
    baseline_parallel = sum(
        cycles for count, cycles in baseline_concurrency.items() if count >= 2
    )
    candidate_parallel = sum(
        cycles for count, cycles in candidate_concurrency.items() if count >= 2
    )
    parallel_delta = candidate_parallel - baseline_parallel
    makespan_delta = run.best.makespan - run.baseline.makespan
    changed_ratio = len(changed) / max(len(run.baseline.events), 1) * 100

    doc = SvgDocument()
    doc.text(78, 72, "Baseline vs Candidate：仅显示差异", css_class="title")
    doc.text(
        80,
        111,
        (
            f"{run.name} · Iteration {run.best_iteration} · "
            "相同指令和纯时间位移已折叠"
        ),
        css_class="subtitle",
        fill=COLOR["muted"],
    )

    facts = [
        (
            "Structurally changed",
            f"{len(changed):,} / {len(run.baseline.events):,}",
            f"{changed_ratio:.1f}% of commands",
            COLOR["foreground"],
        ),
        (
            "Parallel overlap (2+ units)",
            f"{baseline_parallel:,} → {candidate_parallel:,}",
            f"{parallel_delta:+,} cycles",
            COLOR["critical"],
        ),
        (
            "RTL makespan",
            f"{run.baseline.makespan:,} → {run.best.makespan:,}",
            f"{makespan_delta:+,} cycles",
            COLOR["critical"],
        ),
    ]
    for x, (label, value, context, color) in zip((80, 690, 1300), facts):
        doc.text(x, 161, label, css_class="metric-label", fill=COLOR["muted"])
        doc.text(x, 205, value, css_class="metric", fill=color)
        doc.text(x, 238, context, css_class="metric-label", fill=color)

    doc.text(80, 303, "Only changed commands", css_class="panel-title")
    top_transformations = transformations.most_common(3)
    for index, ((before_label, after_label), count) in enumerate(top_transformations):
        y = 340 + index * 92
        before_unit = before_label.split(" · ")[1].lower()
        after_unit = after_label.split(" · ")[1].lower()
        before_color = UNIT_COLORS.get(before_unit, COLOR["baseline"])
        after_color = UNIT_COLORS.get(after_unit, COLOR["best"])
        doc.text(80, y + 38, f"{count}×", css_class="panel-title")
        doc.rect(165, y, 650, 58, fill=before_color, rx=6)
        doc.text(190, y + 38, before_label, css_class="panel-title", fill="#FFFFFF")
        doc.line(845, y + 29, 985, y + 29, stroke=COLOR["muted"], width=3)
        doc.polygon(
            [(985, y + 29), (965, y + 18), (965, y + 40)],
            fill=COLOR["muted"],
        )
        doc.rect(1015, y, 760, 58, fill=after_color, rx=6)
        doc.text(1040, y + 38, after_label, css_class="panel-title", fill="#FFFFFF")

    if not top_transformations:
        doc.text(
            165,
            378,
            "No structurally changed aligned commands",
            css_class="panel-title",
            fill=COLOR["muted"],
        )

    chart_y = 645.0
    doc.text(80, chart_y, "Only changed parallelism outcome", css_class="panel-title")
    plot_x = 390.0
    plot_width = 1385.0
    domain = max(baseline_parallel, candidate_parallel, 1)
    parallel_rows = (
        ("Baseline", baseline_parallel, COLOR["concurrency_3"], chart_y + 48),
        ("Candidate", candidate_parallel, COLOR["concurrency_2"], chart_y + 130),
    )
    for label, cycles, color, y in parallel_rows:
        width = cycles / domain * plot_width
        doc.text(plot_x - 28, y + 34, label, css_class="panel-title", anchor="end")
        doc.rect(plot_x, y, plot_width, 52, fill=COLOR["panel_alt"], rx=5)
        doc.rect(plot_x, y, width, 52, fill=color, rx=5)
        doc.text(
            plot_x + 18,
            y + 35,
            f"{cycles:,} cycles with 2+ active units",
            css_class="panel-title",
            fill="#FFFFFF",
        )

    two_way_delta = candidate_concurrency.get(2, 0) - baseline_concurrency.get(2, 0)
    three_way_delta = candidate_concurrency.get(3, 0) - baseline_concurrency.get(3, 0)
    doc.text(
        390,
        885,
        (
            f"2-way parallel: {baseline_concurrency.get(2, 0):,} → "
            f"{candidate_concurrency.get(2, 0):,}  ({two_way_delta:+,})"
        ),
        css_class="panel-title",
        fill=COLOR["critical"],
    )
    doc.text(
        1120,
        885,
        (
            f"3-way parallel: {baseline_concurrency.get(3, 0):,} → "
            f"{candidate_concurrency.get(3, 0):,}  ({three_way_delta:+,})"
        ),
        css_class="panel-title",
        fill=COLOR["critical"],
    )

    doc.rect(80, 950, 1760, 66, fill="#FEF2F2", rx=8)
    doc.text(
        960,
        992,
        (
            f"仅 {changed_ratio:.1f}% 指令发生结构变化，但并行执行减少 "
            f"{abs(parallel_delta):,} cycles，最终 makespan 增加 {makespan_delta:,} cycles。"
        ),
        css_class="panel-title",
        fill=COLOR["critical"],
        anchor="middle",
    )
    output.write_text(doc.finish(), encoding="utf-8")


def render_schedule_detail(
    run: RunSelection,
    schedule: Schedule,
    *,
    label: str,
    output: Path,
) -> None:
    doc = SvgDocument()
    _draw_header(doc, run, detail=label)
    serial = schedule.serial_cycles
    parallel_saving = max(serial - schedule.makespan, 0)
    metrics = [
        ("Makespan", f"{schedule.makespan:,}", "cycles"),
        ("Serial work", f"{serial:,}", "cycles"),
        ("Parallel saving", f"{parallel_saving:,}", "cycles"),
    ]
    for x, (name, value, unit) in zip((80, 555, 1030), metrics):
        doc.text(x, 152, name, css_class="metric-label", fill=COLOR["muted"])
        doc.text(x, 194, value, css_class="metric", fill=COLOR["best"])
        doc.text(x + 215, 192, unit, css_class="metric-label", fill=COLOR["muted"])
    _draw_schedule_panel(
        doc,
        schedule,
        label=f"{label} · own cycle scale",
        y=300,
        domain_cycles=schedule.makespan,
        lane_height=72,
        lane_gap=20,
    )
    _draw_legend(doc, y=1048)
    output.write_text(doc.finish(), encoding="utf-8")


def _short_run_label(run: RunSelection) -> str:
    match = re.search(r"run-(\d{8})-(\d{6})", run.name)
    if not match:
        return run.name
    day = match.group(1)
    clock = match.group(2)
    return f"{day[4:6]}-{day[6:8]} {clock[:2]}:{clock[2:4]}"


def render_recent_summary(runs: Sequence[RunSelection], output: Path) -> None:
    doc = SvgDocument()
    doc.text(78, 72, "近期 RTL 闭环：Baseline vs Best", css_class="title")
    doc.text(
        80,
        111,
        "同一横轴比较各次运行的绝对 makespan；SVG 可直接插入 PowerPoint",
        css_class="subtitle",
        fill=COLOR["muted"],
    )
    max_cycles = max(run.baseline.makespan for run in runs)
    plot_x = 510.0
    plot_width = 1260.0
    top = 205.0
    row_height = min(245.0, 720.0 / max(len(runs), 1))

    for tick in _nice_ticks(max_cycles):
        x = plot_x + tick / max_cycles * plot_width
        doc.line(x, 170, x, 930, stroke=COLOR["grid"], width=1)
        anchor = "middle" if 0 < tick < max_cycles else "start" if tick == 0 else "end"
        doc.text(x, 960, _format_cycles(tick), css_class="axis", fill=COLOR["muted"], anchor=anchor)
    doc.text(
        plot_x + plot_width / 2,
        997,
        "RTL cycles",
        css_class="axis",
        fill=COLOR["muted"],
        anchor="middle",
    )

    for idx, run in enumerate(runs):
        y = top + idx * row_height
        doc.text(80, y + 14, _short_run_label(run), css_class="panel-title")
        doc.text(80, y + 44, run.goal, css_class="small", fill=COLOR["muted"])
        best_context = (
            f"Iteration {run.best_iteration}" if run.best_iteration is not None else "no promotion"
        )
        doc.text(
            80,
            y + 70,
            f"{best_context} · {run.improvement * 100:.2f}%",
            css_class="small",
            fill=COLOR["best"],
        )

        baseline_width = run.baseline.makespan / max_cycles * plot_width
        best_width = run.best.makespan / max_cycles * plot_width
        doc.rect(plot_x, y - 8, baseline_width, 32, fill=COLOR["baseline"], rx=4)
        baseline_end = plot_x + baseline_width
        baseline_inside = baseline_end + 185 > CANVAS_WIDTH - 50
        doc.text(
            baseline_end - 14 if baseline_inside else baseline_end + 14,
            y + 14,
            f"Baseline {run.baseline.makespan:,}",
            css_class="lane",
            fill="#FFFFFF" if baseline_inside else COLOR["baseline"],
            anchor="end" if baseline_inside else "start",
        )
        doc.rect(plot_x, y + 46, best_width, 32, fill=COLOR["best"], rx=4)
        best_end = plot_x + best_width
        best_inside = best_end + 155 > CANVAS_WIDTH - 50
        doc.text(
            best_end - 14 if best_inside else best_end + 14,
            y + 68,
            f"Best {run.best.makespan:,}",
            css_class="lane",
            fill="#FFFFFF" if best_inside else COLOR["best"],
            anchor="end" if best_inside else "start",
        )

    doc.rect(80, 1025, 25, 12, fill=COLOR["baseline"], rx=2)
    doc.text(115, 1036, "Baseline", css_class="small", fill=COLOR["muted"])
    doc.rect(245, 1025, 25, 12, fill=COLOR["best"], rx=2)
    doc.text(280, 1036, "Best", css_class="small", fill=COLOR["muted"])
    output.write_text(doc.finish(), encoding="utf-8")


def generate_figures(
    runs: Sequence[RunSelection],
    output_root: Path,
    *,
    view: str = "all",
    window_context_cycles: int = 24,
    window_alignment: str = "absolute",
) -> list[Path]:
    if view in {"diff", "windows"} and not any(
        run.selection_kind == "candidate" for run in runs
    ):
        raise VisualizationError(
            f"--view {view} requires an explicit --candidate"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    if not any(run.selection_kind == "candidate" for run in runs):
        summary_path = output_root / "recent-rtl-runs-summary.svg"
        render_recent_summary(runs, summary_path)
        generated.append(summary_path)

    for run in runs:
        run_output = output_root / run.name
        run_output.mkdir(parents=True, exist_ok=True)
        if (
            run.selection_kind == "candidate"
            and run.improvement < 0
            and view in {"all", "comparison", "diff"}
        ):
            regression_path = run_output / "parallelism-regression.svg"
            render_parallelism_regression(run, regression_path)
            generated.append(regression_path)
        if run.selection_kind == "candidate" and view in {
            "all",
            "comparison",
            "diff",
        }:
            diff_path = run_output / "baseline-vs-candidate-diff.svg"
            render_candidate_diff(run, diff_path)
            generated.append(diff_path)
        if run.selection_kind == "candidate" and view == "windows":
            window_paths, window_manifest = render_changed_windows(
                run,
                run_output / "changed-windows",
                context_cycles=window_context_cycles,
                alignment=window_alignment,
            )
            generated.extend([*window_paths, window_manifest])
        if view in {"all", "comparison"}:
            comparison_filename = (
                "baseline-vs-candidate.svg"
                if run.selection_kind == "candidate"
                else "baseline-vs-best.svg"
            )
            path = run_output / comparison_filename
            render_run_comparison(run, path)
            generated.append(path)
        if view in {"all", "detail"}:
            baseline_path = run_output / "baseline-detail.svg"
            selected_filename = (
                "candidate-detail.svg"
                if run.selection_kind == "candidate"
                else "best-detail.svg"
            )
            best_path = run_output / selected_filename
            render_schedule_detail(
                run, run.baseline, label="Baseline", output=baseline_path
            )
            render_schedule_detail(
                run, run.best, label=run.best_label, output=best_path
            )
            generated.extend([baseline_path, best_path])

    manifest = {
        "schema_version": 1,
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "view": view,
        "runs": [
            {
                "run": run.name,
                "goal": run.goal,
                "status": run.status,
                "baseline_cycles": run.baseline.makespan,
                "best_cycles": run.best.makespan,
                "best_iteration": run.best_iteration,
                "selection_kind": run.selection_kind,
                "improvement": run.improvement,
                "baseline_schedule": str(run.baseline.path),
                "best_schedule": str(run.best.path),
            }
            for run in runs
        ],
        "files": [str(path.resolve()) for path in generated],
    }
    manifest_path = output_root / "visualization-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generated.append(manifest_path)
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 16:9 SVG parallelism timelines for recent Verilator RTL "
            "closed-loop baselines and promoted best iterations."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Explicit run directory or run-* name. Repeat for multiple runs. "
            "When omitted, recent runs are discovered automatically."
        ),
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=3,
        help="Number of recent completed RTL runs to discover (default: 3)",
    )
    parser.add_argument(
        "--candidate",
        type=int,
        help=(
            "Compare an explicitly selected, possibly rejected iteration with "
            "the baseline. Requires exactly one --run."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Closed-loop results root (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--view",
        choices=("all", "comparison", "detail", "diff", "windows"),
        default="all",
        help=(
            "Generate all figures, full comparisons, details, or a "
            "candidate-only structural diff/local windows"
        ),
    )
    parser.add_argument(
        "--window-context-cycles",
        type=int,
        default=24,
        help=(
            "Cycles of unchanged context retained before and after each "
            "changed-command group for --view windows (default: 24)"
        ),
    )
    parser.add_argument(
        "--window-alignment",
        choices=("absolute", "attention-start"),
        default="absolute",
        help=(
            "Align local windows by absolute RTL cycles or, for Attention "
            "read windows, by the enclosing Attention start (default: absolute)"
        ),
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include interrupted/running runs during automatic discovery",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results_root = args.results_root.resolve()
    output_root = args.output.resolve()
    try:
        if args.candidate is not None and len(args.run) != 1:
            raise VisualizationError(
                "--candidate requires exactly one explicit --run"
            )
        runs = (
            resolve_requested_runs(
                args.run,
                results_root,
                candidate_iteration=args.candidate,
            )
            if args.run
            else discover_recent_runs(
                results_root,
                args.latest,
                include_incomplete=args.include_incomplete,
            )
        )
        generated = generate_figures(
            runs,
            output_root,
            view=args.view,
            window_context_cycles=args.window_context_cycles,
            window_alignment=args.window_alignment,
        )
    except VisualizationError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(output_root),
        "runs": [run.name for run in runs],
        "generated": [str(path) for path in generated],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
