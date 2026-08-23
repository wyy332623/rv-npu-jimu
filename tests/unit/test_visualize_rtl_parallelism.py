from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "visualize_rtl_parallelism.py"
SPEC = importlib.util.spec_from_file_location("visualize_rtl_parallelism", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
viz = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = viz
SPEC.loader.exec_module(viz)


def _write_schedule(path: Path, makespan: int, serial: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "backend": "verilator-rtl",
                "metrics": {
                    "rtl_predicted_npu_cycles": makespan,
                    "serial_command_cycles": serial,
                },
                "events": [
                    {
                        "target_unit": "load",
                        "start_cycle": 0,
                        "finish_cycle": makespan // 2,
                        "critical": True,
                    },
                    {
                        "target_unit": "load",
                        "start_cycle": makespan // 4,
                        "finish_cycle": 3 * makespan // 4,
                    },
                    {
                        "target_unit": "mvu",
                        "start_cycle": makespan // 2,
                        "finish_cycle": makespan,
                    },
                    {
                        "target_unit": "vector",
                        "start_cycle": 0,
                        "finish_cycle": makespan,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_run(
    root: Path,
    name: str,
    *,
    baseline: int,
    best: int | None,
    status: str = "completed",
) -> Path:
    run = root / name
    _write_schedule(run / "baseline" / "timing-schedule.json", baseline)
    best_iteration = 2 if best is not None else None
    if best is not None:
        _write_schedule(run / "iteration-2" / "timing-schedule.json", best)
    (run / "run-summary.json").write_text(
        json.dumps(
            {
                "goal": "rtl-cycle-optimization-large",
                "status": status,
                "best_iteration": best_iteration,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_load_run_selects_baseline_and_promoted_best(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path, "run-20260822-200156-21529", baseline=100, best=50
    )

    run = viz.load_run(run_dir)

    assert run.baseline.makespan == 100
    assert run.best.makespan == 50
    assert run.best_iteration == 2
    assert run.improvement == pytest.approx(0.5)
    assert run.best_label == "Best（Iteration 2）"


def test_utilization_merges_overlapping_events(tmp_path: Path) -> None:
    schedule_path = tmp_path / "timing-schedule.json"
    _write_schedule(schedule_path, makespan=100)

    utilization = viz.utilization_by_unit(viz.load_schedule(schedule_path))

    assert utilization["load"] == pytest.approx(0.75)
    assert utilization["mvu"] == pytest.approx(0.5)
    assert utilization["vector"] == pytest.approx(1.0)
    assert utilization["store"] == 0.0


def test_discovery_excludes_incomplete_and_falls_back_to_baseline(
    tmp_path: Path,
) -> None:
    completed = _write_run(
        tmp_path, "run-20260822-151237-4535", baseline=100, best=None
    )
    _write_run(
        tmp_path,
        "run-20260822-200156-21529",
        baseline=100,
        best=40,
        status="interrupted",
    )

    runs = viz.discover_recent_runs(tmp_path, 1)

    assert runs[0].run_dir == completed.resolve()
    assert runs[0].best.path == runs[0].baseline.path
    assert runs[0].best_iteration is None


def test_generate_figures_writes_ppt_sized_svg_and_manifest(tmp_path: Path) -> None:
    run = viz.load_run(
        _write_run(
            tmp_path / "results",
            "run-20260822-200156-21529",
            baseline=100,
            best=50,
        )
    )
    output = tmp_path / "figures"

    generated = viz.generate_figures([run], output, view="all")

    expected = output / run.name / "baseline-vs-best.svg"
    assert expected in generated
    svg = expected.read_text(encoding="utf-8")
    assert 'viewBox="0 0 1920 1080"' in svg
    assert "Baseline · full-width timeline" in svg
    assert "Best（Iteration 2） · full-width timeline" in svg
    assert "Critical path" in svg
    assert "Parallel units  1 / 2 / 3+" in svg
    manifest = json.loads((output / "visualization-manifest.json").read_text())
    assert manifest["runs"][0]["baseline_cycles"] == 100
    assert manifest["runs"][0]["best_cycles"] == 50
    assert manifest["runs"][0]["best_iteration"] == 2


def test_explicit_candidate_generates_negative_optimization_evidence(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(
        tmp_path / "results",
        "run-20260817-162615-1077",
        baseline=100,
        best=110,
    )
    candidate_schedule_path = run_dir / "iteration-2" / "timing-schedule.json"
    candidate_data = json.loads(candidate_schedule_path.read_text(encoding="utf-8"))
    candidate_data["events"][0].update(
        {
            "op": "V_RD",
            "target_unit": "vector",
            "duration_cycles": 2,
        }
    )
    candidate_schedule_path.write_text(
        json.dumps(candidate_data),
        encoding="utf-8",
    )
    run = viz.load_run(run_dir, candidate_iteration=2)
    output = tmp_path / "figures"

    generated = viz.generate_figures([run], output, view="all")

    assert run.selection_kind == "candidate"
    assert run.improvement == pytest.approx(-0.1)
    assert run.best_label == "Candidate（Iteration 2 · 未晋升）"
    assert output / run.name / "parallelism-regression.svg" in generated
    assert output / run.name / "baseline-vs-candidate.svg" in generated
    assert output / run.name / "baseline-vs-candidate-diff.svg" in generated
    assert output / run.name / "candidate-detail.svg" in generated
    assert not (output / "recent-rtl-runs-summary.svg").exists()
    regression = (
        output / run.name / "parallelism-regression.svg"
    ).read_text(encoding="utf-8")
    assert "负优化归因" in regression
    assert "Net result" in regression
    diff = (
        output / run.name / "baseline-vs-candidate-diff.svg"
    ).read_text(encoding="utf-8")
    assert "仅显示差异" in diff
    assert "V_RD" in diff
    assert len(viz.changed_event_pairs(run.baseline, run.best)) == 1
    assert sum(viz.concurrency_distribution(run.baseline).values()) == 100

    windows_output = tmp_path / "windows"
    windows_generated = viz.generate_figures(
        [run],
        windows_output,
        view="windows",
        window_context_cycles=6,
    )
    window_path = (
        windows_output
        / run.name
        / "changed-windows"
        / "changed-window-01.svg"
    )
    assert window_path in windows_generated
    window_svg = window_path.read_text(encoding="utf-8")
    assert "修改指令附近的时间轴" in window_svg
    assert "Baseline · local context" in window_svg
    assert "Candidate · local context" in window_svg
    assert 'class="event-label"' in window_svg
    assert 'class="event-label-secondary"' in window_svg
    windows_manifest = json.loads(
        (
            windows_output
            / run.name
            / "changed-windows"
            / "changed-windows-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert windows_manifest["window_count"] == 1
    assert windows_manifest["context_cycles"] == 6
    window_record = windows_manifest["windows"][0]
    assert window_record["baseline_cycles"] == window_record["candidate_cycles"]
    assert window_record["shared_cycles"] == window_record["baseline_cycles"]
    assert "上下共享绝对 cycle 刻度" in window_svg


def test_event_box_label_uses_two_level_fallback_for_narrow_blocks() -> None:
    event = {
        "op": "V_WR_DRAM",
        "duration_cycles": 3,
        "tensor_writes": ["k_pos0_tile0"],
    }

    assert viz._event_box_label_for_width(event, 82) == ("DRAM WR", "3 cy")
    assert viz._event_box_label_for_width(event, 30) is None


def test_attention_start_finds_nearest_q_projection_setup() -> None:
    events = tuple(
        [
            {
                "raw_instruction_idx": raw,
                "expanded_idx": 0,
                "op": "S_WR",
                "start_cycle": 100 + raw,
                "finish_cycle": 101 + raw,
            }
            for raw in (10, 11, 12)
        ]
        + [
            {
                "raw_instruction_idx": 80,
                "expanded_idx": 0,
                "op": "V_RD_DRAM",
                "start_cycle": 200,
                "finish_cycle": 203,
            }
        ]
    )
    schedule = viz.Schedule(
        path=Path("timing-schedule.json"),
        events=events,
        metrics={},
        makespan=300,
        serial_cycles=0,
    )

    start = viz._find_attention_start_event(schedule, [events[-1]])

    assert start is not None
    assert start["raw_instruction_idx"] == 10
