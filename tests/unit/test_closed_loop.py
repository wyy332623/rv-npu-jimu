from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "closed_loop", ROOT / "jimu-dse" / "scripts" / "closed_loop.py"
)
closed_loop = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(closed_loop)


@pytest.fixture
def config():
    return closed_loop.load_config("dram-optimization")


def test_all_builtin_goals_validate():
    for goal in ("dram-optimization", "compute-optimization", "combined"):
        loaded = closed_loop.load_config(goal)
        assert loaded["schema_version"] == 1
        assert loaded["name"] == goal


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.update({"unexpected": True}), "unknown field"),
        (lambda c: c.update({"schema_version": 2}), "schema_version"),
        (lambda c: c["acceptance"]["score"][0].update({"weight": 0.1}), "sum to 1.0"),
        (lambda c: c["prompt"].update({"template": "{not_a_field}"}), "template field"),
        (
            lambda c: c["target"].update({"firmware": "../outside.c"}),
            "inside the repository",
        ),
    ],
)
def test_invalid_configs_are_rejected(config, mutate, message):
    candidate = copy.deepcopy(config)
    mutate(candidate)
    with pytest.raises(closed_loop.ConfigError, match=message):
        closed_loop.validate_config(candidate)


def test_duplicate_metric_is_rejected(config):
    candidate = copy.deepcopy(config)
    candidate["probe"]["metrics"].append(candidate["probe"]["metrics"][0])
    with pytest.raises(closed_loop.ConfigError, match="duplicates"):
        closed_loop.validate_config(candidate)


def test_prompt_is_stable_and_skill_ordered(config):
    first = closed_loop.render_prompt(config, 2, {"total_bytes": 10}, ["cluster A"])
    second = closed_loop.render_prompt(config, 2, {"total_bytes": 10}, ["cluster A"])
    assert first == second
    assert first.index("dag-analyze") < first.index("vrf-cache") < first.index("self-verify")
    assert "cluster A" in first
    assert "Iteration: 2" in first


def test_weighted_score_supports_both_directions():
    score, details = closed_loop.score_metrics(
        {"latency": 100, "quality": 10},
        {"latency": 80, "quality": 11},
        [
            {"metric": "latency", "direction": "minimize", "weight": 0.75},
            {"metric": "quality", "direction": "maximize", "weight": 0.25},
        ],
    )
    assert score == pytest.approx(0.175)
    assert details["latency"]["normalized"] == pytest.approx(0.2)
    assert details["quality"]["normalized"] == pytest.approx(0.1)


def test_zero_baseline_is_finite():
    score, _ = closed_loop.score_metrics(
        {"test_pass": 0}, {"test_pass": 1},
        [{"metric": "test_pass", "direction": "maximize", "weight": 1.0}],
    )
    assert score == 1.0


def test_cli_overrides_environment_last(config, monkeypatch):
    monkeypatch.setenv("JIMU_MAX_ITER", "9")
    monkeypatch.setenv("OPENCODE_MODEL", "env/model")
    result = closed_loop.resolved_config(config, agent="opencode", model="cli/model")
    assert result["loop"]["max_iterations"] == 9
    assert result["agent"]["backend"] == "opencode"
    assert result["agent"]["model"] == "cli/model"


def test_resume_fingerprint_is_stable(config):
    resolved = closed_loop.resolved_config(config)
    assert closed_loop.config_fingerprint(resolved) == closed_loop.config_fingerprint(
        copy.deepcopy(resolved)
    )
    changed = copy.deepcopy(resolved)
    changed["target"]["hardware"]["dim"] = 8
    assert closed_loop.config_fingerprint(resolved) != closed_loop.config_fingerprint(changed)


def test_gate_failure_is_reported(config, monkeypatch):
    monkeypatch.setattr(
        closed_loop, "build_firmware",
        lambda *_: {"passed": False, "exit_code": 1},
    )
    gates = closed_loop.run_gates(
        config, ["firmware/bert/bert_layer.c"], {"passed": True}
    )
    assert not next(x for x in gates if x["type"] == "build")["passed"]


def test_agent_unavailable(monkeypatch, config):
    monkeypatch.setattr(closed_loop.shutil, "which", lambda _: None)
    result = closed_loop.invoke_agent(config, "prompt")
    assert result["status"] == "agent_unavailable"


def test_loop_promotes_valid_improvement_and_restores_worktree(
    monkeypatch, config, tmp_path
):
    resolved = closed_loop.resolved_config(config)
    resolved["loop"].update({"max_iterations": 1, "target_score": None})
    target = ROOT / resolved["target"]["firmware"]
    original = target.read_bytes()
    calls = {"probe": 0}

    def probe(*_args, **_kwargs):
        calls["probe"] += 1
        value = 100 if calls["probe"] <= 2 else 80
        return {
            "passed": True,
            "metrics": {
                "total_bytes": value, "instr_count": 100,
                "mv_mul_count": 10, "mat_rd_ops": 10,
            },
            "clusters": [],
        }

    def agent(*_args):
        target.write_bytes(original + b"\n/* optimized by test */\n")
        return {"status": "completed", "exit_code": 0, "timed_out": False}

    monkeypatch.setattr(closed_loop, "probe_firmware", probe)
    monkeypatch.setattr(closed_loop, "invoke_agent", agent)
    monkeypatch.setattr(
        closed_loop, "run_gates",
        lambda *_: [{"name": "all", "type": "command", "passed": True}],
    )
    summary = closed_loop.execute_run(resolved, results_root=tmp_path)
    assert summary["status"] == "completed"
    assert summary["best_iteration"] == 1
    assert summary["iterations"][0]["promoted"]
    assert target.read_bytes() == original
    run_dir = ROOT / summary["run_dir"]
    if not run_dir.exists():  # custom results roots are absolute in this test
        run_dir = next(tmp_path.iterdir())
    assert (run_dir / "run-summary.json").is_file()
    assert (run_dir / "report.md").is_file()


def test_loop_stops_after_no_change(monkeypatch, config, tmp_path):
    resolved = closed_loop.resolved_config(config)
    resolved["loop"].update({
        "max_iterations": 5, "max_no_improvement": 1, "target_score": None
    })
    monkeypatch.setattr(
        closed_loop, "probe_firmware",
        lambda *_args, **_kwargs: {
            "passed": True,
            "metrics": {
                "total_bytes": 100, "instr_count": 100,
                "mv_mul_count": 10, "mat_rd_ops": 10,
            },
            "clusters": [],
        },
    )
    monkeypatch.setattr(
        closed_loop, "invoke_agent",
        lambda *_: {"status": "completed", "exit_code": 0, "timed_out": False},
    )
    summary = closed_loop.execute_run(resolved, results_root=tmp_path)
    assert summary["stop_reason"] == "no_improvement_limit"
    assert summary["iterations"][0]["status"] == "no_change"
