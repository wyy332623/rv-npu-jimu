from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

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
    for goal in (
        "dram-optimization", "compute-optimization", "combined",
        "cycle-latency-optimization",
        "rtl-cycle-optimization",
        "rtl-dram-optimization",
        "rtl-dram-exploration",
        "rtl-cycle-optimization-large",
    ):
        loaded = closed_loop.load_config(goal)
        assert loaded["schema_version"] == 1
        assert loaded["name"] == goal


def test_large_rtl_goal_uses_packed_dim16_contract():
    config = closed_loop.load_config("rtl-cycle-optimization-large")

    assert config["target"]["layout"] == "packed-v2"
    assert config["target"]["hardware"] == {
        "dim": 16, "hidden": 16, "num_head": 1,
    }
    assert config["probe"]["scoring_sequence_length"] == 16
    assert config["probe"]["workload_manifest"].endswith(
        "bert-dim16-h16-seq16.yaml"
    )
    assert config["probe"]["cycle_model"]["profile"].endswith(
        "jimu-rtl-dim16.yaml"
    )


def test_invalid_or_multihead_wide_packed_layout_is_rejected():
    config = closed_loop.load_config("rtl-cycle-optimization-large")
    invalid = copy.deepcopy(config)
    invalid["target"]["layout"] = "unknown-layout"
    with pytest.raises(closed_loop.ConfigError, match="target.layout"):
        closed_loop.validate_config(invalid)

    multihead = copy.deepcopy(config)
    multihead["target"]["hardware"]["num_head"] = 2
    with pytest.raises(closed_loop.ConfigError, match="num_head=1"):
        closed_loop.validate_config(multihead)


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
    assert "Never run `make clean` from the repository root" in first
    assert "Never delete, move, rename, or modify `jimu-dse/results`" in first
    assert "non-interactive optimization iteration" in first


def test_prompt_includes_rejected_candidate_feedback(config):
    prompt = closed_loop.render_prompt(
        config, 2, {"total_bytes": 90}, ["cluster"],
        attempt_history=[{
            "iteration": 1,
            "status": "not_improved",
            "promoted": False,
            "attempt_feedback": {
                "rtl_predicted_npu_cycles": {
                    "reference": 100, "candidate": 105, "delta": 5,
                },
                "modeled_dram_transaction_bytes": {
                    "reference": 64, "candidate": 48, "delta": -16,
                },
            },
        }],
    )

    assert "Previous candidate feedback" in prompt
    assert '"status": "not_improved"' in prompt
    assert '"rtl_predicted_npu_cycles"' in prompt
    assert "byte reductions are diagnostic" in prompt


def test_cycle_prompt_includes_iteration_work_contract():
    config = closed_loop.load_config("cycle-latency-optimization")
    prompt = closed_loop.render_prompt(
        config, 3, {"predicted_npu_cycles": 3137}, ["cluster A"]
    )

    assert "at most 1 primary optimization hypothesis" in prompt
    assert "By 600 seconds" in prompt
    assert "By 900 seconds" in prompt
    assert "By 1500 seconds" in prompt
    assert "Do not begin a second optimization stage" in prompt
    assert "closed-loop controller owns all configured gates and scoring" in prompt


def test_cycle_goal_scores_parallel_metric_and_renders_scheduler_context():
    config = closed_loop.load_config("cycle-latency-optimization")
    prompt = closed_loop.render_prompt(
        config,
        1,
        {
            "predicted_npu_cycles": 100,
            "parallel_predicted_npu_cycles": 75,
            "memory_compute_overlap_cycles": 25,
        },
        [],
    )

    assert config["acceptance"]["score"] == [{
        "metric": "parallel_predicted_npu_cycles",
        "direction": "minimize",
        "weight": 1.0,
    }]
    assert "single shared DRAM bus can overlap with independent compute" in prompt
    assert "parallel_predicted_npu_cycles" in prompt


def test_rtl_goal_scores_verilator_makespan_and_renders_hardware_context():
    config = closed_loop.load_config("rtl-cycle-optimization")
    prompt = closed_loop.render_prompt(
        config,
        1,
        {
            "rtl_predicted_npu_cycles": 3673,
            "memory_compute_overlap_cycles": 850,
            "rtl_counter_bank_stall_cycles": 76,
        },
        [],
    )

    assert config["acceptance"]["score"] == [{
        "metric": "rtl_predicted_npu_cycles",
        "direction": "minimize",
        "weight": 1.0,
    }]
    assert config["probe"]["cycle_model"]["profile"].endswith(
        "jimu-rtl-dim4.yaml"
    )
    assert "synthesizable Verilator RTL timing core" in prompt
    assert "128-bit semantic scoreboard" in prompt
    assert "rtl_predicted_npu_cycles" in prompt


def test_rtl_dram_goal_replays_only_historical_vrf_cache_strategy():
    config = closed_loop.load_config("rtl-dram-optimization")
    prompt = closed_loop.render_prompt(
        config,
        1,
        {
            "rtl_predicted_npu_cycles": 5000,
            "total_bytes": 10000,
            "memory_compute_overlap_cycles": 600,
        },
        ["K save/load cluster", "V save/load cluster"],
        graph_context="critical K/V DRAM events",
    )

    assert config["target"]["hardware"] == {
        "dim": 2, "hidden": 4, "num_head": 2,
    }
    assert [skill["name"] for skill in config["skills"]] == [
        "dag-analyze", "vrf-cache", "self-verify",
    ]
    assert config["acceptance"]["score"] == [{
        "metric": "rtl_predicted_npu_cycles",
        "direction": "minimize",
        "weight": 1.0,
    }]
    assert config["probe"]["cycle_model"]["profile"].endswith(
        "jimu-rtl-dim2.yaml"
    )
    assert config["probe"]["workload_manifest"].endswith(
        "bert-dim2-seq6.yaml"
    )
    assert config["loop"]["max_iterations"] == 1
    assert "only the historical dim=2 K/V VRF-cache transformation" in prompt
    assert "different optimization even if" in prompt


def test_rtl_timing_schedule_context_uses_rtl_resources(tmp_path):
    schedule_path = tmp_path / "timing-schedule.json"
    schedule_path.write_text(json.dumps({
        "backend": "verilator-rtl",
        "model": "test-rtl-profile",
        "profile": {"memory": {"minimum_transfer_bytes": 16}},
        "metrics": {
            "rtl_predicted_npu_cycles": 100,
            "parallel_predicted_npu_cycles": 100,
            "overlap_saved_cycles": 20,
            "memory_compute_overlap_cycles": 10,
            "logical_dram_payload_bytes": 8,
            "modeled_dram_transaction_bytes": 16,
        },
        "optimization_diagnostics": {
            "resource_bottlenecks": [
                {"resource": "vector", "utilization": 0.6},
            ],
            "top_blockers": [{
                "idx": 4, "op": "V_RD", "wait_cycles": 7,
                "reasons": ["bank"],
            }],
        },
    }), encoding="utf-8")

    context = closed_loop._timing_schedule_context(schedule_path)

    assert "RTL resource mapping" in context
    assert "128-bit RTL scoreboard" in context
    assert "timing_profile_name=test-rtl-profile" in context
    assert "timing_profile_sha256=" in context
    assert "rtl_cycles=100" in context
    assert "logical_dram_payload_bytes=8" in context
    assert "modeled_dram_transaction_bytes=16" in context
    assert "wait=7 cycles, reasons=bank" in context


def test_invalid_rtl_cycle_profile_is_rejected(monkeypatch):
    config = closed_loop.load_config("rtl-cycle-optimization")
    profile_path = ROOT / "jimu-dse" / "timing" / "jimu-rtl-dim4.yaml"
    original_read = closed_loop._read_yaml
    profile = copy.deepcopy(original_read(profile_path))
    profile["rtl"]["resource_bits"] = 64

    monkeypatch.setattr(
        closed_loop,
        "_read_yaml",
        lambda path: profile if path == profile_path.resolve() else original_read(path),
    )
    with pytest.raises(closed_loop.ConfigError, match="requires resource_bits=128"):
        closed_loop.validate_config(config)


def test_timing_schedule_context_is_actionable_and_bounded(tmp_path):
    schedule_path = tmp_path / "timing-schedule.json"
    schedule_path.write_text(json.dumps({
        "metrics": {
            "parallel_predicted_npu_cycles": 75,
            "overlap_saved_cycles": 25,
            "memory_compute_overlap_cycles": 20,
        },
        "optimization_diagnostics": {
            "resource_bottlenecks": [
                {"resource": "vmm", "utilization": 0.8},
                {"resource": "dram_bus", "utilization": 0.5},
            ],
            "critical_event_wait_cycles_by_reason": {
                "dependency:RAW": 12,
            },
            "critical_path_top_events": [{
                "idx": 9,
                "raw_instruction_idx": 7,
                "expanded_idx": 0,
                "op": "MV_MUL",
                "start_cycle": 10,
                "end_cycle": 18,
                "duration_cycles": 8,
                "resources": ["mvu"],
                "blocking_reasons": ["dependency:RAW"],
            }],
            "critical_path_top_blockers": [{
                "idx": 9,
                "op": "MV_MUL",
                "wait_cycles": 10,
                "blocking_reasons": ["dependency:RAW"],
            }],
            "source_mapping_note": "direct source mapping unavailable",
        },
    }), encoding="utf-8")

    context = closed_loop._timing_schedule_context(schedule_path)

    assert f"timing_schedule_file={schedule_path.resolve()}" in context
    assert "vector DRAM=dram_bus+vmm" in context
    assert "All DRAM transfers serialize on dram_bus" in context
    assert "resource_utilization_rank=vmm=80.0%, dram_bus=50.0%" in context
    assert "critical_event_wait_cycles_by_reason=dependency:RAW:12" in context
    assert "idx=9, raw=7" in context
    assert "wait=10 cycles" in context


def test_parallel_prompt_metric_view_prioritizes_score_and_baseline_delta():
    config = closed_loop.load_config("cycle-latency-optimization")
    metrics = {
        "parallel_predicted_npu_cycles": 80,
        "predicted_npu_cycles": 100,
        "overlap_saved_cycles": 20,
        "dram_bus_utilization": 0.5,
        "scalesim_compute_cycles": 40,
        "trace_memory_cycles": 50,
        "auxiliary_cycles": 10,
        "scalesim_layer_count": 12,
        "scalesim_stall_cycles": 0,
        "schedule_chain_count": 1,
    }

    view = closed_loop._prompt_metric_view(
        config, metrics, {"parallel_predicted_npu_cycles": 100}
    )

    assert view["score"] == {"parallel_predicted_npu_cycles": 80}
    comparison = view["comparison_to_run_baseline"][
        "parallel_predicted_npu_cycles"
    ]
    assert comparison == {
        "run_baseline": 100,
        "current_validated_best": 80,
        "absolute_delta": -20.0,
        "improvement_fraction": pytest.approx(0.2),
    }
    assert view["optimization_diagnostics"]["dram_bus_utilization"] == 0.5
    assert view["legacy_model_breakdown"]["trace_memory_cycles"] == 50
    assert "scalesim_layer_count" not in json.dumps(view)
    assert "schedule_chain_count" not in json.dumps(view)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda p: p["scheduler"].update({"queue_depth": 0}),
            "queue_depth must be a positive integer",
        ),
        (
            lambda p: p["scheduler"]["resources"].update({"dram_bus": 0}),
            "resources.dram_bus must be positive",
        ),
        (
            lambda p: p.update({"backend": "scalesim"}),
            "requires backend scalesim-parallel",
        ),
        (
            lambda p: p["scheduler"].update({"cross_chain_overlap": True}),
            "cross_chain_overlap must be false",
        ),
    ],
)
def test_invalid_parallel_cycle_profiles_are_rejected(
    monkeypatch, mutation, message
):
    config = closed_loop.load_config("cycle-latency-optimization")
    profile_path = ROOT / "jimu-dse" / "timing" / "scalesim-parallel-dim4.yaml"
    original_read = closed_loop._read_yaml
    profile = original_read(profile_path)
    mutation(profile)

    monkeypatch.setattr(
        closed_loop,
        "_read_yaml",
        lambda path: profile if path == profile_path.resolve() else original_read(path),
    )
    with pytest.raises(closed_loop.ConfigError, match=message):
        closed_loop.validate_config(config)


def test_invalid_agent_work_budget_is_rejected():
    config = closed_loop.load_config("cycle-latency-optimization")
    config["agent"]["work_budget"]["return_deadline_seconds"] = 1800

    with pytest.raises(closed_loop.ConfigError, match="strictly ordered"):
        closed_loop.validate_config(config)


def test_zero_agent_timeout_disables_hard_limit_and_cli_budget():
    config = closed_loop.load_config("cycle-latency-optimization")
    resolved = closed_loop.resolved_config(config, agent_timeout=0)

    assert resolved["agent"]["timeout_seconds"] == 0
    assert "work_budget" not in resolved["agent"]


def test_run_report_includes_parallel_schedule_metrics():
    common = {
        "predicted_npu_cycles": 100,
        "parallel_predicted_npu_cycles": 80,
        "overlap_saved_cycles": 20,
        "memory_compute_overlap_cycles": 20,
        "max_concurrent_ops": 2,
        "dram_bus_utilization": 0.5,
        "mvu_utilization": 0.4,
        "vmm_utilization": 0.3,
        "mmm_utilization": 0.2,
        "spu_utilization": 0.1,
    }
    report = closed_loop._summary_report({
        "goal": "cycle-latency-optimization",
        "status": "completed",
        "stop_reason": "max_iterations",
        "run_dir": "jimu-dse/results/fake",
        "iterations": [],
        "next_iteration": 1,
        "best_iteration": None,
        "best_score": 0.0,
        "baseline_metrics": common,
        "best_metrics": {**common, "parallel_predicted_npu_cycles": 72},
        "reproduce_command": "python closed_loop.py inspect-run fake",
    })

    assert "Parallel resource schedule" in report
    assert "| Parallel predicted cycles | 80 | 72 |" in report
    assert "| Memory/compute overlap cycles | 20 | 20 |" in report
    assert "| DRAM bus utilization | 50.00% | 50.00% |" in report


def test_completed_report_marks_old_interruption_as_recovered():
    report = closed_loop._summary_report({
        "goal": "rtl-cycle-optimization",
        "status": "completed",
        "stop_reason": "max_iterations",
        "run_dir": "jimu-dse/results/fake",
        "iterations": [],
        "next_iteration": 2,
        "best_iteration": 1,
        "best_score": 0.1,
        "baseline_metrics": {},
        "best_metrics": {},
        "interruptions": [{"reason": "executable_not_found", "iteration": 1}],
        "reproduce_command": "python closed_loop.py inspect-run fake",
    })

    assert "## Earlier interruption (recovered)" in report
    assert "## Latest interruption" not in report


def test_root_clean_preserves_run_results():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    clean_recipe = makefile.split("\nclean:\n", 1)[1].split("\n\n", 1)[0]
    clean_results_recipe = makefile.split("\nclean-results:\n", 1)[1].split(
        "\n\n", 1
    )[0]
    assert "jimu-dse/results" not in clean_recipe
    assert "jimu-dse/results" in clean_results_recipe


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
    monkeypatch.setenv("JIMU_AGENT_TIMEOUT", "120")
    monkeypatch.setenv("OPENCODE_MODEL", "env/model")
    result = closed_loop.resolved_config(
        config, agent="opencode", model="cli/model",
        max_iterations=10, agent_timeout=0,
    )
    assert result["loop"]["max_iterations"] == 10
    assert result["agent"]["backend"] == "opencode"
    assert result["agent"]["model"] == "cli/model"
    assert result["agent"]["timeout_seconds"] == 0


def test_run_parser_accepts_explicit_iteration_and_timeout_overrides():
    args = closed_loop.build_parser().parse_args([
        "run", "--goal", "cycle-latency-optimization",
        "--max-iterations", "10", "--agent-timeout", "0",
        "--full-iterations",
    ])

    assert args.max_iterations == 10
    assert args.agent_timeout == 0
    assert args.full_iterations is True


def test_resume_fingerprint_is_stable(config):
    resolved = closed_loop.resolved_config(config)
    assert closed_loop.config_fingerprint(resolved) == closed_loop.config_fingerprint(
        copy.deepcopy(resolved)
    )
    changed = copy.deepcopy(resolved)
    changed["target"]["hardware"]["dim"] = 8
    assert closed_loop.config_fingerprint(resolved) != closed_loop.config_fingerprint(changed)


def test_timing_profile_fingerprint_tracks_file_contents(
    monkeypatch, tmp_path
):
    config = closed_loop.load_config("rtl-dram-exploration")
    profile = tmp_path / "profile.yaml"
    profile.write_text("name: first\n", encoding="utf-8")
    monkeypatch.setattr(closed_loop, "_repo_path", lambda *_args: profile)

    first = closed_loop.timing_profile_fingerprints(config)
    profile.write_text("name: second\n", encoding="utf-8")
    second = closed_loop.timing_profile_fingerprints(config)

    assert first["cycle_model"]["sha256"] != second["cycle_model"]["sha256"]


def test_candidate_feedback_exposes_parallelism_and_transaction_tradeoff():
    feedback = closed_loop._candidate_feedback(
        {
            "rtl_predicted_npu_cycles": 100,
            "logical_dram_payload_bytes": 64,
            "modeled_dram_transaction_bytes": 96,
            "memory_compute_overlap_cycles": 20,
        },
        {
            "rtl_predicted_npu_cycles": 104,
            "logical_dram_payload_bytes": 48,
            "modeled_dram_transaction_bytes": 80,
            "memory_compute_overlap_cycles": 8,
        },
    )

    assert feedback["rtl_predicted_npu_cycles"]["delta"] == 4
    assert feedback["modeled_dram_transaction_bytes"]["delta"] == -16
    assert feedback["memory_compute_overlap_cycles"]["delta"] == -12


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
    monkeypatch.setattr(closed_loop.shutil, "which", lambda *_args, **_kwargs: None)
    result = closed_loop.invoke_agent(config, "prompt")
    assert result["status"] == "agent_unavailable"


def test_timeout_output_bytes_are_decoded(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["opencode"],
            timeout=1,
            output=b"partial \xff stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(closed_loop.subprocess, "run", timeout)
    result = closed_loop._run(["opencode"], timeout=1)

    assert result["timed_out"]
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
    assert "\ufffd" in result["stdout"]
    json.dumps(result)


def test_run_heartbeat_supports_disabled_timeout():
    beats = []
    result = closed_loop._run(
        [closed_loop.sys.executable, "-c", "import time; time.sleep(0.12)"],
        timeout=None,
        heartbeat=beats.append,
        heartbeat_seconds=0.03,
    )

    assert result["exit_code"] == 0
    assert not result["timed_out"]
    assert beats


def test_run_preserves_full_raw_output_while_bounding_summary(tmp_path):
    prefix = tmp_path / "agent-1"
    result = closed_loop._run(
        [closed_loop.sys.executable, "-c", "print('x' * 25000)"],
        timeout=10,
        raw_output_prefix=prefix,
    )

    raw = (tmp_path / "agent-1.stdout.jsonl").read_text(encoding="utf-8")
    assert len(raw) > 25000
    assert "...[output truncated]..." in result["stdout"]
    assert len(result["stdout"]) <= 20000
    assert (tmp_path / "agent-1.stderr.log").is_file()


def test_project_environment_prefers_local_virtualenv(tmp_path, monkeypatch):
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setattr(closed_loop, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("PATH", "system-bin")

    env = closed_loop._project_environment()

    assert env["PATH"].split(closed_loop.os.pathsep)[0] == str(venv_bin)


def test_agent_lookup_uses_project_environment_path(monkeypatch, config):
    observed = {}
    monkeypatch.setattr(
        closed_loop, "_agent_environment",
        lambda: {"PATH": "/agent-bin:/system-bin"},
    )

    def which(name, path=None):
        observed.update({"name": name, "path": path})
        return None

    monkeypatch.setattr(closed_loop.shutil, "which", which)

    result = closed_loop.invoke_agent(config, "prompt")

    assert result["status"] == "agent_unavailable"
    assert observed == {
        "name": config["agent"]["backend"],
        "path": "/agent-bin:/system-bin",
    }


def test_windows_gitfile_is_resolved_for_wsl():
    resolved = closed_loop._resolve_worktree_git_dir(
        r"D:\repo\.git\worktrees\candidate", platform="posix"
    )

    assert resolved == "/mnt/d/repo/.git/worktrees/candidate"


def test_default_heartbeat_interval_is_twenty_minutes(monkeypatch):
    observed = {}

    class FinishedProcess:
        returncode = 0

        def communicate(self, timeout):
            observed["timeout"] = timeout
            return "", ""

        def kill(self):
            pass

    monkeypatch.setattr(
        closed_loop.subprocess, "Popen", lambda *_args, **_kwargs: FinishedProcess()
    )

    result = closed_loop._run(
        ["agent"], timeout=None, heartbeat=lambda _elapsed: None
    )

    assert result["exit_code"] == 0
    assert observed["timeout"] == 20 * 60


def test_write_json_has_defensive_bytes_fallback(tmp_path):
    output = tmp_path / "nested.json"
    closed_loop._write_json(
        output,
        {"agent": {"stdout": b"partial \xff output"}, "path": output},
    )
    decoded = json.loads(output.read_text(encoding="utf-8"))

    assert isinstance(decoded["agent"]["stdout"], str)
    assert "\ufffd" in decoded["agent"]["stdout"]
    assert decoded["path"] == str(output)


def test_weighted_cost_counts_only_selected_npu_resources():
    events = [
        {
            "uses": [
                ("VRF", 1, 0), ("MRF",), ("SRF", 2), ("REG", 3),
                ("pipe",), ("vpipe_a",), ("DRAM", 100),
            ],
            "defs": [
                ("VRF", 2, 0), ("MRF",), ("SRF", 4), ("REG", 5),
                ("pipe",), ("DRAM", 200),
            ],
        },
        {"uses": [("VRF", 1, 1)], "defs": [("REG", 6)]},
    ]
    dram = {
        "vec_rd_ops": 8, "mat_rd_ops": 4,
        "vec_wr_ops": 5, "mat_wr_ops": 3,
    }
    result = closed_loop.calculate_cost_metrics(
        events, dram,
        {
            "memory_weight": 10,
            "register_weight": 1,
            "register_resources": ["VRF", "MRF", "SRF", "REG"],
        },
    )
    assert result == {
        "memory_access_count": 20,
        "memory_read_count": 12,
        "memory_write_count": 8,
        "register_access_count": 10,
        "register_read_count": 5,
        "register_write_count": 5,
        "estimated_time": 210.0,
    }


def test_weighted_cost_formula_example():
    events = [{"uses": [("VRF",)] * 40, "defs": [("SRF",)] * 40}]
    result = closed_loop.calculate_cost_metrics(
        events,
        {"vec_rd_ops": 10, "mat_rd_ops": 5, "vec_wr_ops": 3, "mat_wr_ops": 2},
        {
            "memory_weight": 10,
            "register_weight": 1,
            "register_resources": ["VRF", "MRF", "SRF", "REG"],
        },
    )
    assert result["memory_access_count"] == 20
    assert result["register_access_count"] == 80
    assert result["estimated_time"] == 280


def test_composite_cost_can_trade_memory_for_register_accesses():
    score, details = closed_loop.score_metrics(
        {"estimated_time": 280},
        {"estimated_time": 275},  # one less DRAM op, five more register ops
        [{
            "metric": "estimated_time", "direction": "minimize", "weight": 1.0
        }],
    )
    assert score > 0
    assert details["estimated_time"]["normalized"] == pytest.approx(5 / 280)


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


def test_loop_stops_after_no_change(monkeypatch, config, tmp_path, capsys):
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
    assert "| 1 | no_change | SKIPPED |" in closed_loop._summary_report(summary)
    progress = capsys.readouterr().err
    assert "run started: goal=dram-optimization" in progress
    assert "baseline probe passed" in progress
    assert "iteration 1/5: agent started" in progress
    assert "status=no_change" in progress
    assert "stop_reason=no_improvement_limit" in progress


def test_full_iterations_ignores_target_and_no_improvement(
    monkeypatch, config, tmp_path
):
    resolved = closed_loop.resolved_config(config, full_iterations=True)
    resolved["loop"].update({
        "max_iterations": 3,
        "max_no_improvement": 1,
        "target_score": 0.1,
    })
    target = ROOT / resolved["target"]["firmware"]
    original = target.read_bytes()
    calls = {"probe": 0, "agent": 0}

    def probe(*_args, **_kwargs):
        calls["probe"] += 1
        value = 80 if calls["probe"] == 3 else 100
        return {
            "passed": True,
            "metrics": {
                "total_bytes": value, "instr_count": 100,
                "mv_mul_count": 10, "mat_rd_ops": 10,
            },
            "clusters": [],
            "graph_context": "test graph",
        }

    def agent(*_args):
        calls["agent"] += 1
        if calls["agent"] == 1:
            target.write_bytes(original + b"\n/* improved */\n")
        return {
            "status": "completed", "exit_code": 0, "timed_out": False,
            "agent_started": True,
        }

    monkeypatch.setattr(closed_loop, "probe_firmware", probe)
    monkeypatch.setattr(closed_loop, "invoke_agent", agent)
    monkeypatch.setattr(
        closed_loop, "run_gates",
        lambda *_: [{"name": "all", "type": "command", "passed": True}],
    )

    summary = closed_loop.execute_run(resolved, results_root=tmp_path)

    assert summary["status"] == "completed"
    assert summary["stop_reason"] == "max_iterations"
    assert len(summary["iterations"]) == 3
    assert summary["best_score"] == pytest.approx(0.16)


def test_timeout_candidate_is_gated_scored_and_promoted(
    monkeypatch, config, tmp_path
):
    resolved = closed_loop.resolved_config(config)
    resolved["loop"].update({
        "max_iterations": 1,
        "target_score": None,
        "evaluate_timeout_candidate": True,
    })
    target = ROOT / resolved["target"]["firmware"]
    original = target.read_bytes()
    calls = {"probe": 0, "gates": 0}

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
            "graph_context": "test graph",
        }

    def agent(*_args):
        target.write_bytes(original + b"\n/* valid timeout candidate */\n")
        return {
            "status": "agent_timeout", "exit_code": None,
            "stdout": "", "stderr": "", "timed_out": True,
            "spawn_error": False, "agent_started": True,
        }

    def gates(*_args):
        calls["gates"] += 1
        return [{"name": "all", "type": "command", "passed": True}]

    monkeypatch.setattr(closed_loop, "probe_firmware", probe)
    monkeypatch.setattr(closed_loop, "invoke_agent", agent)
    monkeypatch.setattr(closed_loop, "run_gates", gates)

    summary = closed_loop.execute_run(resolved, results_root=tmp_path)
    record = summary["iterations"][0]
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())

    assert calls["gates"] == 1
    assert record["timeout_candidate_evaluated"] is True
    assert record["status"] == "accepted_after_timeout"
    assert record["promoted"] is True
    assert summary["best_score"] == pytest.approx(0.16)
    assert (run_dir / "candidate-timeout-1.c").is_file()
    assert target.read_bytes() == original


def test_timeout_candidate_failing_gates_is_not_promoted(
    monkeypatch, config, tmp_path
):
    resolved = closed_loop.resolved_config(config)
    resolved["loop"].update({
        "max_iterations": 1,
        "target_score": None,
        "evaluate_timeout_candidate": True,
    })
    target = ROOT / resolved["target"]["firmware"]
    original = target.read_bytes()
    calls = {"probe": 0}

    def probe(*_args, **_kwargs):
        calls["probe"] += 1
        value = 100 if calls["probe"] <= 2 else 70
        return {
            "passed": True,
            "metrics": {
                "total_bytes": value, "instr_count": 100,
                "mv_mul_count": 10, "mat_rd_ops": 10,
            },
            "clusters": [],
            "graph_context": "test graph",
        }

    def agent(*_args):
        target.write_bytes(original + b"\n/* invalid timeout candidate */\n")
        return {
            "status": "agent_timeout", "exit_code": None,
            "stdout": "", "stderr": "", "timed_out": True,
            "spawn_error": False, "agent_started": True,
        }

    monkeypatch.setattr(closed_loop, "probe_firmware", probe)
    monkeypatch.setattr(closed_loop, "invoke_agent", agent)
    monkeypatch.setattr(
        closed_loop, "run_gates",
        lambda *_: [{"name": "correctness", "type": "command", "passed": False}],
    )

    summary = closed_loop.execute_run(resolved, results_root=tmp_path)
    record = summary["iterations"][0]

    assert record["timeout_candidate_evaluated"] is True
    assert record["status"] == "timeout_candidate_gate_failed"
    assert record["promoted"] is False
    assert summary["best_score"] == 0.0
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    "message, reason",
    [
        ("HTTP 429 Too Many Requests", "quota_exceeded"),
        ("quota exceeded for this account", "quota_exceeded"),
        ("authentication failed: invalid api key", "authentication_failed"),
        ("unknown model foo/bar", "provider_or_model_error"),
        ("configuration error: invalid provider", "provider_or_model_error"),
    ],
)
def test_agent_start_failure_signatures(message, reason):
    result = {
        "status": "agent_failed",
        "exit_code": 1,
        "stdout": "",
        "stderr": message,
        "timed_out": False,
        "spawn_error": False,
        "agent_started": False,
    }
    assert closed_loop.classify_agent_start_failure(result, []) == reason


def test_started_agent_runtime_failure_is_not_start_failure():
    result = {
        "status": "agent_failed",
        "exit_code": 1,
        "stdout": '{"type":"tool.completed"}',
        "stderr": "tool failed",
        "timed_out": False,
        "spawn_error": False,
        "agent_started": True,
    }
    assert closed_loop.classify_agent_start_failure(
        result, ["firmware/bert/bert_layer.c"]
    ) is None


def test_start_failure_checkpoint_resumes_same_iteration(
    monkeypatch, config, tmp_path
):
    resolved = closed_loop.resolved_config(config, full_iterations=True)
    resolved["loop"].update({"max_iterations": 1, "target_score": 0.1})
    target = ROOT / resolved["target"]["firmware"]
    original = target.read_bytes()

    monkeypatch.setattr(
        closed_loop, "probe_firmware",
        lambda *_args, **_kwargs: {
            "passed": True,
            "metrics": {
                "total_bytes": 100, "instr_count": 100,
                "mv_mul_count": 10, "mat_rd_ops": 10,
            },
            "clusters": [],
            "graph_context": "test graph",
        },
    )
    monkeypatch.setattr(
        closed_loop,
        "invoke_agent",
        lambda *_: {
            "status": "agent_start_failed",
            "exit_code": 1,
            "stdout": "",
            "stderr": "HTTP 429 quota exceeded",
            "timed_out": False,
            "spawn_error": False,
            "agent_started": False,
            "failure_reason": "quota_exceeded",
        },
    )

    interrupted = closed_loop.execute_run(resolved, results_root=tmp_path)
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())

    assert interrupted["status"] == "interrupted"
    assert interrupted["stop_reason"] == "agent_start_failed"
    assert interrupted["next_iteration"] == 1
    assert interrupted["iterations"] == []
    assert (run_dir / "candidate_best.c").read_bytes() == original

    monkeypatch.setattr(
        closed_loop,
        "invoke_agent",
        lambda *_: {
            "status": "completed", "exit_code": 0, "stdout": "", "stderr": "",
            "timed_out": False, "spawn_error": False, "agent_started": True,
        },
    )
    resumed = closed_loop.execute_run(resolved, resume=str(run_dir))

    assert resumed["status"] == "completed"
    assert resumed["stop_reason"] == "max_iterations"
    assert len(resumed["iterations"]) == 1
    assert resumed["iterations"][0]["iteration"] == 1
    assert len(resumed["interruptions"]) == 1
    assert target.read_bytes() == original


def test_probe_passes_scoring_sequence_to_graph(monkeypatch, tmp_path):
    config = closed_loop.resolved_config(
        closed_loop.load_config("cycle-latency-optimization")
    )
    monkeypatch.setattr(
        closed_loop, "build_firmware",
        lambda *_: {"passed": True, "elf": "firmware/fake.elf"},
    )
    graph_commands = []

    def fake_run(command, *_args, **_kwargs):
        if "jimu-dse/scripts/visualize_graph.py" in command:
            graph_commands.append(command)
            output_dir = Path(command[command.index("-o") + 1])
            output_dir.mkdir(parents=True)
            (output_dir / "dram_clusters.txt").write_text(
                "cluster evidence", encoding="utf-8"
            )
            return {
                "exit_code": 0,
                "stdout": "Building firmware: dim=4, hidden=4, seq_len=6\n",
                "stderr": "",
                "timed_out": False,
                "spawn_error": False,
            }
        metrics = {name: 1 for name in config["probe"]["metrics"]}
        return {
            "exit_code": 0,
            "stdout": json.dumps(metrics),
            "stderr": "",
            "timed_out": False,
            "spawn_error": False,
        }

    monkeypatch.setattr(closed_loop, "_run", fake_run)
    (tmp_path / "timing-schedule.json").write_text(json.dumps({
        "metrics": {
            "parallel_predicted_npu_cycles": 10,
            "overlap_saved_cycles": 2,
            "memory_compute_overlap_cycles": 2,
        },
        "optimization_diagnostics": {
            "resource_bottlenecks": [],
            "critical_path_top_events": [],
        },
    }), encoding="utf-8")
    probe = closed_loop.probe_firmware(config, 6, tmp_path)

    command = graph_commands[0]
    assert command[command.index("--seq-len") + 1] == "6"
    assert probe["graph_config_matches"]
    assert "seq_len=6" in probe["graph_context"]
    assert "timing_schedule_file=" in probe["graph_context"]
    assert "parallel_cycles=10" in probe["graph_context"]


def test_cycle_goal_enables_unified_timed_and_tensor_evidence():
    config = closed_loop.load_config("cycle-latency-optimization")

    assert config["probe"]["timed_device"]["profile"].endswith(
        "npu-timed-v1.yaml"
    )
    assert config["probe"]["workload_manifest"].endswith(
        "bert-dim4-seq6.yaml"
    )
    assert "timed_wall_cycles" in config["probe"]["metrics"]


def test_generic_target_accepts_declarative_build_command(config):
    candidate = copy.deepcopy(config)
    candidate["target"]["build"] = {
        "command": [
            "riscv64-unknown-elf-gcc", "{firmware}", "-o", "{elf}",
        ],
        "elf": "firmware/build_custom/custom.elf",
        "cwd": ".",
        "environment": {"NATIVE_DIM": "{dim}"},
    }

    closed_loop.validate_config(candidate)

    candidate["target"]["build"]["command"] = "make firmware"
    with pytest.raises(closed_loop.ConfigError, match="non-empty string list"):
        closed_loop.validate_config(candidate)


def test_dataflow_skill_allows_parallel_reordering():
    skill = (
        ROOT / "jimu-dse" / "docs" / "skills" / "isa"
        / "dataflow-optimize.md"
    ).read_text(encoding="utf-8")

    assert "a legal reordering may improve the score" in skill
    assert "do not claim a benefit for pure reordering" not in skill


def test_dataflow_experiment_config_validates():
    experiment = closed_loop.load_experiment_config(
        str(
            ROOT
            / "jimu-dse"
            / "experiments"
            / "dataflow-skill"
            / "experiment.yaml"
        )
    )
    assert experiment["full_iterations"] is True
    assert list(experiment["arms"]) == ["control", "legacy", "treatment"]


def test_skill_evaluation_interleaves_arms_and_aggregates(
    monkeypatch, tmp_path
):
    experiment = closed_loop.load_experiment_config(
        str(
            ROOT
            / "jimu-dse"
            / "experiments"
            / "dataflow-skill"
            / "experiment.yaml"
        )
    )
    monkeypatch.setattr(closed_loop, "RESULTS_DIR", tmp_path)
    calls = []

    def fake_execute(config, **_kwargs):
        names = [item["name"] for item in config["skills"]]
        arm = (
            "treatment" if "dataflow-optimize" in names
            else "legacy" if "vrf-cache" in names
            else "control"
        )
        calls.append(arm)
        score = {"control": 0.05, "legacy": 0.09, "treatment": 0.12}[arm]
        return {
            "status": "completed",
            "run_dir": f"fake/{arm}-{len(calls)}",
            "best_score": score,
            "best_metrics": {"predicted_npu_cycles": 1000 * (1 - score)},
            "iterations": [{
                "iteration": 1,
                "status": "accepted",
                "gates_passed": True,
                "promoted": True,
                "agent": {
                    "stdout": (
                        "<dataflow_hypotheses>[]</dataflow_hypotheses>"
                        if arm == "treatment" else ""
                    )
                },
            }],
            "interruptions": [],
        }

    monkeypatch.setattr(closed_loop, "execute_run", fake_execute)
    summary = closed_loop.evaluate_skill(experiment, repetitions=2)

    assert calls == [
        "control", "legacy", "treatment",
        "control", "legacy", "treatment",
    ]
    assert summary["status"] == "completed"
    assert summary["statistics"]["treatment"]["target_hits"] == 2
    assert summary["statistics"]["treatment"]["median_score"] == pytest.approx(0.12)
    assert summary["statistics"]["treatment"]["hypothesis_adherence_rate"] == 1.0
    assert summary["acceptance"]["passed"]
    experiment_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert (experiment_dir / "summary.json").is_file()
    assert (experiment_dir / "report.md").is_file()


def test_skill_evaluation_stops_on_agent_start_failure(monkeypatch, tmp_path):
    experiment = closed_loop.load_experiment_config(
        str(
            ROOT
            / "jimu-dse"
            / "experiments"
            / "dataflow-skill"
            / "experiment.yaml"
        )
    )
    monkeypatch.setattr(closed_loop, "RESULTS_DIR", tmp_path)
    calls = []

    def interrupted(config, **_kwargs):
        calls.append([item["name"] for item in config["skills"]])
        return {
            "status": "interrupted",
            "run_dir": "fake/interrupted",
            "best_score": 0.0,
            "best_metrics": {},
            "iterations": [],
            "interruptions": [{"reason": "quota_exceeded"}],
        }

    monkeypatch.setattr(closed_loop, "execute_run", interrupted)
    summary = closed_loop.evaluate_skill(experiment, repetitions=3)

    assert summary["status"] == "interrupted"
    assert summary["pending"]["replicate"] == 1
    assert summary["pending"]["arm"] == "control"
    assert len(calls) == 1


def test_deleted_run_directory_preserves_best_candidate(
    monkeypatch, config, tmp_path
):
    resolved = closed_loop.resolved_config(config)
    resolved["loop"].update({"max_iterations": 1, "target_score": None})
    target = ROOT / resolved["target"]["firmware"]
    original = target.read_bytes()

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

    def deleting_agent(*_args):
        target.write_bytes(original + b"\n/* unvalidated candidate */\n")
        run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
        shutil.rmtree(run_dir)
        return {"status": "completed", "exit_code": 0, "timed_out": False}

    monkeypatch.setattr(closed_loop, "invoke_agent", deleting_agent)
    summary = closed_loop.execute_run(resolved, results_root=tmp_path)
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())

    assert summary["status"] == "failed"
    assert summary["stop_reason"] == "run_artifacts_lost"
    assert summary["iterations"][0]["status"] == "infrastructure_error"
    assert (run_dir / "candidate_best.c").read_bytes() == original
    assert (run_dir / "resolved-config.yaml").is_file()
    assert (run_dir / "baseline-probe.json").is_file()
    assert (run_dir / "artifact-recovery.json").is_file()
    assert (run_dir / "run-summary.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert target.read_bytes() == original
