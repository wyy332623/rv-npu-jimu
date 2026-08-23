#!/usr/bin/env python3
"""Configurable closed-loop firmware optimization driver."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by CLI environments
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from emulator.bert_layout import (  # noqa: E402
    LEGACY_LAYOUT,
    PACKED_LAYOUT,
    SUPPORTED_LAYOUTS,
    bert_dram_layout,
)
GOALS_DIR = REPO_ROOT / "jimu-dse" / "goals"
RESULTS_DIR = REPO_ROOT / "jimu-dse" / "results"
SUPPORTED_METRICS = {
    "total_bytes", "instr_count", "mv_mul_count", "mat_rd_ops", "test_pass",
    "dram_elements", "functional_container_bytes", "rtl_payload_bytes",
    "memory_access_count", "memory_read_count", "memory_write_count",
    "register_access_count", "register_read_count", "register_write_count",
    "estimated_time",
    "scalesim_layer_count", "scalesim_compute_cycles",
    "scalesim_stall_cycles", "trace_memory_cycles", "auxiliary_cycles",
    "predicted_npu_cycles",
    "parallel_predicted_npu_cycles", "overlap_saved_cycles",
    "net_parallelism_savings_cycles", "gross_overlap_cycles",
    "scheduler_idle_hole_cycles", "rtl_completion_makespan_cycles",
    "rtl_idle_cycles", "rtl_retirement_tail_cycles",
    "memory_compute_overlap_cycles", "max_concurrent_ops",
    "dram_bus_utilization", "mvu_utilization", "vmm_utilization",
    "mmm_utilization", "spu_utilization", "schedule_chain_count",
    "rtl_predicted_npu_cycles", "serial_command_cycles",
    "memory_compute_overlap_ratio", "load_utilization", "store_utilization",
    "vector_utilization", "control_utilization",
    "logical_dram_payload_bytes", "modeled_dram_transaction_bytes",
    "rtl_counter_cycles", "rtl_counter_active_cycles",
    "rtl_counter_memory_compute_overlap_cycles",
    "rtl_counter_frontend_full_stall_cycles",
    "rtl_counter_dependency_stall_cycles", "rtl_counter_unit_stall_cycles",
    "rtl_counter_dram_stall_cycles", "rtl_counter_bank_stall_cycles",
    "rtl_counter_barrier_stall_cycles", "rtl_counter_dispatches",
    "rtl_counter_completions", "rtl_counter_max_inflight",
    "timed_wall_cycles", "timed_command_count", "timed_poll_reads",
    "timed_active_cycles", "timed_simulator_cycles",
    "timed_first_enqueue_cycle", "timed_last_retire_cycle",
    "timed_frontend_overflow_count", "timed_max_fifo_occupancy",
    "timed_decoder_stall_cycles", "timed_fifo_full_stall_cycles",
    "timed_scoreboard_stall_cycles", "timed_issue_stall_cycles",
    "timed_unit_stall_cycles", "timed_dram_stall_cycles",
    "timed_queue_wait_cycles", "timed_dependency_stall_cycles",
    "timed_resource_stall_cycles",
}
REGISTER_RESOURCES = {"VRF", "MRF", "SRF", "REG"}
TEMPLATE_FIELDS = {
    "goal_name", "goal_description", "iteration", "target_file",
    "hardware", "metrics", "clusters", "skills", "constraints",
    "self_verify", "gate_commands", "cost_model", "graph_context",
}
AGENT_RUNTIME_SAFETY = [
    "Never run `make clean` from the repository root. Use the configured "
    "firmware build or test commands instead.",
    "Never delete, move, rename, or modify `jimu-dse/results` or any "
    "`run-*` directory; those paths contain the active run state.",
    "This is a non-interactive optimization iteration. Do not ask the user "
    "what to do next: either implement one evidence-backed candidate in the "
    "allowed target or return a concrete no-change reason.",
]
TOP_LEVEL_KEYS = {
    "schema_version", "name", "description", "target", "agent", "prompt",
    "skills", "probe", "acceptance", "loop", "artifacts",
}
SECTION_KEYS = {
    "target": {
        "firmware", "baseline", "allowed_files", "hardware", "sequence_lengths",
        "build", "layout",
    },
    "agent": {
        "backend", "model", "timeout_seconds", "context_files", "work_budget",
    },
    "prompt": {
        "template", "goal", "constraints", "self_verify",
    },
    "probe": {
        "metrics", "dag", "cycle_limit", "scoring_sequence_length",
        "cost_model", "cycle_model", "timed_device", "workload_manifest",
    },
    "acceptance": {"gates", "score"},
    "loop": {
        "max_iterations", "min_score_delta", "max_no_improvement",
        "target_score", "mode", "evaluate_timeout_candidate",
    },
    "artifacts": {
        "save_candidates", "save_diffs", "save_prompts", "save_probes",
        "save_graphs",
    },
}


class ConfigError(ValueError):
    pass


ProgressCallback = Callable[[str], None]


def _progress(message: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[JIMU {stamp}] {message}", file=sys.stderr, flush=True)


def _metric_snapshot(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    parts = []
    for item in config["acceptance"]["score"]:
        name = item["metric"]
        if name in metrics:
            parts.append(f"{name}={metrics[name]}")
    for name in ("predicted_npu_cycles", "overlap_saved_cycles"):
        if name in metrics and name not in {
            item["metric"] for item in config["acceptance"]["score"]
        }:
            parts.append(f"{name}={metrics[name]}")
    return ", ".join(parts) or "configured score metrics unavailable"


def _require_yaml() -> None:
    if yaml is None:
        raise ConfigError(
            "PyYAML is required. Install dependencies with: pip install PyYAML"
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    _require_yaml()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def _repo_path(value: str, field: str, must_exist: bool = True) -> Path:
    candidate = (REPO_ROOT / value).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ConfigError(f"{field} must stay inside the repository: {value}") from exc
    if must_exist and not candidate.exists():
        raise ConfigError(f"{field} does not exist: {value}")
    return candidate


def _unknown_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"unknown field(s) in {where}: {', '.join(unknown)}")


def parse_skill(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not match:
        raise ConfigError(f"skill lacks YAML front matter: {path.relative_to(REPO_ROOT)}")
    _require_yaml()
    metadata = yaml.safe_load(match.group(1))
    required = {"name", "description"}
    missing = required - set(metadata or {})
    if not isinstance(metadata, dict) or missing:
        raise ConfigError(
            f"skill {path.relative_to(REPO_ROOT)} missing metadata: "
            f"{', '.join(sorted(missing))}"
        )
    metadata.setdefault("version", "1.0.0")
    metadata.setdefault("category", "general")
    return metadata, match.group(2).strip()


def load_config(goal: str | None = None, config_path: str | None = None) -> dict[str, Any]:
    if config_path:
        path = Path(config_path).resolve()
    else:
        if not goal:
            raise ConfigError("--goal or --config is required")
        path = GOALS_DIR / goal / "goal.yaml"
    if not path.is_file():
        raise ConfigError(f"goal configuration not found: {path}")
    config = _read_yaml(path)
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    public = {k: v for k, v in config.items() if not k.startswith("_")}
    _unknown_keys(public, TOP_LEVEL_KEYS, "root")
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    for required in ("name", "description", "target", "agent", "prompt", "skills",
                     "probe", "acceptance", "loop", "artifacts"):
        if required not in config:
            raise ConfigError(f"missing required field: {required}")
    for section, keys in SECTION_KEYS.items():
        value = config[section]
        if not isinstance(value, dict):
            raise ConfigError(f"{section} must be a mapping")
        _unknown_keys(value, keys, section)

    target = config["target"]
    for key in ("firmware", "baseline", "allowed_files", "hardware", "sequence_lengths"):
        if key not in target:
            raise ConfigError(f"missing required field: target.{key}")
    firmware = _repo_path(target["firmware"], "target.firmware")
    _repo_path(target["baseline"], "target.baseline")
    allowed = target["allowed_files"]
    if not isinstance(allowed, list) or not allowed:
        raise ConfigError("target.allowed_files must be a non-empty list")
    allowed_paths = [_repo_path(x, "target.allowed_files", must_exist=False) for x in allowed]
    if firmware not in allowed_paths:
        raise ConfigError("target.firmware must be included in target.allowed_files")
    hardware = target["hardware"]
    expected_hw = {"dim", "hidden", "num_head"}
    if not isinstance(hardware, dict) or set(hardware) != expected_hw:
        raise ConfigError(f"target.hardware must contain exactly {sorted(expected_hw)}")
    seqs = target["sequence_lengths"]
    if not isinstance(seqs, list) or not seqs or any(not isinstance(x, int) or x < 1 for x in seqs):
        raise ConfigError("target.sequence_lengths must contain positive integers")
    layout_version = target.get("layout", LEGACY_LAYOUT)
    if layout_version not in SUPPORTED_LAYOUTS:
        raise ConfigError(
            f"target.layout must be one of {', '.join(sorted(SUPPORTED_LAYOUTS))}"
        )
    if layout_version == PACKED_LAYOUT:
        try:
            for seq_len in seqs:
                bert_dram_layout(
                    hardware["dim"], hardware["hidden"], seq_len,
                    version=layout_version,
                )
        except ValueError as exc:
            raise ConfigError(f"invalid packed BERT layout: {exc}") from exc
        if hardware["dim"] > 8 and hardware["num_head"] != 1:
            raise ConfigError(
                "packed BERT layouts wider than 8 require num_head=1 with the "
                "current 8-bit repeating vector mask"
            )
    build_spec = target.get("build")
    if build_spec is not None:
        if not isinstance(build_spec, dict):
            raise ConfigError("target.build must be a mapping")
        _unknown_keys(
            build_spec, {"command", "elf", "cwd", "environment"},
            "target.build",
        )
        command = build_spec.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise ConfigError("target.build.command must be a non-empty string list")
        _repo_path(build_spec.get("elf", ""), "target.build.elf", must_exist=False)
        _repo_path(build_spec.get("cwd", "."), "target.build.cwd")
        environment = build_spec.get("environment", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ConfigError("target.build.environment must map strings to strings")

    agent = config["agent"]
    if agent.get("backend") not in {"pi", "opencode"}:
        raise ConfigError("agent.backend must be pi or opencode")
    if not isinstance(agent.get("timeout_seconds"), int) or agent["timeout_seconds"] < 0:
        raise ConfigError(
            "agent.timeout_seconds must be a non-negative integer (0 disables it)"
        )
    for context in agent.get("context_files", []):
        _repo_path(context, "agent.context_files")
    work_budget = agent.get("work_budget")
    if work_budget is not None:
        if not isinstance(work_budget, dict):
            raise ConfigError("agent.work_budget must be a mapping")
        _unknown_keys(
            work_budget,
            {
                "max_primary_hypotheses", "analysis_deadline_seconds",
                "edit_deadline_seconds", "return_deadline_seconds",
            },
            "agent.work_budget",
        )
        for key in (
            "max_primary_hypotheses", "analysis_deadline_seconds",
            "edit_deadline_seconds", "return_deadline_seconds",
        ):
            if not isinstance(work_budget.get(key), int) or work_budget[key] < 1:
                raise ConfigError(f"agent.work_budget.{key} must be a positive integer")
        deadlines = [
            work_budget["analysis_deadline_seconds"],
            work_budget["edit_deadline_seconds"],
            work_budget["return_deadline_seconds"],
        ]
        if agent["timeout_seconds"]:
            deadlines.append(agent["timeout_seconds"])
        if deadlines != sorted(deadlines) or len(set(deadlines)) != len(deadlines):
            raise ConfigError(
                "agent work-budget deadlines must be strictly ordered"
                + (
                    " and end before agent.timeout_seconds"
                    if agent["timeout_seconds"] else ""
                )
            )

    skills = config["skills"]
    if not isinstance(skills, list) or not skills:
        raise ConfigError("skills must be a non-empty ordered list")
    names: set[str] = set()
    for index, skill in enumerate(skills):
        if not isinstance(skill, dict) or set(skill) != {"name", "path"}:
            raise ConfigError(f"skills[{index}] must contain exactly name and path")
        if skill["name"] in names:
            raise ConfigError(f"duplicate skill: {skill['name']}")
        names.add(skill["name"])
        metadata, _ = parse_skill(_repo_path(skill["path"], f"skills[{index}].path"))
        if metadata["name"] != skill["name"]:
            raise ConfigError(
                f"skill name mismatch: config={skill['name']} metadata={metadata['name']}"
            )

    template = config["prompt"].get("template")
    if not isinstance(template, str) or not template.strip():
        raise ConfigError("prompt.template must be a non-empty string")
    used_fields = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", template))
    unknown_fields = used_fields - TEMPLATE_FIELDS
    if unknown_fields:
        raise ConfigError(f"unknown prompt template field(s): {', '.join(sorted(unknown_fields))}")

    metrics = config["probe"].get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("probe.metrics must be a non-empty list")
    if len(metrics) != len(set(metrics)):
        raise ConfigError("probe.metrics contains duplicates")
    unsupported = set(metrics) - SUPPORTED_METRICS
    if unsupported:
        raise ConfigError(f"unsupported probe metric(s): {', '.join(sorted(unsupported))}")
    scoring_seq = config["probe"].get(
        "scoring_sequence_length", target["sequence_lengths"][-1]
    )
    if scoring_seq not in target["sequence_lengths"]:
        raise ConfigError(
            "probe.scoring_sequence_length must appear in target.sequence_lengths"
        )
    cost_model = config["probe"].get("cost_model")
    if cost_model is not None:
        if not isinstance(cost_model, dict):
            raise ConfigError("probe.cost_model must be a mapping")
        _unknown_keys(
            cost_model,
            {"memory_weight", "register_weight", "register_resources"},
            "probe.cost_model",
        )
        for key in ("memory_weight", "register_weight"):
            value = cost_model.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(f"probe.cost_model.{key} must be non-negative")
        resources = cost_model.get("register_resources")
        if (
            not isinstance(resources, list)
            or not resources
            or len(resources) != len(set(resources))
        ):
            raise ConfigError(
                "probe.cost_model.register_resources must be a non-empty unique list"
            )
        invalid_resources = set(resources) - REGISTER_RESOURCES
        if invalid_resources:
            raise ConfigError(
                "unsupported register resource(s): "
                + ", ".join(sorted(invalid_resources))
            )
    if "estimated_time" in metrics and cost_model is None:
        raise ConfigError(
            "probe.cost_model is required when estimated_time is requested"
        )
    cycle_model = config["probe"].get("cycle_model")
    cycle_metrics = {
        "scalesim_layer_count", "scalesim_compute_cycles",
        "scalesim_stall_cycles", "trace_memory_cycles", "auxiliary_cycles",
        "predicted_npu_cycles", "parallel_predicted_npu_cycles",
        "overlap_saved_cycles", "net_parallelism_savings_cycles",
        "gross_overlap_cycles", "scheduler_idle_hole_cycles",
        "rtl_completion_makespan_cycles", "rtl_idle_cycles",
        "rtl_retirement_tail_cycles", "memory_compute_overlap_cycles",
        "max_concurrent_ops", "dram_bus_utilization", "mvu_utilization",
        "vmm_utilization", "mmm_utilization", "spu_utilization",
        "schedule_chain_count", "rtl_predicted_npu_cycles",
        "serial_command_cycles", "memory_compute_overlap_ratio",
        "load_utilization", "store_utilization", "vector_utilization",
        "control_utilization", "rtl_counter_cycles",
        "rtl_counter_active_cycles",
        "rtl_counter_memory_compute_overlap_cycles",
        "rtl_counter_frontend_full_stall_cycles",
        "rtl_counter_dependency_stall_cycles", "rtl_counter_unit_stall_cycles",
        "rtl_counter_dram_stall_cycles", "rtl_counter_bank_stall_cycles",
        "rtl_counter_barrier_stall_cycles", "rtl_counter_dispatches",
        "rtl_counter_completions", "rtl_counter_max_inflight",
    }
    if cycle_model is not None:
        if not isinstance(cycle_model, dict):
            raise ConfigError("probe.cycle_model must be a mapping")
        _unknown_keys(cycle_model, {"profile"}, "probe.cycle_model")
        profile_path = _repo_path(
            cycle_model.get("profile", ""), "probe.cycle_model.profile"
        )
        profile = _read_yaml(profile_path)
        _unknown_keys(
            profile,
            {
                "schema_version", "name", "backend", "source",
                "scalesim_config", "include_scalesim_stalls", "memory",
                "instruction_latencies", "scheduler", "rtl",
            },
            "cycle model profile",
        )
        profile_schema = profile.get("schema_version")
        backend = profile.get("backend")
        expected_backend = {
            1: "scalesim",
            2: "scalesim-parallel",
            3: "verilator-rtl",
        }.get(profile_schema)
        if expected_backend is None:
            raise ConfigError("cycle model schema_version must be 1, 2, or 3")
        if backend != expected_backend:
            raise ConfigError(
                f"cycle model schema_version {profile_schema} requires "
                f"backend {expected_backend}"
            )
        if profile_schema == 3:
            try:
                from emulator.npu_rtl_sim import RtlTimingProfile
                RtlTimingProfile.from_dict(profile)
            except (ImportError, ValueError) as exc:
                raise ConfigError(f"invalid Verilator RTL cycle profile: {exc}") from exc
        else:
            _repo_path(
                profile.get("scalesim_config", ""),
                "cycle model scalesim_config",
            )
            memory = profile.get("memory", {})
            if not isinstance(memory, dict):
                raise ConfigError("cycle model memory must be a mapping")
            _unknown_keys(
                memory,
                {"bytes_per_cycle", "setup_cycles", "element_bytes"},
                "cycle model memory",
            )
            for key in ("bytes_per_cycle", "setup_cycles", "element_bytes"):
                value = memory.get(key)
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ConfigError(f"cycle model memory.{key} must be positive")
            latencies = profile.get("instruction_latencies")
            if not isinstance(latencies, dict) or not latencies:
                raise ConfigError(
                    "cycle model instruction_latencies must be a non-empty mapping"
                )
            for name, value in latencies.items():
                if (
                    not isinstance(name, str) or not isinstance(value, int)
                    or value < 0
                ):
                    raise ConfigError(
                        "cycle model instruction latencies must be non-negative integers"
                    )
            scheduler = profile.get("scheduler")
            if profile_schema == 1 and scheduler is not None:
                raise ConfigError("legacy cycle model profile cannot define scheduler")
        if profile_schema == 2:
            if not isinstance(scheduler, dict):
                raise ConfigError("parallel cycle model scheduler must be a mapping")
            _unknown_keys(
                scheduler,
                {
                    "issue_width", "queue_depth", "implicit_chain_policy",
                    "cross_chain_overlap", "chain_commit_cycles", "resources",
                },
                "cycle model scheduler",
            )
            for key in ("issue_width", "queue_depth"):
                value = scheduler.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ConfigError(
                        f"cycle model scheduler.{key} must be a positive integer"
                    )
            commit = scheduler.get("chain_commit_cycles")
            if not isinstance(commit, int) or isinstance(commit, bool) or commit < 0:
                raise ConfigError(
                    "cycle model scheduler.chain_commit_cycles must be non-negative"
                )
            if scheduler.get("implicit_chain_policy") != "ordered_stream":
                raise ConfigError(
                    "cycle model scheduler.implicit_chain_policy must be ordered_stream"
                )
            if scheduler.get("cross_chain_overlap") is not False:
                raise ConfigError(
                    "cycle model scheduler.cross_chain_overlap must be false"
                )
            resources = scheduler.get("resources")
            expected_resources = {"dram_bus", "vmm", "mmm", "mvu", "spu"}
            if not isinstance(resources, dict) or set(resources) != expected_resources:
                raise ConfigError(
                    "cycle model scheduler.resources must contain exactly "
                    + ", ".join(sorted(expected_resources))
                )
            for name, count in resources.items():
                if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                    raise ConfigError(
                        f"cycle model scheduler.resources.{name} must be positive"
                    )
    if set(metrics) & cycle_metrics and cycle_model is None:
        raise ConfigError(
            "probe.cycle_model is required when cycle metrics are requested"
        )

    timed_metric_names = {name for name in SUPPORTED_METRICS if name.startswith("timed_")}
    timed_device = config["probe"].get("timed_device")
    if timed_device is not None:
        if not isinstance(timed_device, dict):
            raise ConfigError("probe.timed_device must be a mapping")
        _unknown_keys(timed_device, {"profile"}, "probe.timed_device")
        profile_path = _repo_path(
            timed_device.get("profile", ""), "probe.timed_device.profile"
        )
        profile = _read_yaml(profile_path)
        if profile.get("schema_version") != 1 or not isinstance(
            profile.get("timed_device"), dict
        ):
            raise ConfigError(
                "timed device profile must use schema_version 1 and define timed_device"
            )
    if set(metrics) & timed_metric_names and timed_device is None:
        raise ConfigError(
            "probe.timed_device is required when timed device metrics are requested"
        )
    workload_manifest = config["probe"].get("workload_manifest")
    if workload_manifest is not None:
        manifest_path = _repo_path(
            workload_manifest, "probe.workload_manifest"
        )
        manifest = _read_yaml(manifest_path)
        if manifest.get("schema_version", 1) != 1:
            raise ConfigError("workload manifest schema_version must be 1")
        if not isinstance(manifest.get("tensors", []), list):
            raise ConfigError("workload manifest tensors must be a list")

    gates = config["acceptance"].get("gates")
    if not isinstance(gates, list) or not gates:
        raise ConfigError("acceptance.gates must be a non-empty list")
    gate_names: set[str] = set()
    for gate in gates:
        allowed_gate = {"name", "type", "command", "timeout_seconds", "success_codes"}
        if not isinstance(gate, dict):
            raise ConfigError("each acceptance gate must be a mapping")
        _unknown_keys(gate, allowed_gate, "acceptance.gates[]")
        if gate.get("name") in gate_names:
            raise ConfigError(f"duplicate gate: {gate.get('name')}")
        gate_names.add(gate.get("name"))
        if gate.get("type") not in {"allowed_files", "build", "probe", "command"}:
            raise ConfigError(f"unsupported gate type: {gate.get('type')}")
        if gate.get("type") == "command" and not gate.get("command"):
            raise ConfigError(f"command gate {gate.get('name')} requires command")

    score = config["acceptance"].get("score")
    if not isinstance(score, list) or not score:
        raise ConfigError("acceptance.score must be a non-empty list")
    score_names: set[str] = set()
    total_weight = 0.0
    for item in score:
        if not isinstance(item, dict) or set(item) - {
            "metric", "direction", "weight", "target"
        }:
            raise ConfigError("invalid acceptance.score entry")
        metric = item.get("metric")
        if metric in score_names:
            raise ConfigError(f"duplicate score metric: {metric}")
        score_names.add(metric)
        if metric not in metrics:
            raise ConfigError(f"score metric is not probed: {metric}")
        if item.get("direction") not in {"minimize", "maximize"}:
            raise ConfigError(f"invalid score direction for {metric}")
        weight = item.get("weight")
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ConfigError(f"invalid score weight for {metric}")
        total_weight += float(weight)
    if abs(total_weight - 1.0) > 1e-9:
        raise ConfigError(f"score weights must sum to 1.0, got {total_weight}")

    loop = config["loop"]
    mode = loop.get("mode", "goal_driven")
    if mode not in {"goal_driven", "full_iterations"}:
        raise ConfigError("loop.mode must be goal_driven or full_iterations")
    for key in ("max_iterations", "max_no_improvement"):
        if not isinstance(loop.get(key), int) or loop[key] < 1:
            raise ConfigError(f"loop.{key} must be a positive integer")
    if not isinstance(loop.get("min_score_delta"), (int, float)):
        raise ConfigError("loop.min_score_delta must be numeric")
    target_score = loop.get("target_score")
    if target_score is not None and not isinstance(target_score, (int, float)):
        raise ConfigError("loop.target_score must be numeric or null")
    if not isinstance(loop.get("evaluate_timeout_candidate", False), bool):
        raise ConfigError("loop.evaluate_timeout_candidate must be boolean")


def resolved_config(
    config: dict[str, Any], agent: str | None = None, model: str | None = None,
    full_iterations: bool = False, max_iterations: int | None = None,
    agent_timeout: int | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    result["loop"].setdefault("mode", "goal_driven")
    if os.getenv("JIMU_MAX_ITER"):
        result["loop"]["max_iterations"] = int(os.environ["JIMU_MAX_ITER"])
    if os.getenv("JIMU_AGENT_TIMEOUT"):
        result["agent"]["timeout_seconds"] = int(os.environ["JIMU_AGENT_TIMEOUT"])
    if os.getenv("OPENCODE_MODEL"):
        result["agent"]["model"] = os.environ["OPENCODE_MODEL"]
    if agent:
        result["agent"]["backend"] = agent
    if model:
        result["agent"]["model"] = model
    if max_iterations is not None:
        result["loop"]["max_iterations"] = max_iterations
    if agent_timeout is not None:
        result["agent"]["timeout_seconds"] = agent_timeout
        if agent_timeout == 0:
            result["agent"].pop("work_budget", None)
    if full_iterations:
        result["loop"]["mode"] = "full_iterations"
    validate_config(result)
    result["_mode_overridden"] = bool(full_iterations)
    return result


def config_fingerprint(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    canonical = json.dumps(public, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def timing_profile_fingerprints(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Fingerprint timing inputs whose contents are not covered by config paths."""
    result: dict[str, dict[str, str]] = {}
    for label, section in (
        ("cycle_model", config.get("probe", {}).get("cycle_model", {})),
        ("timed_device", config.get("probe", {}).get("timed_device", {})),
    ):
        profile = section.get("profile") if isinstance(section, dict) else None
        if not profile:
            continue
        path = _repo_path(profile, f"probe.{label}.profile")
        result[label] = {
            "path": str(profile),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


_FEEDBACK_METRICS = (
    "rtl_predicted_npu_cycles", "serial_command_cycles",
    "net_parallelism_savings_cycles", "gross_overlap_cycles",
    "scheduler_idle_hole_cycles", "memory_compute_overlap_cycles",
    "logical_dram_payload_bytes", "modeled_dram_transaction_bytes",
    "dram_bus_utilization", "load_utilization", "store_utilization",
    "vector_utilization", "mvu_utilization",
    "rtl_counter_frontend_full_stall_cycles",
    "rtl_counter_dependency_stall_cycles", "rtl_counter_dram_stall_cycles",
    "rtl_counter_bank_stall_cycles",
)


def _candidate_feedback(
    reference: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return compact candidate deltas for the next optimization iteration."""
    feedback: dict[str, dict[str, Any]] = {}
    for name in _FEEDBACK_METRICS:
        if name not in reference or name not in candidate:
            continue
        before, after = reference[name], candidate[name]
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        feedback[name] = {
            "reference": before,
            "candidate": after,
            "delta": after - before,
        }
    return feedback


def _attempt_history_context(attempts: list[dict[str, Any]] | None) -> str:
    if not attempts:
        return ""
    compact = []
    for item in attempts[-3:]:
        compact.append({
            "iteration": item.get("iteration"),
            "status": item.get("status"),
            "promoted": item.get("promoted", False),
            "candidate_delta_vs_previous_best": item.get(
                "attempt_feedback", "candidate was not measured"
            ),
        })
    return (
        "\n\n## Previous candidate feedback\n"
        "Use these measured RTL deltas to avoid repeating rejected resource "
        "migrations. Negative cycle deltas are improvements; byte reductions "
        "are diagnostic and do not override makespan.\n"
        + json.dumps(compact, sort_keys=True, indent=2)
    )


def render_prompt(
    config: dict[str, Any], iteration: int = 1,
    metrics: dict[str, Any] | None = None, clusters: list[str] | None = None,
    graph_context: str | None = None,
    baseline_metrics: dict[str, Any] | None = None,
    attempt_history: list[dict[str, Any]] | None = None,
) -> str:
    skill_parts = []
    for index, item in enumerate(config["skills"], 1):
        metadata, body = parse_skill(_repo_path(item["path"], "skill.path"))
        skill_parts.append(
            f"## Skill {index}: {metadata['name']} v{metadata['version']}\n"
            f"{metadata['description']}\n\n{body}"
        )
    gate_commands = "\n".join(
        gate.get("command", gate["type"]) for gate in config["acceptance"]["gates"]
    )
    constraints = [
        *config["prompt"].get("constraints", []),
        *AGENT_RUNTIME_SAFETY,
    ]
    work_budget = config["agent"].get("work_budget")
    work_contract = ""
    if work_budget:
        work_contract = (
            "\n\n## Iteration work contract\n"
            f"- Work on at most {work_budget['max_primary_hypotheses']} primary "
            "optimization hypothesis. Do not begin a second optimization stage.\n"
            f"- By {work_budget['analysis_deadline_seconds']} seconds: select the "
            "single hypothesis and stop broad exploration.\n"
            f"- By {work_budget['edit_deadline_seconds']} seconds: finish the target "
            "file modification. If no safe edit is ready, return without changing it.\n"
            f"- By {work_budget['return_deadline_seconds']} seconds: finish at most "
            "one targeted quick check, summarize the candidate, and return control.\n"
            "- Do not run the full correctness matrix or create a replacement official "
            "probe; the closed-loop controller owns all configured gates and scoring.\n"
            "- Once a candidate passes the targeted quick check, stop. Record additional "
            "ideas for future iterations instead of implementing them now.\n"
        )
    values = {
        "goal_name": config["name"],
        "goal_description": config["prompt"]["goal"],
        "iteration": iteration,
        "target_file": config["target"]["firmware"],
        "hardware": json.dumps(config["target"]["hardware"], sort_keys=True),
        "cost_model": json.dumps(
            config["probe"].get("cost_model", {}), sort_keys=True, indent=2
        ),
        "metrics": json.dumps(
            _prompt_metric_view(config, metrics or {}, baseline_metrics),
            sort_keys=True, indent=2,
        ),
        "clusters": "\n".join(clusters or ["(not available during preview)"]),
        "graph_context": graph_context or "(not available during preview)",
        "skills": "\n\n".join(skill_parts),
        "constraints": "\n".join(f"- {item}" for item in constraints),
        "self_verify": config["prompt"].get("self_verify", ""),
        "gate_commands": gate_commands,
    }
    feedback = _attempt_history_context(attempt_history)
    return (
        config["prompt"]["template"].format(**values).strip()
        + feedback + work_contract + "\n"
    )


def _prompt_metric_view(
    config: dict[str, Any], metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep scoring feedback actionable while leaving audit fields in artifacts."""
    if not ({"parallel_predicted_npu_cycles", "rtl_predicted_npu_cycles"} & set(metrics)):
        return metrics
    scored_names = [item["metric"] for item in config["acceptance"]["score"]]
    score = {name: metrics[name] for name in scored_names if name in metrics}
    comparison: dict[str, Any] = {}
    if baseline_metrics:
        for name, current in score.items():
            if name not in baseline_metrics:
                continue
            baseline = float(baseline_metrics[name])
            current_value = float(current)
            direction = next(
                item["direction"] for item in config["acceptance"]["score"]
                if item["metric"] == name
            )
            if baseline == 0:
                improvement = 0.0 if current_value == 0 else None
            elif direction == "minimize":
                improvement = (baseline - current_value) / abs(baseline)
            else:
                improvement = (current_value - baseline) / abs(baseline)
            comparison[name] = {
                "run_baseline": baseline_metrics[name],
                "current_validated_best": current,
                "absolute_delta": current_value - baseline,
                "improvement_fraction": improvement,
            }
    optimization_names = (
        "predicted_npu_cycles", "rtl_predicted_npu_cycles",
        "rtl_completion_makespan_cycles", "rtl_idle_cycles",
        "rtl_retirement_tail_cycles", "serial_command_cycles",
        "overlap_saved_cycles", "net_parallelism_savings_cycles",
        "gross_overlap_cycles", "scheduler_idle_hole_cycles",
        "memory_compute_overlap_cycles", "max_concurrent_ops",
        "logical_dram_payload_bytes", "modeled_dram_transaction_bytes",
        "dram_bus_utilization", "mvu_utilization", "vmm_utilization",
        "mmm_utilization", "spu_utilization", "load_utilization",
        "store_utilization", "vector_utilization", "control_utilization",
        "rtl_counter_dependency_stall_cycles",
        "rtl_counter_dram_stall_cycles", "rtl_counter_bank_stall_cycles",
        "rtl_counter_barrier_stall_cycles",
    )
    breakdown_names = (
        "scalesim_compute_cycles", "trace_memory_cycles", "auxiliary_cycles",
    )
    return {
        "score": score,
        "comparison_to_run_baseline": comparison or "preview unavailable",
        "optimization_diagnostics": {
            name: metrics[name] for name in optimization_names if name in metrics
        },
        "legacy_model_breakdown": {
            name: metrics[name] for name in breakdown_names if name in metrics
        },
        "audit_note": (
            "Complete raw metrics, layer/stall counts, chain count, and event "
            "timeline remain in the probe and timing-schedule artifacts."
        ),
    }


def score_metrics(
    baseline: dict[str, float], candidate: dict[str, float],
    score_config: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    details: dict[str, Any] = {}
    total = 0.0
    for item in score_config:
        name = item["metric"]
        base = float(baseline[name])
        value = float(candidate[name])
        if base == 0:
            normalized = 0.0 if value == 0 else (-1.0 if item["direction"] == "minimize" else 1.0)
        elif item["direction"] == "minimize":
            normalized = (base - value) / abs(base)
        else:
            normalized = (value - base) / abs(base)
        contribution = normalized * float(item["weight"])
        total += contribution
        details[name] = {
            "baseline": base, "value": value, "direction": item["direction"],
            "weight": item["weight"], "normalized": normalized,
            "contribution": contribution,
        }
    return total, details


def calculate_cost_metrics(
    events: list[dict[str, Any]], dram_stats: dict[str, Any],
    cost_model: dict[str, Any] | None,
) -> dict[str, float]:
    """Count NPU DRAM operations and selected register def/use accesses."""
    model = cost_model or {
        "memory_weight": 10,
        "register_weight": 1,
        "register_resources": ["VRF", "MRF", "SRF", "REG"],
    }
    memory_reads = int(dram_stats.get("vec_rd_ops", 0)) + int(
        dram_stats.get("mat_rd_ops", 0)
    )
    memory_writes = int(dram_stats.get("vec_wr_ops", 0)) + int(
        dram_stats.get("mat_wr_ops", 0)
    )
    resources = set(model["register_resources"])
    register_reads = sum(
        1
        for event in events
        for resource in event.get("uses", [])
        if resource and resource[0] in resources
    )
    register_writes = sum(
        1
        for event in events
        for resource in event.get("defs", [])
        if resource and resource[0] in resources
    )
    memory_accesses = memory_reads + memory_writes
    register_accesses = register_reads + register_writes
    estimated = (
        memory_accesses * float(model["memory_weight"])
        + register_accesses * float(model["register_weight"])
    )
    return {
        "memory_access_count": memory_accesses,
        "memory_read_count": memory_reads,
        "memory_write_count": memory_writes,
        "register_access_count": register_accesses,
        "register_read_count": register_reads,
        "register_write_count": register_writes,
        "estimated_time": estimated,
    }


def _project_environment() -> dict[str, str]:
    """Build a stable PATH for project Python and user-installed agent CLIs."""
    env = os.environ.copy()
    candidates = (
        REPO_ROOT / ".venv" / "bin",
        REPO_ROOT / ".venv" / "Scripts",
        Path.home() / ".npm-global" / "bin",
        Path.home() / ".local" / "bin",
    )
    prefixes = [str(directory) for directory in candidates if directory.is_dir()]
    if prefixes:
        env["PATH"] = os.pathsep.join((*prefixes, env.get("PATH", "")))
    return env


def _run(
    command: list[str] | str, timeout: int | float | None,
    shell: bool = False, heartbeat: Callable[[float], None] | None = None,
    heartbeat_seconds: float = 20.0 * 60.0,
    env: dict[str, str] | None = None,
    raw_output_prefix: Path | None = None,
) -> dict[str, Any]:
    if env is None:
        env = _project_environment()

    def output_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    def bounded_output(value: str | bytes | None) -> str:
        decoded = output_text(value)
        if len(decoded) <= 20000:
            return decoded
        return decoded[:4000] + "\n...[output truncated]...\n" + decoded[-15970:]

    def captured_output(value: str | bytes | None, stream: str) -> str:
        decoded = output_text(value)
        if raw_output_prefix is not None:
            suffix = ".stdout.jsonl" if stream == "stdout" else ".stderr.log"
            output_path = Path(f"{raw_output_prefix}{suffix}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(decoded, encoding="utf-8")
        return bounded_output(decoded)

    started = time.monotonic()
    try:
        if heartbeat is not None:
            proc = subprocess.Popen(
                command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, shell=shell, env=env,
            )
            deadline = None if not timeout else started + float(timeout)
            try:
                while True:
                    remaining = (
                        None if deadline is None
                        else max(0.0, deadline - time.monotonic())
                    )
                    wait_for = heartbeat_seconds
                    if remaining is not None:
                        wait_for = min(wait_for, remaining)
                    try:
                        stdout, stderr = proc.communicate(
                            timeout=max(wait_for, 0.001)
                        )
                        return {
                            "exit_code": proc.returncode,
                            "stdout": captured_output(stdout, "stdout"),
                            "stderr": captured_output(stderr, "stderr"),
                            "timed_out": False,
                            "spawn_error": False,
                            "duration_seconds": round(
                                time.monotonic() - started, 3
                            ),
                        }
                    except subprocess.TimeoutExpired:
                        elapsed = time.monotonic() - started
                        if deadline is not None and time.monotonic() >= deadline:
                            proc.kill()
                            stdout, stderr = proc.communicate()
                            return {
                                "exit_code": None,
                                "stdout": captured_output(stdout, "stdout"),
                                "stderr": captured_output(stderr, "stderr"),
                                "timed_out": True,
                                "spawn_error": False,
                                "duration_seconds": round(elapsed, 3),
                            }
                        heartbeat(elapsed)
            except BaseException:
                proc.kill()
                proc.communicate()
                raise
        proc = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True,
            timeout=timeout, shell=shell, env=env,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": captured_output(proc.stdout, "stdout"),
            "stderr": captured_output(proc.stderr, "stderr"),
            "timed_out": False,
            "spawn_error": False,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": None,
            "stdout": captured_output(exc.stdout, "stdout"),
            "stderr": captured_output(exc.stderr, "stderr"),
            "timed_out": True,
            "spawn_error": False,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except OSError as exc:
        return {
            "exit_code": None, "stdout": "", "stderr": str(exc),
            "timed_out": False, "spawn_error": True,
            "duration_seconds": round(time.monotonic() - started, 3),
        }


def _resolve_worktree_git_dir(value: str, platform: str | None = None) -> str:
    """Resolve a gitfile target across Windows-hosted WSL worktrees."""
    platform = platform or os.name
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if match and platform != "nt":
        remainder = match.group(2).replace("\\", "/")
        return f"/mnt/{match.group(1).lower()}/{remainder}"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


def _agent_environment() -> dict[str, str]:
    """Give subprocess agents a usable Git worktree in Windows/WSL setups."""
    env = _project_environment()
    gitfile = REPO_ROOT / ".git"
    if not gitfile.is_file():
        return env
    match = re.match(
        r"^gitdir:\s*(.+?)\s*$", gitfile.read_text(encoding="utf-8"), re.S
    )
    if not match:
        return env
    git_dir = _resolve_worktree_git_dir(match.group(1))
    if Path(git_dir).is_dir():
        env.update({"GIT_DIR": git_dir, "GIT_WORK_TREE": str(REPO_ROOT)})
    return env


def build_firmware(config: dict[str, Any], seq_len: int) -> dict[str, Any]:
    hw = config["target"]["hardware"]
    dim, hidden, heads = hw["dim"], hw["hidden"], hw["num_head"]
    build_spec = config["target"].get("build")
    if build_spec is not None:
        fields = {
            "firmware": config["target"]["firmware"],
            "dim": dim, "hidden": hidden, "num_head": heads,
            "seq_len": seq_len,
            "elf": build_spec["elf"],
            "repo": str(REPO_ROOT),
        }
        command = [value.format(**fields) for value in build_spec["command"]]
        env = os.environ.copy()
        env.update({
            key: value.format(**fields)
            for key, value in build_spec.get("environment", {}).items()
        })
        cwd = _repo_path(build_spec.get("cwd", "."), "target.build.cwd")
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command, cwd=cwd, env=env, text=True, capture_output=True,
                timeout=300,
            )
            result = {
                "exit_code": proc.returncode, "stdout": proc.stdout[-20000:],
                "stderr": proc.stderr[-20000:], "timed_out": False,
            }
        except (subprocess.TimeoutExpired, OSError) as exc:
            result = {
                "exit_code": None, "stdout": "", "stderr": str(exc),
                "timed_out": isinstance(exc, subprocess.TimeoutExpired),
            }
        elf_path = _repo_path(build_spec["elf"], "target.build.elf", must_exist=False)
        result.update({
            "duration_seconds": round(time.monotonic() - started, 3),
            "elf": str(elf_path.relative_to(REPO_ROOT)),
            "passed": result["exit_code"] == 0 and elf_path.is_file(),
        })
        return result
    layout = bert_dram_layout(
        dim, hidden, seq_len,
        version=config["target"].get("layout", LEGACY_LAYOUT),
    )
    env = os.environ.copy()
    env.update(layout.build_environment())
    env.update({
        "CC": os.getenv("CC", "riscv64-unknown-elf-gcc"),
        "NUM_HEAD": str(heads),
    })
    command = ["make", "-C", "firmware", f"BUILD_DIR=build_dim{dim}", "clean", "all"]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command, cwd=REPO_ROOT, env=env, text=True, capture_output=True,
            timeout=300,
        )
        result = {
            "exit_code": proc.returncode, "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-20000:], "timed_out": False,
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        result = {"exit_code": None, "stdout": "", "stderr": str(exc), "timed_out": True}
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    result["elf"] = f"firmware/build_dim{dim}/bert.elf"
    result["passed"] = result["exit_code"] == 0 and (REPO_ROOT / result["elf"]).is_file()
    return result


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _timing_schedule_context(schedule_path: Path) -> str:
    """Render bounded, actionable timing evidence for the agent prompt."""
    lines = [
        "## Parallel timing schedule",
        f"- timing_schedule_file={_display_path(schedule_path)}",
    ]
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        lines.append(f"- schedule_summary_unavailable={type(exc).__name__}")
        return "\n".join(lines)

    profile = schedule.get("profile", {})
    profile_canonical = json.dumps(
        profile, sort_keys=True, separators=(",", ":")
    ).encode()
    lines.extend([
        f"- timing_profile_name={schedule.get('model', 'unknown')}",
        "- timing_profile_sha256=" + hashlib.sha256(profile_canonical).hexdigest(),
    ])

    if schedule.get("backend") == "verilator-rtl":
        lines.extend([
            "- RTL resource mapping: DRAM commands use load/store plus the "
            "shared dram_bus; MV_MUL=mvu; other math/vector operations=vector; "
            "scalar/configuration operations=control.",
            "- The 128-bit RTL scoreboard enforces explicit semantic RAW/WAR/WAW; "
            "local SRAM bank ports, unit initiation intervals, finite ROB space, "
            "and chain fences add structural constraints.",
            "- DRAM and independent MVU/vector work execute concurrently. Use "
            "dependency_resources and rtl_stall_cycles_by_reason before moving "
            "a command.",
            "- RTL stall counters measure pressure on the oldest blocked command. "
            "A younger command can still dispatch in the same cycle, so these "
            "counters are not additive makespan losses.",
            "- rtl_completion_makespan_cycles is the last operation completion; "
            "rtl_idle_cycles includes the remaining in-order retirement tail.",
        ])
    else:
        lines.extend([
            "- Resource mapping: vector DRAM=dram_bus+vmm; matrix "
            "DRAM=dram_bus+mmm; MV_MUL=mvu; M_RD/M_WR=mmm; scalar and "
            "configuration operations=spu; other vector/activation operations=vmm.",
            "- Two events may overlap only when they have no RAW/WAR/WAW, "
            "configuration-fence, overlapping DRAM-address, shared-resource, "
            "queue-window, or chain-boundary conflict.",
            "- All DRAM transfers serialize on dram_bus. A vector DRAM transfer may "
            "overlap an independent MVU/MMM/SPU operation but not VMM work; a "
            "matrix DRAM transfer may overlap independent MVU/VMM/SPU work but not "
            "MMM work.",
        ])

    metrics = schedule.get("metrics", {})
    lines.extend([
        f"- parallel_cycles={metrics.get('parallel_predicted_npu_cycles', 'n/a')}",
        f"- rtl_cycles={metrics.get('rtl_predicted_npu_cycles', 'n/a')}",
        "- rtl_idle_cycles="
        f"{metrics.get('rtl_idle_cycles', 'n/a')}",
        "- net_parallelism_savings_cycles="
        f"{metrics.get('net_parallelism_savings_cycles', metrics.get('overlap_saved_cycles', 'n/a'))}",
        "- gross_overlap_cycles="
        f"{metrics.get('gross_overlap_cycles', 'n/a')}",
        "- scheduler_idle_hole_cycles="
        f"{metrics.get('scheduler_idle_hole_cycles', 'n/a')}",
        "- memory_compute_overlap_cycles="
        f"{metrics.get('memory_compute_overlap_cycles', 'n/a')}",
        "- logical_dram_payload_bytes="
        f"{metrics.get('logical_dram_payload_bytes', 'n/a')}",
        "- modeled_dram_transaction_bytes="
        f"{metrics.get('modeled_dram_transaction_bytes', 'n/a')}",
    ])
    diagnostics = schedule.get("optimization_diagnostics", {})
    bottlenecks = diagnostics.get("resource_bottlenecks", [])
    if bottlenecks:
        rendered = ", ".join(
            f"{item['resource']}={float(item['utilization']):.1%}"
            for item in bottlenecks
        )
        lines.append(f"- resource_utilization_rank={rendered}")
    waits = diagnostics.get("critical_event_wait_cycles_by_reason", {})
    if waits:
        lines.append(
            "- critical_event_wait_cycles_by_reason="
            + ", ".join(f"{name}:{cycles}" for name, cycles in waits.items())
        )
    events = diagnostics.get("critical_path_top_events", [])
    if events:
        lines.extend(["", "### Longest causal-chain events"])
        for event in events:
            lines.append(
                "- idx={idx}, raw={raw}, expanded={expanded}, op={op}, "
                "cycles=[{start},{end}), duration={duration}, resources={resources}, "
                "blocked_by={blocked}".format(
                    idx=event.get("idx"), raw=event.get("raw_instruction_idx"),
                    expanded=event.get("expanded_idx"), op=event.get("op"),
                    start=event.get("start_cycle"), end=event.get("end_cycle"),
                    duration=event.get("duration_cycles"),
                    resources="+".join(event.get("resources", [])) or "none",
                    blocked=",".join(event.get("blocking_reasons", [])) or "ready",
                )
            )
    blockers = diagnostics.get(
        "critical_path_top_blockers", diagnostics.get("top_blockers", [])
    )
    if blockers:
        lines.extend(["", "### Largest causal-chain waits"])
        for item in blockers:
            lines.append(
                f"- idx={item.get('idx')}, op={item.get('op')}, "
                f"wait={item.get('wait_cycles')} cycles, reasons="
                + ",".join(
                    item.get("blocking_reasons", item.get("reasons", []))
                )
            )
    note = diagnostics.get("source_mapping_note")
    if note:
        lines.append(f"- Trace mapping limitation: {note}.")
    return "\n".join(lines)


def _graph_context(
    dag_dir: Path, config: dict[str, Any], seq_len: int, dag: dict[str, Any],
    timing_schedule_path: Path | None = None,
    cross_layer_path: Path | None = None,
) -> str:
    """Build a bounded, configuration-stamped graph summary for the agent."""
    hw = config["target"]["hardware"]
    lines = [
        "Graph configuration:",
        (
            f"- dim={hw['dim']}, hidden={hw['hidden']}, "
            f"num_head={hw['num_head']}, seq_len={seq_len}"
        ),
        f"- artifact_directory={dag_dir}",
        "- These dynamic graphs describe the scored trace; the parallel timing "
        "schedule adds RAW/WAR/WAW, DRAM-range, and structural resource edges.",
        "- Reordering can improve parallel_predicted_npu_cycles when it exposes "
        "legal memory/compute overlap without lengthening the critical path.",
    ]
    if timing_schedule_path is not None:
        lines.extend(["", _timing_schedule_context(timing_schedule_path)])
    if cross_layer_path is not None and cross_layer_path.is_file():
        cross_layer = cross_layer_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        if cross_layer:
            lines.extend(["", "## Cross-layer tensor/command evidence", cross_layer[:6000]])
    stdout = dag.get("stdout", "")
    for pattern, label in (
        (r"(\d+) nodes,\s*(\d+) edges", "instruction_graph"),
        (r"(\d+) instructions\s*->\s*(\d+) micro-ops", "micro_ops"),
        (r"(\d+) clusters,\s*(\d+) micro-ops", "dram_clusters"),
        (r"(\d+) operators,\s*(\d+) edges", "operator_graph"),
        (r"(\d+) symbolic ops", "symbolic_graph"),
    ):
        match = re.search(pattern, stdout)
        if match:
            lines.append(f"- {label}: {', '.join(match.groups())}")

    budget = 12000
    for name in (
        "dram_clusters.txt", "micro_op_dag.txt", "op_graph.txt", "sym_graph.txt",
    ):
        path = dag_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        remaining = budget - sum(len(item) for item in lines)
        if remaining <= 200:
            break
        excerpt = text[: min(remaining, 3000)]
        if len(excerpt) < len(text):
            excerpt += "\n[truncated]"
        lines.extend(["", f"## {name}", excerpt])
    return "\n".join(lines)


def probe_firmware(
    config: dict[str, Any], seq_len: int, output_dir: Path | None = None,
) -> dict[str, Any]:
    build = build_firmware(config, seq_len)
    if not build["passed"]:
        return {"passed": False, "build": build, "metrics": {}, "clusters": []}
    hw = config["target"]["hardware"]
    timing_schedule_path = (
        output_dir / "timing-schedule.json" if output_dir is not None else None
    )
    cross_layer_path = (
        output_dir / "cross-layer-graph.txt" if output_dir is not None else None
    )
    cross_layer_json_path = (
        output_dir / "cross-layer-graph.json" if output_dir is not None else None
    )
    timed_profile = config["probe"].get("timed_device", {}).get("profile")
    workload_manifest = config["probe"].get("workload_manifest")
    script = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'jimu-dse/scripts')
import numpy as np
from emulator.npu_device_mini import NpuDeviceMini, MEM_DRAM
from emulator.npu_device_timed import TimedDeviceProfile, TimedNpuDevice
from emulator.npu_event_trace import EventTracer
from emulator.npu_cross_layer_graph import build_cross_layer_graph
from emulator.firmware_runner import load_initializer
from emulator.trace_recorder import TraceRecorder
from emulator.workload import WorkloadManifest
from iss.mini_rv64 import MiniRV64
from closed_loop import calculate_cost_metrics
dim={hw['dim']}; h={hw['hidden']}; sl={seq_len}
manifest=WorkloadManifest.load({workload_manifest!r}) if {bool(workload_manifest)!r} else None
npu=NpuDeviceMini(native_dim=dim); npu.set_hidden_size(h); npu.set_seq_len(sl)
npu._vrf[MEM_DRAM][0:h]=np.zeros(h,dtype=np.float32)
initializer=load_initializer(manifest.initializer if manifest else None)
if initializer:
    initializer(npu, manifest)
tracer=EventTracer(npu, manifest=manifest)
timed=None
device=npu
if {bool(timed_profile)!r}:
    timed=TimedNpuDevice(
        npu, TimedDeviceProfile.load({timed_profile!r}), manifest=manifest
    )
    device=timed
rec=TraceRecorder(device); cpu=MiniRV64()
cpu.set_mmio_device(rec); cpu.load_elf('{build['elf']}')
cpu.run(cycles={config['probe'].get('cycle_limit', 300000)})
if timed:
    timed.run_until_idle()
ds=npu.get_dram_stats()
dram_elements=(ds.get('vec_rd_elements',0)+ds.get('vec_wr_elements',0)+ds.get('mat_rd_elements',0)+ds.get('mat_wr_elements',0))
functional_container_bytes=dram_elements*4
mv=sum(1 for e in tracer.events if ((e['raw'] if isinstance(e,dict) else e.inst)>>24)&0xFF in (7,27))
metrics={{'total_bytes':functional_container_bytes,'dram_elements':dram_elements,'functional_container_bytes':functional_container_bytes,'instr_count':len(rec.inst_trace),'mv_mul_count':mv,'mat_rd_ops':ds.get('mat_rd_ops',0),'test_pass':0}}
metrics.update(calculate_cost_metrics(tracer.events, ds, {config["probe"].get("cost_model")!r}))
if timed:
    metrics.update(timed.metrics())
cycle_model={config["probe"].get("cycle_model")!r}
cycle_schedule=None
if cycle_model:
    import yaml
    with open(cycle_model['profile'], encoding='utf-8') as handle:
        timing_profile=yaml.safe_load(handle)
    if timing_profile.get('backend') == 'verilator-rtl':
        from emulator.npu_rtl_sim import simulate_trace as simulate_rtl_trace
        cycle_schedule=simulate_rtl_trace(
            tracer.events,
            {{'native_dim':dim, 'hidden':h, 'seq_len':sl}},
            timing_profile,
            artifact_path={(str(timing_schedule_path) if timing_schedule_path else None)!r},
        )
        metrics.update(cycle_schedule['metrics'])
        metrics['rtl_payload_bytes']=dram_elements*int(
            cycle_schedule.get('profile', {{}}).get('memory_element_bytes', 2)
        )
    else:
        sys.path.insert(0, 'jimu-dse/timing')
        from scalesim_adapter import simulate_trace as simulate_scalesim_trace
        metrics.update(simulate_scalesim_trace(
            tracer.events, {{'dim':dim, 'hidden':h}}, timing_profile,
            schedule_path={(str(timing_schedule_path) if timing_schedule_path else None)!r},
        ))
        if {bool(timing_schedule_path)!r}:
            schedule_file=Path({(str(timing_schedule_path) if timing_schedule_path else '')!r})
            if schedule_file.is_file():
                cycle_schedule=json.loads(schedule_file.read_text(encoding='utf-8'))
schedule=(
    cycle_schedule.get('events', []) if cycle_schedule is not None
    else (timed.timeline if timed else None)
)
if schedule is None and {str(timing_schedule_path) if timing_schedule_path else None!r}:
    schedule_file=Path({str(timing_schedule_path) if timing_schedule_path else None!r})
    if schedule_file.is_file():
        schedule=json.loads(schedule_file.read_text(encoding='utf-8'))
if {str(cross_layer_json_path) if cross_layer_json_path else None!r}:
    graph=build_cross_layer_graph(
        tracer.events, manifest=manifest, schedule=schedule,
        profile_name=(
            cycle_schedule.get('model') if cycle_schedule is not None
            else (timed.profile.name if timed else None)
        ),
    )
    graph_json=Path({str(cross_layer_json_path) if cross_layer_json_path else None!r})
    graph_json.parent.mkdir(parents=True, exist_ok=True)
    graph.write_json(graph_json)
    Path({str(cross_layer_path) if cross_layer_path else None!r}).write_text(
        graph.to_text(), encoding='utf-8'
    )
print(json.dumps(metrics))
tracer.unpatch()
"""
    result = _run([sys.executable, "-c", script], timeout=300)
    metrics: dict[str, Any] = {}
    if result["exit_code"] == 0:
        try:
            metrics = json.loads(result["stdout"].strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
    wanted = config["probe"]["metrics"]
    metrics = {key: metrics[key] for key in wanted if key in metrics}
    schedule_context = (
        _timing_schedule_context(timing_schedule_path)
        if timing_schedule_path and timing_schedule_path.is_file()
        else None
    )
    cross_layer_context = (
        cross_layer_path.read_text(encoding="utf-8", errors="replace")[:6000]
        if cross_layer_path and cross_layer_path.is_file()
        else None
    )
    probe = {
        "passed": result["exit_code"] == 0 and all(x in metrics for x in wanted if x != "test_pass"),
        "build": build, "process": result, "metrics": metrics, "clusters": [],
        "graph_context": "\n\n".join(
            item for item in (schedule_context, cross_layer_context) if item
        ) or "(graph generation disabled)",
        "sequence_length": seq_len,
    }
    if timing_schedule_path and timing_schedule_path.is_file():
        try:
            probe["timing_schedule"] = str(
                timing_schedule_path.relative_to(REPO_ROOT)
            )
        except ValueError:
            probe["timing_schedule"] = str(timing_schedule_path)
    if output_dir and config["probe"].get("dag", {}).get("enabled"):
        dag_dir = output_dir / f"dag-seq{seq_len}"
        dag = _run([
            sys.executable, "jimu-dse/scripts/visualize_graph.py", "--phase", "all",
            "--dim", str(hw["dim"]), "--hidden", str(hw["hidden"]),
            "--seq-len", str(seq_len), "--num-head", str(hw["num_head"]),
            "-o", str(dag_dir),
        ], timeout=300)
        probe["dag"] = dag
        cluster_file = dag_dir / "dram_clusters.txt"
        if cluster_file.is_file():
            probe["clusters"] = cluster_file.read_text(encoding="utf-8").splitlines()
        observed = {
            int(value) for value in re.findall(r"seq_len=(\d+)", dag.get("stdout", ""))
        }
        config_matches = not observed or observed == {seq_len}
        probe["graph_config_matches"] = config_matches
        probe["graph_context"] = _graph_context(
            dag_dir, config, seq_len, dag, timing_schedule_path,
            cross_layer_path,
        )
        probe["passed"] = (
            probe["passed"] and dag.get("exit_code") == 0 and config_matches
        )
    return probe


FATAL_AGENT_PATTERNS = (
    ("quota_exceeded", (
        r"\bquota\b", r"rate[ _-]?limit", r"too many requests",
        r"insufficient (?:credits|quota)", r"(?:http|status)[^\n]{0,12}429",
    )),
    ("authentication_failed", (
        r"invalid api key", r"authentication (?:failed|required)",
        r"\bunauthorized\b", r"(?:http|status)[^\n]{0,12}(?:401|402|403)",
    )),
    ("provider_or_model_error", (
        r"(?:unknown|invalid|unsupported) (?:provider|model)",
        r"(?:provider|model)[^\n]{0,40}(?:not found|does not exist)",
        r"failed to (?:load|initialize) (?:provider|model)",
    )),
    ("configuration_error", (
        r"(?:config|configuration)[^\n]{0,40}(?:invalid|error|failed)",
        r"failed to parse (?:config|configuration)",
    )),
)


def _fatal_agent_reason(result: dict[str, Any]) -> str | None:
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".lower()
    for reason, patterns in FATAL_AGENT_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            return reason
    return None


def _structured_agent_started(result: dict[str, Any]) -> bool:
    for raw_line in f"{result.get('stdout', '')}\n{result.get('stderr', '')}".splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line).strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        event_name = str(
            event.get("type") or event.get("event") or event.get("kind") or ""
        ).lower()
        if any(token in event_name for token in ("session", "message", "step", "tool")):
            return True
    return False


def classify_agent_start_failure(
    result: dict[str, Any], changed_files: list[str] | None = None,
) -> str | None:
    """Return a fatal startup reason, using target changes only as fallback."""
    if result.get("status") == "agent_unavailable":
        return result.get("failure_reason", "executable_not_found")
    if result.get("spawn_error"):
        return "process_spawn_error"
    fatal = result.get("failure_reason") or _fatal_agent_reason(result)
    if fatal:
        return fatal
    if (
        result.get("exit_code") not in (0, None)
        and not result.get("agent_started")
        and not changed_files
    ):
        return "startup_failed_without_agent_activity"
    if (
        result.get("timed_out")
        and not result.get("agent_started")
        and not changed_files
    ):
        return "startup_timeout"
    return None


def invoke_agent(
    config: dict[str, Any], prompt: str,
    heartbeat: Callable[[float], None] | None = None,
    raw_output_prefix: Path | None = None,
) -> dict[str, Any]:
    backend = config["agent"]["backend"]
    agent_env = _agent_environment()
    executable = shutil.which(backend, path=agent_env.get("PATH"))
    if not executable:
        return {
            "status": "agent_unavailable", "exit_code": None,
            "stdout": "", "stderr": f"{backend} not found",
            "timed_out": False, "spawn_error": True, "agent_started": False,
            "failure_reason": "executable_not_found",
        }
    configured_timeout = int(config["agent"]["timeout_seconds"])
    timeout = configured_timeout or None
    if backend == "opencode":
        version = _run([executable, "--version"], timeout=30)
        if version.get("spawn_error"):
            version.update({
                "status": "agent_start_failed", "agent_started": False,
                "failure_reason": "preflight_failed",
            })
            return version
        help_result = _run([executable, "run", "--help"], timeout=30)
        command = [executable, "run", "--model", config["agent"]["model"]]
        help_text = f"{help_result.get('stdout', '')}\n{help_result.get('stderr', '')}"
        if "--format" in help_text:
            command += ["--format", "json"]
        for item in config["agent"].get("context_files", []):
            command += ["-f", item]
        command += ["--dangerously-skip-permissions", prompt]
    else:
        command = [executable]
        for item in config["skills"]:
            command += ["--skill", str(_repo_path(item["path"], "skill.path"))]
        command += ["-p", prompt]
    result = _run(
        command, timeout=timeout, heartbeat=heartbeat,
        env=agent_env, raw_output_prefix=raw_output_prefix,
    )
    if raw_output_prefix is not None:
        result["stdout_log"] = _display_path(
            Path(f"{raw_output_prefix}.stdout.jsonl")
        )
        result["stderr_log"] = _display_path(
            Path(f"{raw_output_prefix}.stderr.log")
        )
    result["agent_started"] = (
        result["exit_code"] == 0 or _structured_agent_started(result)
    )
    fatal_reason = _fatal_agent_reason(result)
    if fatal_reason:
        result["failure_reason"] = fatal_reason
    result["status"] = (
        "agent_start_failed" if result.get("spawn_error") or fatal_reason
        else "agent_timeout" if result["timed_out"] and result["agent_started"]
        else "agent_failed" if result["exit_code"] != 0
        else "completed"
    )
    return result


def run_gates(
    config: dict[str, Any], changed_files: list[str], probe: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    results = []
    allowed = set(config["target"]["allowed_files"])
    for gate in config["acceptance"]["gates"]:
        gate_started = time.monotonic()
        if progress:
            progress(f"gate {gate['name']} started")
        item = {"name": gate["name"], "type": gate["type"]}
        if gate["type"] == "allowed_files":
            unexpected = sorted(set(changed_files) - allowed)
            item.update({"passed": not unexpected, "unexpected_files": unexpected})
        elif gate["type"] == "build":
            build = build_firmware(config, scoring_sequence_length(config))
            item.update({"passed": build["passed"], "result": build})
        elif gate["type"] == "probe":
            item.update({"passed": probe["passed"]})
        else:
            command = gate["command"].format(
                dim=config["target"]["hardware"]["dim"],
                hidden=config["target"]["hardware"]["hidden"],
                firmware=config["target"]["firmware"],
            )
            result = _run(
                command, int(gate.get("timeout_seconds", 300)), shell=True,
                heartbeat=(
                    lambda elapsed: progress(
                        f"gate {gate['name']} still running, elapsed={elapsed:.0f}s"
                    )
                ) if progress else None,
            )
            success_codes = gate.get("success_codes", [0])
            item.update({"passed": result["exit_code"] in success_codes, "result": result})
            if "test_pass" in config["probe"]["metrics"]:
                probe["metrics"]["test_pass"] = 1 if item["passed"] else 0
        results.append(item)
        if progress:
            progress(
                f"gate {gate['name']} {'passed' if item['passed'] else 'failed'} "
                f"in {time.monotonic() - gate_started:.1f}s"
            )
    return results


def scoring_sequence_length(config: dict[str, Any]) -> int:
    return int(config["probe"].get(
        "scoring_sequence_length", config["target"]["sequence_lengths"][-1]
    ))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    def json_default(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, Path):
            return str(value)
        raise TypeError(
            f"Object of type {value.__class__.__name__} is not JSON serializable"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def _recover_deleted_run_directory(
    run_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    best_bytes: bytes,
    baseline_probe: dict[str, Any] | None,
) -> None:
    """Recreate the minimum resumable evidence after run artifacts are removed."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(run_dir / "resolved-config.yaml", config)
    if baseline_probe is not None:
        _write_json(run_dir / "baseline-probe.json", baseline_probe)
    for record in summary.get("iterations", []):
        _write_json(run_dir / f"iteration-{record['iteration']}.json", record)
    (run_dir / "candidate_best.c").write_bytes(best_bytes)
    _write_json(
        run_dir / "artifact-recovery.json",
        {
            "status": "recovered_after_run_directory_deletion",
            "lost_artifacts": [
                "candidate files other than candidate_best.c",
                "diffs",
                "prompts",
                "probe details other than the baseline",
                "graphs",
            ],
        },
    )


def _source_snapshot() -> dict[str, bytes]:
    """Snapshot agent-editable source/config files for scope enforcement."""
    roots = [
        REPO_ROOT / "firmware", REPO_ROOT / "emulator", REPO_ROOT / "iss",
        REPO_ROOT / "kernels", REPO_ROOT / "scripts", REPO_ROOT / "tests",
        REPO_ROOT / "jimu-dse" / "scripts", REPO_ROOT / "jimu-dse" / "goals",
        REPO_ROOT / "jimu-dse" / "docs", REPO_ROOT / "jimu-dse" / "timing",
    ]
    suffixes = {
        ".c", ".h", ".S", ".py", ".sh", ".md", ".yaml", ".yml", ".json",
        ".cfg",
    }
    files = [
        REPO_ROOT / "Makefile", REPO_ROOT / "README.md",
        REPO_ROOT / "requirements.txt", REPO_ROOT / "requirements-timing.txt",
    ]
    for root in roots:
        if root.is_dir():
            files.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix in suffixes
                and "__pycache__" not in path.parts
                and not any(part.startswith("build") for part in path.parts)
            )
    return {
        path.relative_to(REPO_ROOT).as_posix(): path.read_bytes()
        for path in files if path.is_file()
    }


def _snapshot_changes(before: dict[str, bytes]) -> tuple[list[str], dict[str, bytes]]:
    after = _source_snapshot()
    changed = sorted(
        path for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
    return changed, after


def _restore_unauthorized(
    before: dict[str, bytes], after: dict[str, bytes], allowed: set[str],
) -> None:
    for relative in (set(before) | set(after)) - allowed:
        if before.get(relative) == after.get(relative):
            continue
        path = REPO_ROOT / relative
        if relative in before:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(before[relative])
        elif path.is_file():
            path.unlink()


def _summary_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Closed-loop run: {summary['goal']}", "",
        f"- Status: `{summary['status']}`",
        f"- Stop reason: `{summary['stop_reason']}`",
        f"- Loop mode: `{summary.get('loop_mode', 'goal_driven')}`",
        f"- Completed iterations: `{len(summary.get('iterations', []))}`",
        f"- Next iteration: `{summary.get('next_iteration')}`",
        f"- Best iteration: `{summary.get('best_iteration')}`",
        f"- Best score: `{summary.get('best_score', 0):.6f}`",
        f"- Results: `{summary['run_dir']}`", "",
        "## Iterations", "",
        "| Iteration | Status | Gates | Score | Promoted |",
        "|---:|---|---|---:|---|",
    ]
    for item in summary["iterations"]:
        gates = (
            "SKIPPED" if "gates" not in item
            else "PASS" if item.get("gates_passed") else "FAIL"
        )
        lines.append(
            f"| {item['iteration']} | {item['status']} | {gates} | "
            f"{item.get('score', 0):.6f} | {item.get('promoted', False)} |"
        )
    if summary.get("interruptions"):
        latest = summary["interruptions"][-1]
        interruption_heading = (
            "## Latest interruption"
            if summary.get("status") == "interrupted"
            else "## Earlier interruption (recovered)"
        )
        lines += [
            "", interruption_heading, "",
            f"- Reason: `{latest.get('reason')}`",
            f"- Iteration: `{latest.get('iteration')}`",
            f"- Exit code: `{latest.get('exit_code')}`",
            f"- Message: `{latest.get('message', '')}`",
        ]
    baseline = summary.get("baseline_metrics", {})
    best = summary.get("best_metrics", {})
    if "estimated_time" in baseline:
        base_time = float(baseline["estimated_time"])
        best_time = float(best.get("estimated_time", base_time))
        improvement = 0.0 if base_time == 0 else (base_time - best_time) * 100.0 / abs(base_time)
        model = summary.get("cost_model", {})
        lines += [
            "", "## Estimated cost", "",
            "> This is a weighted cost estimate, not cycle-accurate hardware time.",
            "",
            f"- Formula: `memory_access_count × {model.get('memory_weight')} + "
            f"register_access_count × {model.get('register_weight')}`",
            f"- Scoring sequence length: `{summary.get('scoring_sequence_length')}`",
            "",
            "| Metric | Baseline | Best |",
            "|---|---:|---:|",
            f"| Memory reads | {baseline.get('memory_read_count', 0)} | {best.get('memory_read_count', 0)} |",
            f"| Memory writes | {baseline.get('memory_write_count', 0)} | {best.get('memory_write_count', 0)} |",
            f"| Register reads | {baseline.get('register_read_count', 0)} | {best.get('register_read_count', 0)} |",
            f"| Register writes | {baseline.get('register_write_count', 0)} | {best.get('register_write_count', 0)} |",
            f"| Estimated time | {base_time:g} | {best_time:g} |",
            f"| Improvement | — | {improvement:.2f}% |",
        ]
    if "predicted_npu_cycles" in baseline:
        base_cycles = float(baseline["predicted_npu_cycles"])
        best_cycles = float(best.get("predicted_npu_cycles", base_cycles))
        improvement = (
            0.0 if base_cycles == 0
            else (base_cycles - best_cycles) * 100.0 / abs(base_cycles)
        )
        lines += [
            "", "## SCALE-Sim timing estimate", "",
            "> SCALE-Sim v2 models MVU GEMMs; trace-derived costs model this "
            "NPU's explicit memory and auxiliary instructions.",
            "",
            "| Metric | Baseline | Best |",
            "|---|---:|---:|",
            f"| SCALE-Sim GEMM layers | {baseline.get('scalesim_layer_count', 0)} | {best.get('scalesim_layer_count', 0)} |",
            f"| SCALE-Sim MVU compute cycles | {baseline.get('scalesim_compute_cycles', 0)} | {best.get('scalesim_compute_cycles', 0)} |",
            f"| SCALE-Sim diagnostic stall cycles | {baseline.get('scalesim_stall_cycles', 0)} | {best.get('scalesim_stall_cycles', 0)} |",
            f"| Trace memory cycles | {baseline.get('trace_memory_cycles', 0)} | {best.get('trace_memory_cycles', 0)} |",
            f"| Auxiliary instruction cycles | {baseline.get('auxiliary_cycles', 0)} | {best.get('auxiliary_cycles', 0)} |",
            f"| Predicted NPU cycles | {base_cycles:g} | {best_cycles:g} |",
            f"| Improvement | — | {improvement:.2f}% |",
        ]
    if "rtl_predicted_npu_cycles" in baseline:
        base_rtl = float(baseline["rtl_predicted_npu_cycles"])
        best_rtl = float(best.get("rtl_predicted_npu_cycles", base_rtl))
        rtl_improvement = (
            0.0 if base_rtl == 0
            else (base_rtl - best_rtl) * 100.0 / abs(base_rtl)
        )
        lines += [
            "", "## Verilator RTL resource schedule", "",
            "> This is the cycle count from the synthesizable Jimu command/control "
            "RTL under the versioned profile; numerical values are gated by the "
            "functional emulator.",
            "",
            "| Metric | Baseline | Best |",
            "|---|---:|---:|",
            f"| RTL predicted cycles | {base_rtl:g} | {best_rtl:g} |",
            f"| RTL fully idle cycles | {baseline.get('rtl_idle_cycles', 0)} | {best.get('rtl_idle_cycles', 0)} |",
            f"| Retirement tail cycles | {baseline.get('rtl_retirement_tail_cycles', 0)} | {best.get('rtl_retirement_tail_cycles', 0)} |",
            f"| Net parallelism savings | {baseline.get('net_parallelism_savings_cycles', baseline.get('overlap_saved_cycles', 0))} | {best.get('net_parallelism_savings_cycles', best.get('overlap_saved_cycles', 0))} |",
            f"| Gross overlapped work | {baseline.get('gross_overlap_cycles', 0)} | {best.get('gross_overlap_cycles', 0)} |",
            f"| Scheduler idle holes | {baseline.get('scheduler_idle_hole_cycles', 0)} | {best.get('scheduler_idle_hole_cycles', 0)} |",
            f"| Memory/compute overlap cycles | {baseline.get('memory_compute_overlap_cycles', 0)} | {best.get('memory_compute_overlap_cycles', 0)} |",
            f"| Logical DRAM payload bytes | {baseline.get('logical_dram_payload_bytes', 0)} | {best.get('logical_dram_payload_bytes', 0)} |",
            f"| Modeled DRAM transaction bytes | {baseline.get('modeled_dram_transaction_bytes', 0)} | {best.get('modeled_dram_transaction_bytes', 0)} |",
            f"| Maximum concurrent operations | {baseline.get('max_concurrent_ops', 0)} | {best.get('max_concurrent_ops', 0)} |",
            f"| Load utilization | {float(baseline.get('load_utilization', 0)):.2%} | {float(best.get('load_utilization', 0)):.2%} |",
            f"| Store utilization | {float(baseline.get('store_utilization', 0)):.2%} | {float(best.get('store_utilization', 0)):.2%} |",
            f"| MVU utilization | {float(baseline.get('mvu_utilization', 0)):.2%} | {float(best.get('mvu_utilization', 0)):.2%} |",
            f"| Vector utilization | {float(baseline.get('vector_utilization', 0)):.2%} | {float(best.get('vector_utilization', 0)):.2%} |",
            f"| Dependency stall counter | {baseline.get('rtl_counter_dependency_stall_cycles', 0)} | {best.get('rtl_counter_dependency_stall_cycles', 0)} |",
            f"| DRAM stall counter | {baseline.get('rtl_counter_dram_stall_cycles', 0)} | {best.get('rtl_counter_dram_stall_cycles', 0)} |",
            f"| SRAM bank stall counter | {baseline.get('rtl_counter_bank_stall_cycles', 0)} | {best.get('rtl_counter_bank_stall_cycles', 0)} |",
            f"| RTL-cycle improvement | — | {rtl_improvement:.2f}% |",
            "",
            "> Stall counters are non-additive pressure indicators; do not sum "
            "them to explain the makespan delta.",
        ]
    if (
        "parallel_predicted_npu_cycles" in baseline
        and "rtl_predicted_npu_cycles" not in baseline
    ):
        base_parallel = float(baseline["parallel_predicted_npu_cycles"])
        best_parallel = float(best.get(
            "parallel_predicted_npu_cycles", base_parallel
        ))
        parallel_improvement = (
            0.0 if base_parallel == 0
            else (base_parallel - best_parallel) * 100.0 / abs(base_parallel)
        )
        lines += [
            "", "## Parallel resource schedule", "",
            "> This is a configurable dependency/resource timing model, not an "
            "RTL- or silicon-measured cycle count.",
            "",
            "| Metric | Baseline | Best |",
            "|---|---:|---:|",
            f"| Parallel predicted cycles | {base_parallel:g} | {best_parallel:g} |",
            f"| Saved by overlap | {baseline.get('overlap_saved_cycles', 0)} | {best.get('overlap_saved_cycles', 0)} |",
            f"| Memory/compute overlap cycles | {baseline.get('memory_compute_overlap_cycles', 0)} | {best.get('memory_compute_overlap_cycles', 0)} |",
            f"| Maximum concurrent operations | {baseline.get('max_concurrent_ops', 0)} | {best.get('max_concurrent_ops', 0)} |",
            f"| DRAM bus utilization | {float(baseline.get('dram_bus_utilization', 0)):.2%} | {float(best.get('dram_bus_utilization', 0)):.2%} |",
            f"| MVU utilization | {float(baseline.get('mvu_utilization', 0)):.2%} | {float(best.get('mvu_utilization', 0)):.2%} |",
            f"| VMM utilization | {float(baseline.get('vmm_utilization', 0)):.2%} | {float(best.get('vmm_utilization', 0)):.2%} |",
            f"| MMM utilization | {float(baseline.get('mmm_utilization', 0)):.2%} | {float(best.get('mmm_utilization', 0)):.2%} |",
            f"| SPU utilization | {float(baseline.get('spu_utilization', 0)):.2%} | {float(best.get('spu_utilization', 0)):.2%} |",
            f"| Parallel-cycle improvement | — | {parallel_improvement:.2f}% |",
        ]
    lines += ["", "## Reproduce", "", f"`{summary['reproduce_command']}`", ""]
    return "\n".join(lines)


def _write_run_checkpoint(
    run_dir: Path, config: dict[str, Any], summary: dict[str, Any], best_bytes: bytes,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    _write_yaml(run_dir / "resolved-config.yaml", public_config)
    (run_dir / "candidate_best.c").write_bytes(best_bytes)
    _write_json(run_dir / "run-summary.json", summary)
    (run_dir / "report.md").write_text(_summary_report(summary), encoding="utf-8")


def _error_excerpt(agent_result: dict[str, Any]) -> str:
    text = agent_result.get("stderr") or agent_result.get("stdout") or "no diagnostic output"
    clean = re.sub(r"\x1b\[[0-9;]*m", "", str(text))
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    return " | ".join(lines[-3:])[-1000:]


def execute_run(
    config: dict[str, Any], resume: str | None = None,
    results_root: Path | None = None,
    progress: ProgressCallback | None = _progress,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    log = progress or (lambda _message: None)
    target = _repo_path(config["target"]["firmware"], "target.firmware")
    original = target.read_bytes()
    current_profile_fingerprints = timing_profile_fingerprints(config)

    if resume:
        resume_path = Path(resume)
        run_dir = (
            resume_path.resolve()
            if resume_path.is_absolute()
            else (REPO_ROOT / resume_path).resolve()
        )
        if not run_dir.is_dir():
            raise ConfigError(f"resume directory not found: {run_dir}")
        prior_config = _read_yaml(run_dir / "resolved-config.yaml")
        if config_fingerprint(prior_config) != config_fingerprint(config):
            raise ConfigError("resume configuration is incompatible with the selected goal")
        summary_path = run_dir / "run-summary.json"
        candidate_path = run_dir / "candidate_best.c"
        baseline_probe_path = run_dir / "baseline-probe.json"
        if not summary_path.is_file() or not candidate_path.is_file():
            raise ConfigError("resume directory lacks run-summary.json or candidate_best.c")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        prior_profile_fingerprints = summary.get("timing_profile_fingerprints")
        if (
            prior_profile_fingerprints is not None
            and prior_profile_fingerprints != current_profile_fingerprints
        ):
            raise ConfigError(
                "resume timing profile contents differ from the original run"
            )
        if summary.get("status") == "completed":
            raise ConfigError("completed runs cannot be resumed")
        if not baseline_probe_path.is_file():
            raise ConfigError("resume directory has no baseline-probe.json")
        baseline_probe = json.loads(baseline_probe_path.read_text(encoding="utf-8"))
        baseline_metrics = summary.get("baseline_metrics")
        if not isinstance(baseline_metrics, dict):
            raise ConfigError("resume summary has no baseline_metrics")
        best_bytes = candidate_path.read_bytes()
        best_score = float(summary.get("best_score", 0.0))
        no_improvement = int(summary.get("no_improvement", 0))
        start_iteration = int(
            summary.get("next_iteration", len(summary.get("iterations", [])) + 1)
        )
        summary.update({"status": "running", "stop_reason": ""})
    else:
        run_tag = dt.datetime.now().strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        run_dir = (results_root or RESULTS_DIR) / run_tag
        run_dir.mkdir(parents=True, exist_ok=False)
        baseline_probe = None
        baseline_metrics = {}
        # Optimise the firmware the caller actually supplied.  The committed
        # baseline remains a versioned reference, but silently replacing a
        # dirty/current target would optimise the wrong program and lose the
        # user's starting point from candidate artifacts.
        best_bytes = original
        best_score = 0.0
        no_improvement = 0
        start_iteration = 1

    try:
        run_dir_display = str(run_dir.relative_to(REPO_ROOT))
    except ValueError:
        run_dir_display = str(run_dir)

    target.write_bytes(best_bytes)
    if not resume:
        summary = {
            "schema_version": 2,
            "goal": config["name"],
            "config_fingerprint": config_fingerprint(config),
            "timing_profile_fingerprints": current_profile_fingerprints,
            "run_dir": run_dir_display,
            "status": "running",
            "stop_reason": "",
            "loop_mode": config["loop"].get("mode", "goal_driven"),
            "max_iterations": int(config["loop"]["max_iterations"]),
            "mode_overridden": bool(config.get("_mode_overridden")),
            "best_iteration": None,
            "best_score": 0.0,
            "best_metrics": {},
            "iterations": [],
            "interruptions": [],
            "next_iteration": 1,
            "no_improvement": 0,
            "scoring_sequence_length": scoring_sequence_length(config),
            "cost_model": config["probe"].get("cost_model"),
            "reproduce_command": (
                "python3 jimu-dse/scripts/closed_loop.py run "
                f"--config {run_dir_display}/resolved-config.yaml "
                f"--resume {run_dir_display}"
            ),
        }
    else:
        summary.setdefault("interruptions", [])
        summary.setdefault(
            "timing_profile_fingerprints", current_profile_fingerprints
        )
        summary["loop_mode"] = config["loop"].get("mode", "goal_driven")
        summary["mode_overridden"] = bool(
            summary.get("mode_overridden") or config.get("_mode_overridden")
        )

    _write_run_checkpoint(run_dir, config, summary, best_bytes)
    timeout_value = int(config["agent"]["timeout_seconds"])
    timeout_label = "disabled" if timeout_value == 0 else f"{timeout_value}s"
    log(
        f"run {'resumed' if resume else 'started'}: goal={config['name']}, "
        f"mode={config['loop'].get('mode', 'goal_driven')}, "
        f"iterations={start_iteration}-{config['loop']['max_iterations']}, "
        f"agent={config['agent']['backend']}, model={config['agent']['model']}, "
        f"agent_timeout={timeout_label}"
    )
    log(f"artifacts: {run_dir_display}")
    try:
        if not resume:
            baseline_started = time.monotonic()
            log(
                "baseline probe started: "
                f"seq={scoring_sequence_length(config)}"
            )
            baseline_probe = probe_firmware(
                config, scoring_sequence_length(config), run_dir / "baseline",
            )
            _write_json(run_dir / "baseline-probe.json", baseline_probe)
            if not baseline_probe["passed"]:
                summary.update({"status": "failed", "stop_reason": "baseline_probe_failed"})
                log("baseline probe failed; inspect baseline-probe.json and report.md")
                return summary
            baseline_metrics = baseline_probe["metrics"]
            log(
                "baseline probe passed in "
                f"{time.monotonic() - baseline_started:.1f}s: "
                f"{_metric_snapshot(config, baseline_metrics)}"
            )
            summary["baseline_metrics"] = baseline_metrics
            summary["best_metrics"] = copy.deepcopy(baseline_metrics)
            best_bytes = target.read_bytes()
            _write_run_checkpoint(run_dir, config, summary, best_bytes)

        for iteration in range(
            start_iteration, config["loop"]["max_iterations"] + 1
        ):
            iteration_label = (
                f"iteration {iteration}/{config['loop']['max_iterations']}"
            )
            target.write_bytes(best_bytes)
            before = target.read_bytes()
            source_before = _source_snapshot()
            pre_probe_started = time.monotonic()
            log(f"{iteration_label}: pre-agent probe started")
            current_probe = probe_firmware(
                config, scoring_sequence_length(config),
                run_dir / f"pre-iteration-{iteration}",
            )
            if not current_probe["passed"]:
                summary.update({
                    "status": "failed",
                    "stop_reason": "pre_agent_probe_failed",
                    "next_iteration": iteration,
                })
                _write_json(
                    run_dir / f"pre-probe-{iteration}.json", current_probe
                )
                log(f"{iteration_label}: pre-agent probe failed")
                return summary
            log(
                f"{iteration_label}: pre-agent probe passed in "
                f"{time.monotonic() - pre_probe_started:.1f}s: "
                f"{_metric_snapshot(config, current_probe['metrics'])}"
            )
            prompt = render_prompt(
                config, iteration, current_probe["metrics"],
                current_probe["clusters"], current_probe.get("graph_context"),
                baseline_metrics, summary.get("iterations"),
            )
            if config["artifacts"].get("save_prompts"):
                (run_dir / f"prompt-{iteration}.txt").write_text(prompt, encoding="utf-8")
            log(f"{iteration_label}: agent started")
            agent_result = invoke_agent(
                config, prompt,
                (lambda elapsed: log(
                    f"{iteration_label}: agent still running, "
                    f"elapsed={elapsed:.0f}s"
                )) if progress else None,
                run_dir / f"agent-{iteration}",
            )
            changed, source_after = _snapshot_changes(source_before)
            log(
                f"{iteration_label}: agent finished with "
                f"status={agent_result.get('status')}, "
                f"elapsed={float(agent_result.get('duration_seconds', 0)):.1f}s, "
                f"changed_files={len(changed)}"
            )
            _restore_unauthorized(
                source_before, source_after, set(config["target"]["allowed_files"])
            )
            if not run_dir.is_dir():
                target.write_bytes(best_bytes)
                record = {
                    "iteration": iteration,
                    "status": "infrastructure_error",
                    "agent": agent_result,
                    "changed_files": changed,
                    "promoted": False,
                    "gates_passed": False,
                    "score": 0.0,
                    "error": "active run directory was deleted while the agent was running",
                }
                summary["iterations"].append(record)
                summary.update({
                    "status": "failed",
                    "stop_reason": "run_artifacts_lost",
                    "artifact_recovery": {
                        "recreated": True,
                        "best_candidate_preserved": True,
                    },
                })
                _recover_deleted_run_directory(
                    run_dir, config, summary, best_bytes, baseline_probe
                )
                return summary
            startup_reason = classify_agent_start_failure(agent_result, changed)
            if startup_reason:
                if agent_result.get("status") != "agent_unavailable":
                    agent_result["status"] = "agent_start_failed"
                agent_result["failure_reason"] = startup_reason
                target.write_bytes(best_bytes)
                failure = {
                    "timestamp": dt.datetime.now().isoformat(),
                    "iteration": iteration,
                    "reason": startup_reason,
                    "backend": config["agent"]["backend"],
                    "model": config["agent"]["model"],
                    "exit_code": agent_result.get("exit_code"),
                    "timed_out": agent_result.get("timed_out", False),
                    "message": _error_excerpt(agent_result),
                    "agent": agent_result,
                }
                summary["interruptions"].append(failure)
                summary.update({
                    "status": "interrupted",
                    "stop_reason": "agent_start_failed",
                    "next_iteration": iteration,
                    "no_improvement": no_improvement,
                })
                stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                _write_json(
                    run_dir / f"agent-start-failure-{iteration}-{stamp}.json",
                    failure,
                )
                _write_run_checkpoint(run_dir, config, summary, best_bytes)
                log(
                    f"{iteration_label}: agent startup interrupted the run: "
                    f"{startup_reason}"
                )
                return summary
            record: dict[str, Any] = {
                "iteration": iteration, "status": agent_result["status"],
                "agent": agent_result, "changed_files": changed, "promoted": False,
                "gates_passed": False, "score": 0.0,
            }
            target_exists = target.is_file()
            current_candidate = target.read_bytes() if target_exists else b""
            target_changed = not target_exists or current_candidate != before
            evaluate_timeout_candidate = (
                agent_result["status"] == "agent_timeout"
                and config["loop"].get("evaluate_timeout_candidate", False)
                and target_changed
            )
            record["timeout_candidate_evaluated"] = evaluate_timeout_candidate
            if (
                agent_result["status"] != "completed"
                and not evaluate_timeout_candidate
            ) or not changed:
                record["status"] = agent_result["status"] if changed else (
                    "no_change" if agent_result["status"] == "completed" else agent_result["status"]
                )
                target.write_bytes(best_bytes)
                no_improvement += 1
                log(
                    f"{iteration_label}: no candidate validation; "
                    f"status={record['status']}"
                )
            else:
                candidate = current_candidate
                record["candidate_target_missing"] = not target_exists
                candidate_name = (
                    f"candidate-timeout-{iteration}.c"
                    if evaluate_timeout_candidate else f"candidate-{iteration}.c"
                )
                if (
                    config["artifacts"].get("save_candidates")
                    or evaluate_timeout_candidate
                ):
                    (run_dir / candidate_name).write_bytes(candidate)
                validation_started = time.monotonic()
                log(f"{iteration_label}: candidate probe and gates started")
                candidate_probe = probe_firmware(
                    config, scoring_sequence_length(config),
                    run_dir / f"iteration-{iteration}",
                )
                log(
                    f"{iteration_label}: candidate probe "
                    f"{'passed' if candidate_probe['passed'] else 'failed'}; "
                    f"{_metric_snapshot(config, candidate_probe.get('metrics', {}))}"
                )
                gates = run_gates(
                    config, changed, candidate_probe,
                    (lambda message: log(f"{iteration_label}: {message}"))
                    if progress else None,
                )
                if config["artifacts"].get("save_probes"):
                    _write_json(run_dir / f"probe-{iteration}.json", candidate_probe)
                record["probe"] = candidate_probe
                record["attempt_feedback"] = _candidate_feedback(
                    current_probe["metrics"], candidate_probe.get("metrics", {})
                )
                record["gates"] = gates
                record["gates_passed"] = all(x["passed"] for x in gates)
                gate_summary = ", ".join(
                    f"{gate['name']}={'PASS' if gate['passed'] else 'FAIL'}"
                    for gate in gates
                )
                log(
                    f"{iteration_label}: validation finished in "
                    f"{time.monotonic() - validation_started:.1f}s; {gate_summary}"
                )
                if config["artifacts"].get("save_diffs"):
                    diff = "".join(difflib.unified_diff(
                        before.decode(errors="replace").splitlines(True),
                        candidate.decode(errors="replace").splitlines(True),
                        fromfile="best", tofile=f"candidate-{iteration}",
                    ))
                    (run_dir / f"diff-{iteration}.patch").write_text(diff, encoding="utf-8")
                if record["gates_passed"]:
                    score, details = score_metrics(
                        baseline_metrics, candidate_probe["metrics"],
                        config["acceptance"]["score"],
                    )
                    record.update({
                        "score": score,
                        "score_details": details,
                        "status": (
                            "accepted_after_timeout"
                            if evaluate_timeout_candidate else "accepted"
                        ),
                    })
                    if score >= best_score + config["loop"]["min_score_delta"]:
                        best_score, best_bytes = score, candidate
                        record["promoted"] = True
                        summary["best_iteration"] = iteration
                        summary["best_score"] = score
                        summary["best_metrics"] = copy.deepcopy(candidate_probe["metrics"])
                        no_improvement = 0
                        log(
                            f"{iteration_label}: promoted; score={score:.6f}, "
                            f"{_metric_snapshot(config, candidate_probe['metrics'])}"
                        )
                    else:
                        record["status"] = (
                            "not_improved_after_timeout"
                            if evaluate_timeout_candidate else "not_improved"
                        )
                        target.write_bytes(best_bytes)
                        no_improvement += 1
                        log(
                            f"{iteration_label}: not promoted; score={score:.6f}, "
                            f"best_score={best_score:.6f}"
                        )
                else:
                    record["status"] = (
                        "timeout_candidate_gate_failed"
                        if evaluate_timeout_candidate else "gate_failed"
                    )
                    target.write_bytes(best_bytes)
                    no_improvement += 1
                    failed_gates = ", ".join(
                        gate["name"] for gate in gates if not gate["passed"]
                    )
                    log(
                        f"{iteration_label}: candidate rejected by gates: "
                        f"{failed_gates}"
                    )
            summary["iterations"].append(record)
            _write_json(run_dir / f"iteration-{iteration}.json", record)
            summary["next_iteration"] = iteration + 1
            summary["no_improvement"] = no_improvement
            _write_run_checkpoint(run_dir, config, summary, best_bytes)
            log(
                f"{iteration_label}: checkpoint saved; "
                f"status={record['status']}, best_score={best_score:.6f}, "
                f"consecutive_no_promotion={no_improvement}"
            )
            if config["loop"].get("mode", "goal_driven") == "goal_driven":
                target_score = config["loop"].get("target_score")
                if target_score is not None and best_score >= target_score:
                    summary["stop_reason"] = "target_score_reached"
                    log(f"stopping early: target score {target_score} reached")
                    break
                if no_improvement >= config["loop"]["max_no_improvement"]:
                    summary["stop_reason"] = "no_improvement_limit"
                    log(
                        "stopping early: consecutive no-promotion limit "
                        f"{config['loop']['max_no_improvement']} reached"
                    )
                    break
        else:
            summary["stop_reason"] = "max_iterations"

        summary["status"] = "completed"
        return summary
    except FileNotFoundError:
        if run_dir.is_dir():
            raise
        summary.update({
            "status": "failed",
            "stop_reason": "run_artifacts_lost",
            "artifact_recovery": {
                "recreated": True,
                "best_candidate_preserved": True,
            },
        })
        _recover_deleted_run_directory(
            run_dir, config, summary, best_bytes, baseline_probe
        )
        return summary
    finally:
        target.write_bytes(original)
        if not summary["stop_reason"]:
            summary["stop_reason"] = "internal_error"
            summary["status"] = "failed"
        _write_run_checkpoint(run_dir, config, summary, best_bytes)
        log(
            f"run finished: status={summary['status']}, "
            f"stop_reason={summary['stop_reason']}, "
            f"best_iteration={summary.get('best_iteration')}, "
            f"best_score={float(summary.get('best_score', 0)):.6f}"
        )


def load_experiment_config(path: str) -> dict[str, Any]:
    experiment_path = Path(path).resolve()
    data = _read_yaml(experiment_path)
    allowed = {
        "schema_version", "name", "goal", "target_score", "repetitions",
        "full_iterations", "arms",
    }
    _unknown_keys(data, allowed, "experiment")
    if data.get("schema_version") != 1:
        raise ConfigError("experiment schema_version must be 1")
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise ConfigError("experiment.name must be a non-empty string")
    if not isinstance(data.get("goal"), str):
        raise ConfigError("experiment.goal must name a built-in goal")
    load_config(data["goal"])
    if (
        not isinstance(data.get("target_score"), (int, float))
        or data["target_score"] <= 0
    ):
        raise ConfigError("experiment.target_score must be positive")
    if not isinstance(data.get("repetitions"), int) or data["repetitions"] < 1:
        raise ConfigError("experiment.repetitions must be a positive integer")
    if not isinstance(data.get("full_iterations", True), bool):
        raise ConfigError("experiment.full_iterations must be boolean")
    arms = data.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"control", "legacy", "treatment"}:
        raise ConfigError(
            "experiment.arms must contain exactly control, legacy, and treatment"
        )
    base = load_config(data["goal"])
    for arm_name, skills in arms.items():
        if not isinstance(skills, list) or not skills:
            raise ConfigError(f"experiment arm {arm_name} must list skills")
        candidate = copy.deepcopy(base)
        candidate["skills"] = skills
        validate_config(candidate)
    data["_config_path"] = str(experiment_path)
    return data


def _skill_hashes(skills: list[dict[str, str]]) -> dict[str, str]:
    return {
        item["name"]: hashlib.sha256(
            _repo_path(item["path"], "skill.path").read_bytes()
        ).hexdigest()
        for item in skills
    }


def _run_statistics(
    runs: list[dict[str, Any]], target_score: float,
) -> dict[str, Any]:
    scores = [float(item.get("best_score", 0.0)) for item in runs]
    iteration_records = [
        record for run in runs for record in run.get("iterations", [])
    ]
    gated_records = [
        record for record in iteration_records if "gates" in record
    ]
    first_improvements = [
        min(
            (
                int(record["iteration"])
                for record in run.get("iterations", [])
                if record.get("promoted")
            ),
            default=None,
        )
        for run in runs
    ]
    cycle_values = [
        float(run.get("best_metrics", {}).get("predicted_npu_cycles"))
        for run in runs
        if "predicted_npu_cycles" in run.get("best_metrics", {})
    ]
    parallel_cycle_values = [
        float(run.get("best_metrics", {}).get("parallel_predicted_npu_cycles"))
        for run in runs
        if "parallel_predicted_npu_cycles" in run.get("best_metrics", {})
    ]
    component_changes: dict[str, list[float]] = {
        name: [] for name in (
            "scalesim_compute_cycles", "trace_memory_cycles", "auxiliary_cycles",
            "predicted_npu_cycles", "parallel_predicted_npu_cycles",
            "memory_compute_overlap_cycles",
        )
    }
    for run in runs:
        baseline = run.get("baseline_metrics", {})
        best = run.get("best_metrics", {})
        for name, changes in component_changes.items():
            if name in baseline and name in best:
                changes.append(float(baseline[name]) - float(best[name]))
    hypotheses = 0
    for record in iteration_records:
        agent = record.get("agent", {})
        output = f"{agent.get('stdout', '')}\n{agent.get('stderr', '')}"
        if (
            "<dataflow_hypotheses>" in output
            and "</dataflow_hypotheses>" in output
        ):
            hypotheses += 1
    return {
        "runs": len(runs),
        "completed_runs": sum(run.get("status") == "completed" for run in runs),
        "target_hits": sum(score >= target_score for score in scores),
        "target_hit_rate": (
            sum(score >= target_score for score in scores) / len(runs) if runs else 0.0
        ),
        "best_score": max(scores, default=0.0),
        "median_score": statistics.median(scores) if scores else 0.0,
        "median_predicted_npu_cycles": (
            statistics.median(cycle_values) if cycle_values else None
        ),
        "median_parallel_predicted_npu_cycles": (
            statistics.median(parallel_cycle_values)
            if parallel_cycle_values else None
        ),
        "gate_pass_rate": (
            sum(bool(record.get("gates_passed")) for record in gated_records)
            / len(gated_records)
            if gated_records else 0.0
        ),
        "gate_evaluated_candidates": len(gated_records),
        "agent_start_failures": sum(
            len(run.get("interruptions", [])) for run in runs
        ),
        "agent_timeouts": sum(
            record.get("status") == "agent_timeout" for record in iteration_records
        ),
        "agent_failures": sum(
            record.get("status") == "agent_failed" for record in iteration_records
        ),
        "hypothesis_adherence_rate": (
            hypotheses / len(iteration_records) if iteration_records else 0.0
        ),
        "first_improvement_iterations": first_improvements,
        "median_component_cycle_reduction": {
            name: statistics.median(changes) if changes else None
            for name, changes in component_changes.items()
        },
    }


def _experiment_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Skill evaluation: {summary['name']}", "",
        f"- Status: `{summary['status']}`",
        f"- Target score: `{summary['target_score']:.4f}`",
        f"- Repetitions: `{summary['repetitions']}`",
        f"- Loop mode: `{summary['loop_mode']}`",
        f"- Results: `{summary['experiment_dir']}`", "",
        "## Arm comparison", "",
        "| Arm | Runs | Target hits | Hit rate | Median score | Best score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("control", "legacy", "treatment"):
        item = summary.get("statistics", {}).get(arm, {})
        lines.append(
            f"| {arm} | {item.get('runs', 0)} | {item.get('target_hits', 0)} | "
            f"{item.get('target_hit_rate', 0):.1%} | "
            f"{item.get('median_score', 0):.4f} | {item.get('best_score', 0):.4f} |"
        )
    lines += [
        "", "## Median cycle reduction", "",
        "| Arm | Compute | Memory | Auxiliary | Legacy total | Parallel total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("control", "legacy", "treatment"):
        changes = (
            summary.get("statistics", {})
            .get(arm, {})
            .get("median_component_cycle_reduction", {})
        )
        values = [
            changes.get(name) for name in (
                "scalesim_compute_cycles", "trace_memory_cycles",
                "auxiliary_cycles", "predicted_npu_cycles",
                "parallel_predicted_npu_cycles",
            )
        ]
        rendered = ["—" if value is None else f"{value:g}" for value in values]
        lines.append(f"| {arm} | {' | '.join(rendered)} |")
    acceptance = summary.get("acceptance", {})
    if acceptance:
        lines += [
            "", "## Acceptance", "",
            f"- Treatment reaches target in at least two runs: "
            f"`{acceptance.get('treatment_two_of_three')}`",
            f"- Treatment improves over control: "
            f"`{acceptance.get('treatment_beats_control')}`",
            f"- Overall skill criterion: `{acceptance.get('passed')}`",
        ]
    if summary.get("pending"):
        lines += [
            "", "## Pending work", "",
            f"- Replicate: `{summary['pending'].get('replicate')}`",
            f"- Arm: `{summary['pending'].get('arm')}`",
            f"- Run: `{summary['pending'].get('run_dir')}`",
        ]
    lines += [
        "", "## Reproduce", "",
        f"`{summary.get('reproduce_command', '')}`",
    ]
    return "\n".join(lines) + "\n"


def _write_experiment_checkpoint(
    experiment_dir: Path, config: dict[str, Any], summary: dict[str, Any],
) -> None:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    _write_yaml(experiment_dir / "experiment-config.yaml", public)
    _write_json(experiment_dir / "summary.json", summary)
    (experiment_dir / "report.md").write_text(
        _experiment_report(summary), encoding="utf-8"
    )


def _finalize_experiment_statistics(summary: dict[str, Any]) -> None:
    target = float(summary["target_score"])
    by_arm = {
        arm: [run for run in summary["runs"] if run["arm"] == arm]
        for arm in ("control", "legacy", "treatment")
    }
    summary["statistics"] = {
        arm: _run_statistics([item["summary"] for item in runs], target)
        for arm, runs in by_arm.items()
    }
    treatment = summary["statistics"]["treatment"]
    control = summary["statistics"]["control"]
    required_hits = (2 * int(summary["repetitions"]) + 2) // 3
    two_of_three = treatment["target_hits"] >= required_hits
    beats_control = (
        treatment["target_hit_rate"] > control["target_hit_rate"]
        or (
            treatment["target_hit_rate"] == control["target_hit_rate"]
            and treatment["median_score"] >= control["median_score"] + 0.03
        )
    )
    summary["acceptance"] = {
        "required_treatment_hits": required_hits,
        "treatment_two_of_three": two_of_three,
        "treatment_beats_control": beats_control,
        "passed": two_of_three and beats_control,
    }


def evaluate_skill(
    experiment: dict[str, Any], repetitions: int | None = None,
    agent: str | None = None, model: str | None = None,
    full_iterations: bool = False, resume: str | None = None,
) -> dict[str, Any]:
    requested_repetitions = repetitions
    repetitions = repetitions or int(experiment["repetitions"])
    use_full_iterations = bool(
        full_iterations or experiment.get("full_iterations", True)
    )
    if resume:
        resume_path = Path(resume)
        experiment_dir = (
            resume_path.resolve()
            if resume_path.is_absolute()
            else (REPO_ROOT / resume_path).resolve()
        )
        summary_path = experiment_dir / "summary.json"
        if not summary_path.is_file():
            raise ConfigError(f"experiment summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "completed":
            raise ConfigError("completed experiments cannot be resumed")
        if summary.get("config_fingerprint") != config_fingerprint(experiment):
            raise ConfigError("resume experiment configuration is incompatible")
        if (
            requested_repetitions is not None
            and requested_repetitions != int(summary["repetitions"])
        ):
            raise ConfigError("resume repetitions differ from the interrupted experiment")
        repetitions = int(summary["repetitions"])
        agent = agent or summary.get("agent")
        model = model or summary.get("model")
        use_full_iterations = summary.get("loop_mode") == "full_iterations"
        summary.update({"status": "running"})
    else:
        tag = dt.datetime.now().strftime("experiment-%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        experiment_dir = RESULTS_DIR / tag
        experiment_dir.mkdir(parents=True, exist_ok=False)
        try:
            display = str(experiment_dir.relative_to(REPO_ROOT))
        except ValueError:
            display = str(experiment_dir)
        base_agent = load_config(experiment["goal"])["agent"]
        effective_agent = agent or base_agent["backend"]
        effective_model = model or base_agent["model"]
        summary = {
            "schema_version": 1,
            "name": experiment["name"],
            "status": "running",
            "experiment_dir": display,
            "config_fingerprint": config_fingerprint(experiment),
            "target_score": float(experiment["target_score"]),
            "repetitions": repetitions,
            "loop_mode": "full_iterations" if use_full_iterations else "goal_driven",
            "agent": effective_agent,
            "model": effective_model,
            "runs": [],
            "pending": None,
            "statistics": {},
            "reproduce_command": (
                ".venv/bin/python jimu-dse/scripts/closed_loop.py evaluate-skill "
                f"--config {experiment['_config_path']} "
                f"--repetitions {repetitions} --agent {effective_agent} "
                f"--model {effective_model} --resume {display}"
            ),
        }

    tasks = [
        (replicate, arm)
        for replicate in range(1, repetitions + 1)
        for arm in ("control", "legacy", "treatment")
    ]
    completed_keys = {
        (int(item["replicate"]), item["arm"]) for item in summary["runs"]
    }
    pending = summary.get("pending")
    try:
        for replicate, arm in tasks:
            if (replicate, arm) in completed_keys:
                continue
            run_config = load_config(experiment["goal"])
            run_config["skills"] = copy.deepcopy(experiment["arms"][arm])
            run_config = resolved_config(
                run_config, agent=agent, model=model,
                full_iterations=use_full_iterations,
            )
            run_config["loop"]["target_score"] = float(experiment["target_score"])
            validate_config(run_config)
            child_resume = None
            if (
                pending
                and int(pending.get("replicate", -1)) == replicate
                and pending.get("arm") == arm
            ):
                child_resume = pending.get("run_dir")
            run_summary = execute_run(
                run_config,
                resume=child_resume,
                results_root=experiment_dir / "runs",
            )
            if run_summary["status"] == "interrupted":
                summary["status"] = "interrupted"
                summary["pending"] = {
                    "replicate": replicate,
                    "arm": arm,
                    "run_dir": run_summary["run_dir"],
                }
                _finalize_experiment_statistics(summary)
                return summary
            if run_summary["status"] != "completed":
                summary["status"] = "failed"
                summary["pending"] = {
                    "replicate": replicate,
                    "arm": arm,
                    "run_dir": run_summary["run_dir"],
                }
                _finalize_experiment_statistics(summary)
                return summary
            summary["runs"].append({
                "replicate": replicate,
                "arm": arm,
                "run_dir": run_summary["run_dir"],
                "skill_hashes": _skill_hashes(run_config["skills"]),
                "summary": run_summary,
            })
            summary["pending"] = None
            pending = None
            _finalize_experiment_statistics(summary)
            _write_experiment_checkpoint(experiment_dir, experiment, summary)
        summary["status"] = "completed"
        _finalize_experiment_statistics(summary)
        return summary
    finally:
        _write_experiment_checkpoint(experiment_dir, experiment, summary)


def _print_agent_start_error(summary: dict[str, Any]) -> None:
    interruption = (summary.get("interruptions") or [{}])[-1]
    print("OpenCode agent failed to start", file=sys.stderr)
    print(f"reason: {interruption.get('reason', 'unknown')}", file=sys.stderr)
    print(
        f"iteration: {interruption.get('iteration')}/"
        f"{summary.get('max_iterations', '?')}",
        file=sys.stderr,
    )
    print(
        f"backend/model: {interruption.get('backend')} / "
        f"{interruption.get('model')}",
        file=sys.stderr,
    )
    print(f"exit code: {interruption.get('exit_code')}", file=sys.stderr)
    print(f"message: {interruption.get('message', '')}", file=sys.stderr)
    print(f"progress saved: {summary.get('run_dir')}", file=sys.stderr)
    print(f"best score: {summary.get('best_score', 0):.6f}", file=sys.stderr)
    print(f"resume: {summary.get('reproduce_command')}", file=sys.stderr)


def list_goals() -> int:
    found = 0
    for path in sorted(GOALS_DIR.glob("*/goal.yaml")):
        try:
            config = load_config(config_path=str(path))
            print(f"{config['name']}\t{config['description']}")
            found += 1
        except ConfigError as exc:
            print(f"{path.parent.name}\tINVALID: {exc}")
    return 0 if found else 1


def inspect_run(path: str) -> int:
    summary_path = Path(path).resolve() / "run-summary.json"
    if not summary_path.is_file():
        raise ConfigError(f"run summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(_summary_report(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-goals")
    for name in ("validate-config", "render-prompt"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--goal")
        cmd.add_argument("--config")
    run = sub.add_parser("run")
    run.add_argument("--goal", default="dram-optimization")
    run.add_argument("--config")
    run.add_argument("--agent", choices=["pi", "opencode"])
    run.add_argument("--model")
    run.add_argument("--resume")
    run.add_argument(
        "--max-iterations", type=int,
        help="override loop.max_iterations (CLI overrides environment and YAML)",
    )
    run.add_argument(
        "--agent-timeout", type=int, metavar="SECONDS",
        help="override the per-iteration agent timeout; 0 disables it",
    )
    run.add_argument(
        "--full-iterations", action="store_true",
        help="ignore score/no-improvement early stops and run max_iterations",
    )
    run.add_argument(
        "--quiet", action="store_true",
        help="suppress progress messages; the final report is still printed",
    )
    evaluate = sub.add_parser("evaluate-skill")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--repetitions", type=int)
    evaluate.add_argument("--agent", choices=["pi", "opencode"])
    evaluate.add_argument("--model")
    evaluate.add_argument("--resume")
    evaluate.add_argument("--full-iterations", action="store_true")
    inspect = sub.add_parser("inspect-run")
    inspect.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-goals":
            return list_goals()
        if args.command == "inspect-run":
            return inspect_run(args.run_dir)
        if args.command == "evaluate-skill":
            experiment = load_experiment_config(args.config)
            summary = evaluate_skill(
                experiment,
                repetitions=args.repetitions,
                agent=args.agent,
                model=args.model,
                full_iterations=args.full_iterations,
                resume=args.resume,
            )
            print(_experiment_report(summary), end="")
            if summary["status"] == "interrupted":
                pending = summary.get("pending", {})
                run_summary_path = (
                    REPO_ROOT / pending.get("run_dir", "") / "run-summary.json"
                )
                if run_summary_path.is_file():
                    child = json.loads(run_summary_path.read_text(encoding="utf-8"))
                    _print_agent_start_error(child)
                print(
                    f"experiment resume: {summary.get('reproduce_command')}",
                    file=sys.stderr,
                )
                return 3
            return 0 if summary["status"] == "completed" else 1
        config = load_config(args.goal, args.config)
        if args.command == "validate-config":
            print(f"OK: {config['name']} ({config['_config_path']})")
            return 0
        config = resolved_config(
            config, getattr(args, "agent", None), getattr(args, "model", None),
            getattr(args, "full_iterations", False),
            getattr(args, "max_iterations", None),
            getattr(args, "agent_timeout", None),
        )
        if args.command == "render-prompt":
            print(render_prompt(config), end="")
            return 0
        summary = execute_run(
            config, args.resume,
            progress=None if getattr(args, "quiet", False) else _progress,
        )
        print(_summary_report(summary))
        if summary["status"] == "interrupted":
            _print_agent_start_error(summary)
            return 3
        return 0 if summary["status"] == "completed" else 1
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
