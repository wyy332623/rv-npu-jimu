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
import subprocess
import sys
import time
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by CLI environments
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
GOALS_DIR = REPO_ROOT / "jimu-dse" / "goals"
RESULTS_DIR = REPO_ROOT / "jimu-dse" / "results"
SUPPORTED_METRICS = {
    "total_bytes", "instr_count", "mv_mul_count", "mat_rd_ops", "test_pass",
    "memory_access_count", "memory_read_count", "memory_write_count",
    "register_access_count", "register_read_count", "register_write_count",
    "estimated_time",
    "scalesim_layer_count", "scalesim_compute_cycles",
    "scalesim_stall_cycles", "trace_memory_cycles", "auxiliary_cycles",
    "predicted_npu_cycles",
}
REGISTER_RESOURCES = {"VRF", "MRF", "SRF", "REG"}
TEMPLATE_FIELDS = {
    "goal_name", "goal_description", "iteration", "target_file",
    "hardware", "metrics", "clusters", "skills", "constraints",
    "self_verify", "gate_commands", "cost_model",
}
AGENT_RUNTIME_SAFETY = [
    "Never run `make clean` from the repository root. Use the configured "
    "firmware build or test commands instead.",
    "Never delete, move, rename, or modify `jimu-dse/results` or any "
    "`run-*` directory; those paths contain the active run state.",
]
TOP_LEVEL_KEYS = {
    "schema_version", "name", "description", "target", "agent", "prompt",
    "skills", "probe", "acceptance", "loop", "artifacts",
}
SECTION_KEYS = {
    "target": {
        "firmware", "baseline", "allowed_files", "hardware", "sequence_lengths",
    },
    "agent": {"backend", "model", "timeout_seconds", "context_files"},
    "prompt": {
        "template", "goal", "constraints", "self_verify",
    },
    "probe": {
        "metrics", "dag", "cycle_limit", "scoring_sequence_length",
        "cost_model", "cycle_model",
    },
    "acceptance": {"gates", "score"},
    "loop": {
        "max_iterations", "min_score_delta", "max_no_improvement",
        "target_score",
    },
    "artifacts": {
        "save_candidates", "save_diffs", "save_prompts", "save_probes",
        "save_graphs",
    },
}


class ConfigError(ValueError):
    pass


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
    required = {"name", "version", "category", "description"}
    missing = required - set(metadata or {})
    if not isinstance(metadata, dict) or missing:
        raise ConfigError(
            f"skill {path.relative_to(REPO_ROOT)} missing metadata: "
            f"{', '.join(sorted(missing))}"
        )
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

    agent = config["agent"]
    if agent.get("backend") not in {"pi", "opencode"}:
        raise ConfigError("agent.backend must be pi or opencode")
    for context in agent.get("context_files", []):
        _repo_path(context, "agent.context_files")

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
        "predicted_npu_cycles",
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
                "instruction_latencies",
            },
            "cycle model profile",
        )
        if profile.get("schema_version") != 1:
            raise ConfigError("cycle model schema_version must be 1")
        if profile.get("backend") != "scalesim":
            raise ConfigError("cycle model backend must be scalesim")
        _repo_path(profile.get("scalesim_config", ""), "cycle model scalesim_config")
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
            if not isinstance(name, str) or not isinstance(value, int) or value < 0:
                raise ConfigError(
                    "cycle model instruction latencies must be non-negative integers"
                )
    if set(metrics) & cycle_metrics and cycle_model is None:
        raise ConfigError(
            "probe.cycle_model is required when SCALE-Sim cycle metrics are requested"
        )

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
    for key in ("max_iterations", "max_no_improvement"):
        if not isinstance(loop.get(key), int) or loop[key] < 1:
            raise ConfigError(f"loop.{key} must be a positive integer")
    if not isinstance(loop.get("min_score_delta"), (int, float)):
        raise ConfigError("loop.min_score_delta must be numeric")


def resolved_config(
    config: dict[str, Any], agent: str | None = None, model: str | None = None
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
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
    validate_config(result)
    return result


def config_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def render_prompt(
    config: dict[str, Any], iteration: int = 1,
    metrics: dict[str, Any] | None = None, clusters: list[str] | None = None,
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
    values = {
        "goal_name": config["name"],
        "goal_description": config["prompt"]["goal"],
        "iteration": iteration,
        "target_file": config["target"]["firmware"],
        "hardware": json.dumps(config["target"]["hardware"], sort_keys=True),
        "cost_model": json.dumps(
            config["probe"].get("cost_model", {}), sort_keys=True, indent=2
        ),
        "metrics": json.dumps(metrics or {}, sort_keys=True, indent=2),
        "clusters": "\n".join(clusters or ["(not available during preview)"]),
        "skills": "\n\n".join(skill_parts),
        "constraints": "\n".join(f"- {item}" for item in constraints),
        "self_verify": config["prompt"].get("self_verify", ""),
        "gate_commands": gate_commands,
    }
    return config["prompt"]["template"].format(**values).strip() + "\n"


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


def _run(command: list[str] | str, timeout: int, shell: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True,
            timeout=timeout, shell=shell,
        )
        return {
            "exit_code": proc.returncode, "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-20000:], "timed_out": False,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": None, "stdout": (exc.stdout or "")[-20000:],
            "stderr": (exc.stderr or "")[-20000:], "timed_out": True,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except OSError as exc:
        return {
            "exit_code": None, "stdout": "", "stderr": str(exc),
            "timed_out": False,
            "duration_seconds": round(time.monotonic() - started, 3),
        }


def build_firmware(config: dict[str, Any], seq_len: int) -> dict[str, Any]:
    hw = config["target"]["hardware"]
    dim, hidden, heads = hw["dim"], hw["hidden"], hw["num_head"]
    proj_base = hidden * seq_len + 4
    mat_size = hidden * hidden
    stride = mat_size + hidden
    num_tiles = hidden // dim
    ln_base = proj_base + 6 * stride
    ln_size = num_tiles * 8
    env = os.environ.copy()
    env.update({
        "CC": os.getenv("CC", "riscv64-unknown-elf-gcc"),
        "NATIVE_DIM": str(dim), "SEQ_LEN": str(seq_len),
        "_HIDDEN_SIZE": str(hidden), "_PROJ_BASE": str(proj_base),
        "_MAT_SIZE": str(mat_size), "_STRIDE": str(stride),
        "_NUM_TILES": str(num_tiles), "_LN1_GAMMA": str(ln_base),
        "_LN1_BETA": str(ln_base + ln_size),
        "_LN2_GAMMA": str(ln_base + 2 * ln_size),
        "_LN2_BETA": str(ln_base + 3 * ln_size),
        "_SCRATCH": "1280", "NUM_HEAD": str(heads),
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


def probe_firmware(
    config: dict[str, Any], seq_len: int, output_dir: Path | None = None,
) -> dict[str, Any]:
    build = build_firmware(config, seq_len)
    if not build["passed"]:
        return {"passed": False, "build": build, "metrics": {}, "clusters": []}
    hw = config["target"]["hardware"]
    script = f"""
import json, sys
sys.path.insert(0, '.')
sys.path.insert(0, 'jimu-dse/scripts')
import numpy as np
from emulator.npu_device_mini import NpuDeviceMini, MEM_DRAM
from emulator.npu_event_trace import EventTracer
from emulator.trace_recorder import TraceRecorder
from iss.mini_rv64 import MiniRV64
from closed_loop import calculate_cost_metrics
dim={hw['dim']}; h={hw['hidden']}; sl={seq_len}
npu=NpuDeviceMini(native_dim=dim); npu.set_hidden_size(h); npu.set_seq_len(sl)
npu._vrf[MEM_DRAM][0:h]=np.zeros(h,dtype=np.float32)
tracer=EventTracer(npu); rec=TraceRecorder(npu); cpu=MiniRV64()
cpu.set_mmio_device(rec); cpu.load_elf('{build['elf']}')
cpu.run(cycles={config['probe'].get('cycle_limit', 300000)})
ds=npu.get_dram_stats()
total=(ds.get('vec_rd_elements',0)+ds.get('vec_wr_elements',0)+ds.get('mat_rd_elements',0)+ds.get('mat_wr_elements',0))*4
mv=sum(1 for e in tracer.events if ((e['raw'] if isinstance(e,dict) else e.inst)>>24)&0xFF in (7,27))
metrics={{'total_bytes':total,'instr_count':len(rec.inst_trace),'mv_mul_count':mv,'mat_rd_ops':ds.get('mat_rd_ops',0),'test_pass':0}}
metrics.update(calculate_cost_metrics(tracer.events, ds, {config["probe"].get("cost_model")!r}))
cycle_model={config["probe"].get("cycle_model")!r}
if cycle_model:
    import yaml
    sys.path.insert(0, 'jimu-dse/timing')
    from scalesim_adapter import simulate_trace
    with open(cycle_model['profile'], encoding='utf-8') as handle:
        timing_profile=yaml.safe_load(handle)
    metrics.update(simulate_trace(tracer.events, {{'dim':dim, 'hidden':h}}, timing_profile))
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
    probe = {
        "passed": result["exit_code"] == 0 and all(x in metrics for x in wanted if x != "test_pass"),
        "build": build, "process": result, "metrics": metrics, "clusters": [],
        "sequence_length": seq_len,
    }
    if output_dir and config["probe"].get("dag", {}).get("enabled"):
        dag_dir = output_dir / f"dag-seq{seq_len}"
        dag = _run([
            sys.executable, "jimu-dse/scripts/visualize_graph.py", "--phase", "all",
            "--dim", str(hw["dim"]), "--hidden", str(hw["hidden"]),
            "--seq-len", "1", "--num-head", str(hw["num_head"]),
            "-o", str(dag_dir),
        ], timeout=300)
        probe["dag"] = dag
        cluster_file = dag_dir / "dram_clusters.txt"
        if cluster_file.is_file():
            probe["clusters"] = cluster_file.read_text(encoding="utf-8").splitlines()
    return probe


def invoke_agent(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    backend = config["agent"]["backend"]
    executable = shutil.which(backend)
    if not executable:
        return {"status": "agent_unavailable", "exit_code": None, "stderr": f"{backend} not found"}
    timeout = int(config["agent"]["timeout_seconds"])
    if backend == "opencode":
        command = [executable, "run", "--model", config["agent"]["model"]]
        for item in config["agent"].get("context_files", []):
            command += ["-f", item]
        command += ["--dangerously-skip-permissions", prompt]
    else:
        command = [executable]
        for item in config["skills"]:
            command += ["--skill", str(_repo_path(item["path"], "skill.path"))]
        command += ["-p", prompt]
    result = _run(command, timeout=timeout)
    result["status"] = (
        "agent_timeout" if result["timed_out"]
        else "agent_failed" if result["exit_code"] != 0
        else "completed"
    )
    return result


def run_gates(
    config: dict[str, Any], changed_files: list[str], probe: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    allowed = set(config["target"]["allowed_files"])
    for gate in config["acceptance"]["gates"]:
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
            result = _run(command, int(gate.get("timeout_seconds", 300)), shell=True)
            success_codes = gate.get("success_codes", [0])
            item.update({"passed": result["exit_code"] in success_codes, "result": result})
            if "test_pass" in config["probe"]["metrics"]:
                probe["metrics"]["test_pass"] = 1 if item["passed"] else 0
        results.append(item)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


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
        f"- Best iteration: `{summary.get('best_iteration')}`",
        f"- Best score: `{summary.get('best_score', 0):.6f}`",
        f"- Results: `{summary['run_dir']}`", "",
        "## Iterations", "",
        "| Iteration | Status | Gates | Score | Promoted |",
        "|---:|---|---|---:|---|",
    ]
    for item in summary["iterations"]:
        gates = "PASS" if item.get("gates_passed") else "FAIL"
        lines.append(
            f"| {item['iteration']} | {item['status']} | {gates} | "
            f"{item.get('score', 0):.6f} | {item.get('promoted', False)} |"
        )
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
    lines += ["", "## Reproduce", "", f"`{summary['reproduce_command']}`", ""]
    return "\n".join(lines)


def execute_run(
    config: dict[str, Any], resume: str | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    target = _repo_path(config["target"]["firmware"], "target.firmware")
    baseline_path = _repo_path(config["target"]["baseline"], "target.baseline")
    run_tag = dt.datetime.now().strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    run_dir = (results_root or RESULTS_DIR) / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        run_dir_display = str(run_dir.relative_to(REPO_ROOT))
    except ValueError:
        run_dir_display = str(run_dir)

    if resume:
        prior = Path(resume).resolve()
        prior_config = _read_yaml(prior / "resolved-config.yaml")
        if config_fingerprint(prior_config) != config_fingerprint(config):
            raise ConfigError("resume configuration is incompatible with the selected goal")
        start_file = prior / "candidate_best.c"
        if not start_file.is_file():
            raise ConfigError("resume directory has no candidate_best.c")
    else:
        start_file = baseline_path

    original = target.read_bytes()
    target.write_bytes(start_file.read_bytes())
    best_bytes = target.read_bytes()
    baseline_probe: dict[str, Any] | None = None
    _write_yaml(run_dir / "resolved-config.yaml", config)
    summary: dict[str, Any] = {
        "schema_version": 1, "goal": config["name"], "config_fingerprint": config_fingerprint(config),
        "run_dir": run_dir_display, "status": "running",
        "stop_reason": "", "best_iteration": None, "best_score": 0.0, "iterations": [],
        "scoring_sequence_length": scoring_sequence_length(config),
        "cost_model": config["probe"].get("cost_model"),
        "reproduce_command": (
            f"python3 jimu-dse/scripts/closed_loop.py run --goal {config['name']} "
            f"--resume {run_dir_display}"
        ),
    }
    try:
        baseline_probe = probe_firmware(
            config, scoring_sequence_length(config),
            run_dir / "baseline" if config["artifacts"].get("save_graphs") else None,
        )
        if not baseline_probe["passed"]:
            summary.update({"status": "failed", "stop_reason": "baseline_probe_failed"})
            return summary
        baseline_metrics = baseline_probe["metrics"]
        summary["baseline_metrics"] = baseline_metrics
        summary["best_metrics"] = copy.deepcopy(baseline_metrics)
        _write_json(run_dir / "baseline-probe.json", baseline_probe)
        best_bytes = target.read_bytes()
        best_score = 0.0
        no_improvement = 0

        for iteration in range(1, config["loop"]["max_iterations"] + 1):
            target.write_bytes(best_bytes)
            before = target.read_bytes()
            source_before = _source_snapshot()
            current_probe = probe_firmware(config, scoring_sequence_length(config))
            prompt = render_prompt(
                config, iteration, current_probe["metrics"], current_probe["clusters"]
            )
            if config["artifacts"].get("save_prompts"):
                (run_dir / f"prompt-{iteration}.txt").write_text(prompt, encoding="utf-8")
            agent_result = invoke_agent(config, prompt)
            changed, source_after = _snapshot_changes(source_before)
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
            record: dict[str, Any] = {
                "iteration": iteration, "status": agent_result["status"],
                "agent": agent_result, "changed_files": changed, "promoted": False,
                "gates_passed": False, "score": 0.0,
            }
            if agent_result["status"] != "completed" or not changed:
                record["status"] = agent_result["status"] if changed else (
                    "no_change" if agent_result["status"] == "completed" else agent_result["status"]
                )
                target.write_bytes(best_bytes)
                no_improvement += 1
            else:
                candidate = target.read_bytes()
                if config["artifacts"].get("save_candidates"):
                    (run_dir / f"candidate-{iteration}.c").write_bytes(candidate)
                candidate_probe = probe_firmware(
                    config, scoring_sequence_length(config),
                    run_dir / f"iteration-{iteration}" if config["artifacts"].get("save_graphs") else None,
                )
                gates = run_gates(config, changed, candidate_probe)
                if config["artifacts"].get("save_probes"):
                    _write_json(run_dir / f"probe-{iteration}.json", candidate_probe)
                record["probe"] = candidate_probe
                record["gates"] = gates
                record["gates_passed"] = all(x["passed"] for x in gates)
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
                    record.update({"score": score, "score_details": details, "status": "accepted"})
                    if score >= best_score + config["loop"]["min_score_delta"]:
                        best_score, best_bytes = score, candidate
                        record["promoted"] = True
                        summary["best_iteration"] = iteration
                        summary["best_score"] = score
                        summary["best_metrics"] = copy.deepcopy(candidate_probe["metrics"])
                        no_improvement = 0
                    else:
                        record["status"] = "not_improved"
                        target.write_bytes(best_bytes)
                        no_improvement += 1
                else:
                    record["status"] = "gate_failed"
                    target.write_bytes(best_bytes)
                    no_improvement += 1
            summary["iterations"].append(record)
            _write_json(run_dir / f"iteration-{iteration}.json", record)
            target_score = config["loop"].get("target_score")
            if target_score is not None and best_score >= target_score:
                summary["stop_reason"] = "target_score_reached"
                break
            if no_improvement >= config["loop"]["max_no_improvement"]:
                summary["stop_reason"] = "no_improvement_limit"
                break
        else:
            summary["stop_reason"] = "max_iterations"

        (run_dir / "candidate_best.c").write_bytes(best_bytes)
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
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "run-summary.json", summary)
        (run_dir / "report.md").write_text(_summary_report(summary), encoding="utf-8")


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
        config = load_config(args.goal, args.config)
        if args.command == "validate-config":
            print(f"OK: {config['name']} ({config['_config_path']})")
            return 0
        config = resolved_config(
            config, getattr(args, "agent", None), getattr(args, "model", None)
        )
        if args.command == "render-prompt":
            print(render_prompt(config), end="")
            return 0
        summary = execute_run(config, args.resume)
        print(_summary_report(summary))
        return 0 if summary["status"] == "completed" else 1
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
