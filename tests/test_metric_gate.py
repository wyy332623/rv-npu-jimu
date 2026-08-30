import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_GATE_PATH = REPO_ROOT / "jimu-dse" / "scripts" / "metric_gate.py"
SPEC = importlib.util.spec_from_file_location("jimu_metric_gate", METRIC_GATE_PATH)
assert SPEC and SPEC.loader
metric_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metric_gate
SPEC.loader.exec_module(metric_gate)


def metrics(total_bytes, instr_count):
    return {
        "total_bytes": total_bytes,
        "instr_count": instr_count,
        "dram_stats": {"vec_rd_ops": 1},
    }


def test_accepts_seq6_improvement_without_seq2_or_instruction_regression():
    result = metric_gate.evaluate_g1_metrics(
        metrics(608, 1200),
        metrics(736, 3600),
        metrics(608, 1210),
        metrics(700, 3900),
        instruction_regression_limit=0.10,
    )

    assert result["metric_pass"] is True
    assert result["seq6"]["delta_bytes"] == -36


def test_rejects_seq2_regression_even_when_seq6_improves():
    result = metric_gate.evaluate_g1_metrics(
        metrics(608, 1200),
        metrics(736, 3600),
        metrics(612, 1200),
        metrics(700, 3600),
        instruction_regression_limit=0.10,
    )

    assert result["metric_pass"] is False
    assert "seq2 total_bytes regressed" in result["failure_reasons"][0]


def test_requires_strict_seq6_improvement():
    result = metric_gate.evaluate_g1_metrics(
        metrics(608, 1200),
        metrics(736, 3600),
        metrics(600, 1200),
        metrics(736, 3500),
        instruction_regression_limit=0.10,
    )

    assert result["metric_pass"] is False
    assert any("did not strictly improve" in reason for reason in result["failure_reasons"])


def test_rejects_excessive_instruction_regression():
    result = metric_gate.evaluate_g1_metrics(
        metrics(608, 1200),
        metrics(736, 3600),
        metrics(600, 1200),
        metrics(700, 4000),
        instruction_regression_limit=0.10,
    )

    assert result["metric_pass"] is False
    assert any("instruction count" in reason for reason in result["failure_reasons"])


def test_disabled_instruction_gate_allows_instruction_growth():
    result = metric_gate.evaluate_g1_metrics(
        metrics(608, 1200),
        metrics(736, 3600),
        metrics(600, 1200),
        metrics(700, 100000),
        instruction_regression_limit=0.10,
        instruction_gate_enabled=False,
    )

    assert result["metric_pass"] is True
    assert result["instruction_gate_enabled"] is False
    assert result["instruction_regression_limit"] is None
    assert result["seq6"]["max_instr_count"] is None


def test_rejects_metric_comparison_with_different_firmware_config():
    before2 = metrics(608, 1200)
    before6 = metrics(736, 3600)
    after2 = metrics(600, 1200)
    after6 = metrics(700, 3600)
    before2["firmware_config"] = {"dim": 2, "seq_len": 2}
    after2["firmware_config"] = {"dim": 2, "seq_len": 2}
    before6["firmware_config"] = {"dim": 2, "seq_len": 6}
    after6["firmware_config"] = {"dim": 2, "seq_len": 1}

    result = metric_gate.evaluate_g1_metrics(
        before2,
        before6,
        after2,
        after6,
        instruction_regression_limit=0.10,
    )

    assert result["metric_pass"] is False
    assert any("seq6 firmware_config changed" in reason for reason in result["failure_reasons"])
