import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "jimu-dse" / "scripts" / "validation_gate.py"
SPEC = importlib.util.spec_from_file_location("jimu_validation_gate", GATE_PATH)
assert SPEC and SPEC.loader
validation_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_gate
SPEC.loader.exec_module(validation_gate)


def test_accepts_exact_pass_count_with_no_skips():
    result = validation_gate.analyze_pytest_output(
        "...... [100%]\n6 passed in 1.23s\n",
        pytest_returncode=0,
        expected_passed=6,
    )

    assert result["correctness_pass"] is True
    assert result["returncode"] == 0
    assert result["passed"] == 6
    assert result["skipped"] == 0


def test_rejects_too_few_passed_tests():
    result = validation_gate.analyze_pytest_output(
        "2 passed, 4 deselected in 0.50s\n",
        pytest_returncode=0,
        expected_passed=6,
    )

    assert result["correctness_pass"] is False
    assert result["returncode"] == 126
    assert "expected 6" in result["failure_reason"]


def test_rejects_pytest_skips():
    result = validation_gate.analyze_pytest_output(
        "6 passed, 1 skipped in 1.23s\n",
        pytest_returncode=0,
        expected_passed=6,
    )

    assert result["returncode"] == 125
    assert result["skipped"] == 1


def test_diagnostic_round_text_is_not_a_pytest_skip():
    result = validation_gate.analyze_pytest_output(
        "Round 2: skipped (install amaranth)\n6 passed in 1.23s\n",
        pytest_returncode=0,
        expected_passed=6,
    )

    assert result["correctness_pass"] is True
    assert result["skipped"] == 0


def test_preserves_pytest_failure_status():
    result = validation_gate.analyze_pytest_output(
        "1 failed, 5 passed in 1.23s\n",
        pytest_returncode=1,
        expected_passed=6,
    )

    assert result["returncode"] == 1
    assert result["pytest_returncode"] == 1
