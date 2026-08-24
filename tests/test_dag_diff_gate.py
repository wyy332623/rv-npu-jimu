import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "jimu-dse" / "scripts" / "dag_diff_gate.py"
SPEC = importlib.util.spec_from_file_location("jimu_dag_diff_gate", GATE_PATH)
dag_diff_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dag_diff_gate
SPEC.loader.exec_module(dag_diff_gate)


def dram_op(kind, tensor_slice, address=0x300):
    resource = {
        "space": "DRAM",
        "address": address,
        "address_hex": f"0x{address:x}",
        "slice": tensor_slice,
    }
    return {
        "kind": kind,
        "defs": [resource] if kind == "DRAM_STORE" else [],
        "uses": [resource] if kind == "DRAM_LOAD" else [],
    }


def fused_dram_read(kind, tensor_slice, address):
    resource = {
        "space": "DRAM",
        "address": address,
        "address_hex": f"0x{address:x}",
        "slice": tensor_slice,
    }
    return {"kind": kind, "defs": [], "uses": [resource]}


def candidate_data():
    return {
        "candidates": [
            {
                "id": "candidate-dram-0007",
                "eligible": True,
                "tensor_slice": "K[pos=0,tile=0]",
                "estimated_saving": {
                    "projected_seq2_bytes": 48,
                    "projected_seq6_bytes": 336,
                    "projection_assumption": "test",
                },
            }
        ]
    }


def macro_data():
    return {
        "macros": [
            {
                "id": "macro-dram-l1-attention",
                "level": "L1",
                "eligible": True,
                "allocation": {
                    "allocation_proven": True,
                    "cross_config_proven": True,
                    "validation_matrix_complete": True,
                },
                "expected_dram_resources": [
                    {
                        "tensor_slice": "K[pos=0,tile=0]",
                        "dram_address": 0x300,
                    },
                    {
                        "tensor_slice": "V[pos=0,tile=0]",
                        "dram_address": 0x400,
                    },
                ],
                "estimated_saving": {
                    "projected_seq2_bytes": 96,
                    "projected_seq6_bytes": 672,
                    "projection_assumption": "test macro",
                },
            }
        ]
    }


def metric(total_bytes):
    return {"total_bytes": total_bytes}


def evaluate(source, after_ops, **kwargs):
    before_ops = [
        dram_op("DRAM_STORE", "K[pos=0,tile=0]"),
        dram_op("DRAM_LOAD", "K[pos=0,tile=0]"),
    ]
    return dag_diff_gate.evaluate_dag_evidence(
        candidate_data(),
        before_ops,
        after_ops,
        source,
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(608),
        after_seq6=metric(700),
        **kwargs,
    )


def test_accepts_unique_candidate_with_structural_and_measured_improvement():
    result = evaluate(
        "// JIMU_DAG_CANDIDATE: candidate-dram-0007\n",
        [dram_op("DRAM_STORE", "K[pos=0,tile=0]")],
        before_edges=[{"type": "RAW"}, {"type": "RAW"}],
        after_edges=[{"type": "RAW"}],
        before_lifetimes=[
            {
                "resource": {"bank": 1, "row": 0},
                "interval": {"start_index": 0, "end_index": 3},
            },
            {
                "resource": {"bank": 2, "row": 0},
                "interval": {"start_index": 1, "end_index": 2},
            },
        ],
        after_lifetimes=[
            {
                "resource": {"bank": 1, "row": 0},
                "interval": {"start_index": 0, "end_index": 3},
            }
        ],
        before_metadata={"firmware_config": {"dim": 2}},
        after_metadata={"firmware_config": {"dim": 2}},
        before_micro_ops_sha256="before-sha",
        after_micro_ops_sha256="after-sha",
    )

    assert result["evidence_pass"] is True
    assert result["declaration"]["selected_candidate_id"] == (
        "candidate-dram-0007"
    )
    assert result["selected_tensor_delta"]["removed_load_ops"] == 1
    assert result["tensor_delta"][0]["removed_load_ops"] == 1
    assert result["metric_evidence"]["seq6"]["saved_bytes"] == 36
    assert result["dag_delta"]["edge_type_delta"]["RAW"]["removed"] == 1
    assert (
        result["dag_delta"]["vrf_pressure"]["before"]["peak_live_ranges"] == 2
    )
    assert (
        result["dag_delta"]["vrf_pressure"]["after"]["peak_live_ranges"] == 1
    )
    assert result["artifact_identity"]["micro_ops_identical"] is False
    assert result["artifact_identity"]["before"]["micro_ops_sha256"] == (
        "before-sha"
    )


def test_rejects_missing_or_duplicate_candidate_marker():
    missing = evaluate("", [])
    duplicate = evaluate(
        "\n".join(
            [
                "// JIMU_DAG_CANDIDATE: candidate-dram-0007",
                "// JIMU_DAG_CANDIDATE: candidate-dram-0007",
            ]
        ),
        [],
    )

    assert missing["evidence_pass"] is False
    assert "missing JIMU_DAG_CANDIDATE declaration" in (
        missing["failure_reasons"]
    )
    assert duplicate["evidence_pass"] is False
    assert any(
        "exactly one" in reason for reason in duplicate["failure_reasons"]
    )


def test_rejects_unknown_candidate_or_unrelated_dag_change():
    unknown = evaluate(
        "// JIMU_DAG_CANDIDATE: candidate-dram-9999\n",
        [],
    )
    unrelated = evaluate(
        "// JIMU_DAG_CANDIDATE: candidate-dram-0007\n",
        [
            dram_op("DRAM_STORE", "K[pos=0,tile=0]"),
            dram_op("DRAM_LOAD", "K[pos=0,tile=0]"),
        ],
    )

    assert unknown["evidence_pass"] is False
    assert any("not eligible" in reason for reason in unknown["failure_reasons"])
    assert unrelated["evidence_pass"] is False
    assert any(
        "no matching Tensor" in reason
        for reason in unrelated["failure_reasons"]
    )
    assert any(
        "baseline evidence may have been overwritten" in reason
        for reason in unrelated["failure_reasons"]
    )


def test_reports_observed_tensor_when_declared_candidate_does_not_match():
    before_ops = [
        dram_op("DRAM_STORE", "K[pos=0,tile=0]"),
        dram_op("DRAM_LOAD", "K[pos=0,tile=0]"),
        dram_op("DRAM_STORE", "SCRATCH[tile=1]"),
        dram_op("DRAM_LOAD", "SCRATCH[tile=1]"),
    ]
    after_ops = before_ops[:2]

    result = dag_diff_gate.evaluate_dag_evidence(
        candidate_data(),
        before_ops,
        after_ops,
        "// JIMU_DAG_CANDIDATE: candidate-dram-0007\n",
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(608),
        after_seq6=metric(700),
    )

    assert result["evidence_pass"] is False
    assert result["tensor_delta"][0]["tensor_slice"] == "SCRATCH[tile=1]"
    assert any(
        "observed reductions belong to: SCRATCH[tile=1]" in reason
        for reason in result["failure_reasons"]
    )


def test_rejects_dag_comparison_with_different_firmware_config():
    result = evaluate(
        "// JIMU_DAG_CANDIDATE: candidate-dram-0007\n",
        [dram_op("DRAM_STORE", "K[pos=0,tile=0]")],
        before_metadata={"firmware_config": {"dim": 2}},
        after_metadata={"firmware_config": {"dim": 4}},
    )

    assert result["evidence_pass"] is False
    assert "before/after DAG firmware_config differs" in (
        result["failure_reasons"]
    )


def test_rejects_metric_direction_mismatch():
    result = dag_diff_gate.evaluate_dag_evidence(
        candidate_data(),
        [
            dram_op("DRAM_STORE", "K[pos=0,tile=0]"),
            dram_op("DRAM_LOAD", "K[pos=0,tile=0]"),
        ],
        [dram_op("DRAM_STORE", "K[pos=0,tile=0]")],
        "// JIMU_DAG_CANDIDATE: candidate-dram-0007\n",
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(612),
        after_seq6=metric(736),
    )

    assert result["evidence_pass"] is False
    assert "seq2 measured DRAM traffic regressed" in result["failure_reasons"]
    assert (
        "seq6 measured DRAM traffic did not improve"
        in result["failure_reasons"]
    )


def test_disabled_gate_records_failure_without_rejecting():
    result = evaluate("", [], gate_enabled=False)

    assert result["consistency_pass"] is False
    assert result["evidence_pass"] is True


def evaluate_macro(after_ops):
    before_ops = [
        dram_op("DRAM_STORE", "K[pos=0,tile=0]", 0x300),
        dram_op("DRAM_LOAD", "K[pos=0,tile=0]", 0x300),
        dram_op("DRAM_STORE", "V[pos=0,tile=0]", 0x400),
        dram_op("DRAM_LOAD", "V[pos=0,tile=0]", 0x400),
    ]
    return dag_diff_gate.evaluate_dag_evidence(
        candidate_data(),
        before_ops,
        after_ops,
        "// JIMU_DAG_MACRO: macro-dram-l1-attention\n",
        before_macros=macro_data(),
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(560),
        after_seq6=metric(600),
    )


def test_accepts_complete_macro_with_exact_address_reductions():
    result = evaluate_macro([])

    assert result["evidence_pass"] is True
    assert result["declaration"]["selected_kind"] == "macro"
    assert result["declaration"]["selected_macro_id"] == (
        "macro-dram-l1-attention"
    )
    assert len(result["selected_resource_delta"]) == 2
    assert all(
        resource["structural_reduction"]
        for resource in result["selected_resource_delta"]
    )


def test_rejects_partial_macro_implementation():
    result = evaluate_macro(
        [
            dram_op("DRAM_STORE", "V[pos=0,tile=0]", 0x400),
            dram_op("DRAM_LOAD", "V[pos=0,tile=0]", 0x400),
        ]
    )

    assert result["evidence_pass"] is False
    assert any(
        "macro member has no exact-address" in reason
        for reason in result["failure_reasons"]
    )


def test_rejects_macro_scope_creep_and_mixed_markers():
    before_ops = [
        dram_op("DRAM_STORE", "K[pos=0,tile=0]", 0x300),
        dram_op("DRAM_LOAD", "K[pos=0,tile=0]", 0x300),
        dram_op("DRAM_STORE", "V[pos=0,tile=0]", 0x400),
        dram_op("DRAM_LOAD", "V[pos=0,tile=0]", 0x400),
        dram_op("DRAM_STORE", "SCRATCH[tile=0]", 0x700),
    ]
    result = dag_diff_gate.evaluate_dag_evidence(
        candidate_data(),
        before_ops,
        [],
        "// JIMU_DAG_MACRO: macro-dram-l1-attention\n",
        before_macros=macro_data(),
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(550),
        after_seq6=metric(590),
    )
    mixed = dag_diff_gate.evaluate_dag_evidence(
        candidate_data(),
        before_ops[:4],
        [],
        "// JIMU_DAG_MACRO: macro-dram-l1-attention\n"
        "// JIMU_DAG_CANDIDATE: candidate-dram-0007\n",
        before_macros=macro_data(),
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(550),
        after_seq6=metric(590),
    )

    assert result["evidence_pass"] is False
    assert any("outside its strict scope" in reason for reason in result["failure_reasons"])
    assert mixed["evidence_pass"] is False
    assert any("exactly one DAG declaration" in reason for reason in mixed["failure_reasons"])


def test_pr6_rejects_observed_only_allocation_and_out_of_order_l2():
    observed_only = macro_data()
    observed_only["macros"][0]["allocation"]["cross_config_proven"] = False
    no_cross_proof = dag_diff_gate.evaluate_dag_evidence(
        candidate_data(),
        [
            dram_op("DRAM_STORE", "K[pos=0,tile=0]", 0x300),
            dram_op("DRAM_LOAD", "K[pos=0,tile=0]", 0x300),
            dram_op("DRAM_STORE", "V[pos=0,tile=0]", 0x400),
            dram_op("DRAM_LOAD", "V[pos=0,tile=0]", 0x400),
        ],
        [],
        "// JIMU_DAG_MACRO: macro-dram-l1-attention\n",
        before_macros=observed_only,
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(560),
        after_seq6=metric(600),
    )
    assert no_cross_proof["evidence_pass"] is False
    assert any(
        "no complete cross-config VRF allocation" in reason
        for reason in no_cross_proof["failure_reasons"]
    )

    staged = macro_data()
    staged["macros"].append(
        {
            "id": "macro-dram-l2-loop-invariants",
            "level": "L2",
            "eligible": True,
            "allocation": {
                "allocation_proven": True,
                "cross_config_proven": True,
                "validation_matrix_complete": True,
            },
            "expected_dram_resources": [
                {"tensor_slice": "B_Q[tile=0]", "dram_address": 0x500}
            ],
            "estimated_saving": {
                "projected_seq2_bytes": 8,
                "projected_seq6_bytes": 40,
                "projection_assumption": "test",
            },
        }
    )
    out_of_order = dag_diff_gate.evaluate_dag_evidence(
        candidate_data(),
        [dram_op("DRAM_LOAD", "B_Q[tile=0]", 0x500)],
        [],
        "// JIMU_DAG_MACRO: macro-dram-l2-loop-invariants\n",
        before_macros=staged,
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(600),
        after_seq6=metric(696),
    )
    assert out_of_order["evidence_pass"] is False
    assert any(
        "before lower level is complete" in reason
        for reason in out_of_order["failure_reasons"]
    )


def test_pr6_accepts_l2_reduction_folded_into_vv_binop():
    l2_macros = {
        "macros": [
            {
                "id": "macro-dram-l2-loop-invariants",
                "level": "L2",
                "eligible": True,
                "allocation": {
                    "allocation_proven": True,
                    "cross_config_proven": True,
                    "validation_matrix_complete": True,
                },
                "expected_dram_resources": [
                    {"tensor_slice": "B_Q[tile=0]", "dram_address": 0x500}
                ],
                "estimated_saving": {
                    "projected_seq2_bytes": 8,
                    "projected_seq6_bytes": 40,
                    "projection_assumption": "test",
                },
            }
        ]
    }
    result = dag_diff_gate.evaluate_dag_evidence(
        {"candidates": []},
        [fused_dram_read("VV_BINOP", "B_Q[tile=0]", 0x500)],
        [],
        "// JIMU_DAG_MACRO: macro-dram-l2-loop-invariants\n",
        before_macros=l2_macros,
        before_seq2=metric(608),
        before_seq6=metric(736),
        after_seq2=metric(600),
        after_seq6=metric(696),
    )

    assert result["evidence_pass"] is True
    assert result["selected_resource_delta"][0]["removed_load_ops"] == 1
