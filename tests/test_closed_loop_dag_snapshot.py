from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT / "jimu-dse" / "scripts" / "npu_closed_loop.sh"
).read_text(encoding="utf-8")


def test_candidate_metric_probe_cannot_refresh_agent_dag():
    assert 'local dag_mode="${2:-refresh}"' in SCRIPT
    assert 'B6=$(probe "${SL6}" refresh)' in SCRIPT
    assert 'B6_NEW=$(probe "${SL6}" metric-only)' in SCRIPT


def test_pr4_uses_immutable_iteration_input_snapshot():
    assert 'DAG_BEFORE_DIR="${RESULTS}/dag_before_iter${iter}"' in SCRIPT
    assert 'cp -a "${RESULTS}/dag_agent/." "${DAG_BEFORE_DIR}/"' in SCRIPT
    assert '--before-dag "${DAG_BEFORE_DIR}"' in SCRIPT
    assert '--before-dag "${RESULTS}/dag_agent"' not in SCRIPT
    assert '-f "${DAG_BEFORE_DIR}/${dag_file}"' in SCRIPT
    assert "next_macro_contract.json next_macro_contract.md" in SCRIPT


def test_closed_loop_attaches_and_requires_macro_evidence():
    assert "candidates.json macro_candidates.json" in SCRIPT
    assert "next_macro_contract.json next_macro_contract.md; do" in SCRIPT
    assert "Declare the exact marker supplied by the contract" in SCRIPT
    assert "JIMU_AGENT_RETRIES" in SCRIPT


def test_pr5_generates_concrete_evidence_but_attaches_only_contract():
    assert "generate_agent_dag()" in SCRIPT
    assert '--seq-len ${SL2}' in SCRIPT
    assert '--seq-len ${SL6}' in SCRIPT
    assert "--seq-len 1" not in SCRIPT
    assert 'merge_dag_sequences.py' in SCRIPT
    assert '--dag "${SL2}=${short_dir}"' in SCRIPT
    assert '--dag "${SL6}=${output_dir}"' in SCRIPT
    for artifact in (
        "multiseq_metadata.json",
        "loop_invariants.json",
        "multiseq_summary.md",
        "candidate_evidence.jsonl",
        "allocation_proof.json",
        "allocation_summary.md",
    ):
        assert artifact in SCRIPT
    assert "micro_ops.jsonl edges.jsonl; do" not in SCRIPT
    assert 'CONTRACT_STATUS' in SCRIPT
    assert 'CONTRACT_ID' in SCRIPT
    assert "next_macro_contract.json is the authoritative selection" in SCRIPT
    assert "do not select another macro" in SCRIPT
    assert "multiseq_summary.md allocation_summary.md loop_invariants.json" not in SCRIPT


def test_pr6_generates_the_complete_bert_validation_matrix():
    for config in (
        '"2,4,${SL2},2"',
        '"2,4,${SL6},2"',
        '"4,4,${SL2},2"',
        '"4,4,${SL6},2"',
        '"4,8,${SL2},2"',
        '"4,8,${SL6},2"',
    ):
        assert config in SCRIPT
    assert '--required-config "${proof_id}"' in SCRIPT
    assert '--proof-dag "${proof_dir}"' in SCRIPT
