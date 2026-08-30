import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "jimu-dse" / "scripts"))

from dag_contract import build_contract, write_contract  # noqa: E402


def _macro(macro_id, level, *, eligible=True, rank=1):
    return {
        "id": macro_id,
        "level": level,
        "rank": rank,
        "eligible": eligible,
        "implementation_ready": eligible,
        "family": "test",
        "expected_dram_resources": [
            {
                "tensor_slice": "W_K[tile=0]",
                "dram_address": 48,
                "dram_address_hex": "0x30",
            }
        ],
        "allocation": {
            "allocation_proven": eligible,
            "cross_config_proven": eligible,
            "validation_matrix_complete": True,
            "config_results": [{"config_id": "dim2-h4-head2-seq6"}],
        },
        "estimated_saving": {"projected_seq6_bytes": 80},
    }


def _write_evidence(tmp_path, macros):
    (tmp_path / "macro_candidates.json").write_text(
        json.dumps({"macros": macros}), encoding="utf-8"
    )
    (tmp_path / "allocation_proof.json").write_text("{}", encoding="utf-8")


def test_contract_selects_lowest_ready_level(tmp_path):
    _write_evidence(
        tmp_path,
        [
            _macro("macro-dram-l3-kv-weight-stationary", "L3", rank=1),
            _macro("macro-dram-l2-loop-invariants", "L2", rank=2),
        ],
    )

    contract = build_contract(tmp_path)

    assert contract["status"] == "ready"
    assert contract["selected_macro"]["id"] == (
        "macro-dram-l2-loop-invariants"
    )
    assert contract["agent_policy"]["do_not_reprove"] is True
    assert contract["evidence_identity"]["macro_candidates_sha256"]


def test_contract_writer_blocks_without_complete_proof(tmp_path):
    _write_evidence(
        tmp_path,
        [_macro("macro-dram-l3-weight-stationary", "L3", eligible=False)],
    )

    paths = write_contract(tmp_path)
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))

    assert contract["status"] == "blocked-no-eligible-scope"
    assert contract["selected_macro"] is None
    assert paths["contract_summary"].is_file()


def test_contract_uses_workload_target_file_from_multiseq_metadata(tmp_path):
    _write_evidence(tmp_path, [_macro("macro-dram-l2-loop-invariants", "L2")])
    (tmp_path / "multiseq_metadata.json").write_text(
        json.dumps(
            {
                "workload": {
                    "name": "adder_140p",
                    "target_file": "adderboard/firmware/adder_140p.c",
                }
            }
        ),
        encoding="utf-8",
    )

    contract = build_contract(tmp_path)

    assert contract["agent_policy"]["edit_scope"] == (
        "adderboard/firmware/adder_140p.c only"
    )


def test_contract_preserves_q_retained_output_allocation(tmp_path):
    macro = _macro("macro-dram-l3-q-weight-stationary", "L3")
    macro["allocation"]["partial_sum_allocation"] = {
        "bank_name": "MEM_MVM_ACC_VRF",
        "base": 0,
        "end": 12,
    }
    macro["allocation"]["retained_output_allocation"] = {
        "bank_name": "MEM_MVM_ACC_VRF",
        "base": 12,
        "end": 36,
    }
    _write_evidence(tmp_path, [macro])

    selected = build_contract(tmp_path)["selected_macro"]

    assert selected["partial_sum_allocation"]["end"] == 12
    assert selected["retained_output_allocation"] == {
        "bank_name": "MEM_MVM_ACC_VRF",
        "base": 12,
        "end": 36,
    }


def test_contract_preserves_transient_scratch_allocation(tmp_path):
    macro = _macro("macro-dram-l1-transient-scratch-bank13", "L1")
    macro["allocation"]["transient_allocation"] = {
        "bank_name": "MEM_MVM_ACC_VRF",
        "base": 0,
        "end": 2,
        "retained_q_base": 12,
    }
    _write_evidence(tmp_path, [macro])

    selected = build_contract(tmp_path)["selected_macro"]

    assert selected["transient_allocation"] == {
        "bank_name": "MEM_MVM_ACC_VRF",
        "base": 0,
        "end": 2,
        "retained_q_base": 12,
    }


def test_contract_preserves_l3_state_allocations(tmp_path):
    macro = _macro("macro-dram-l3-self-output-weight-stationary", "L3")
    macro["allocation"]["state_allocations"] = [
        {"name": "attention-context", "base": 12, "end": 36},
        {"name": "self-output", "base": 36, "end": 60},
    ]
    _write_evidence(tmp_path, [macro])

    selected = build_contract(tmp_path)["selected_macro"]

    assert selected["state_allocations"] == [
        {"name": "attention-context", "base": 12, "end": 36},
        {"name": "self-output", "base": 36, "end": 60},
    ]


def test_contract_preserves_l3_l2_x_retention(tmp_path):
    macro = _macro("macro-dram-l3-ffn-intermediate-weight-stationary", "L3")
    macro["allocation"]["l2_x_retention"] = {
        "bank": 6,
        "bank_name": "MEM_MFU_INITIAL_VRF",
        "required_until": "residual2 after FFN_OUTPUT",
    }
    _write_evidence(tmp_path, [macro])
    contract = build_contract(tmp_path)

    selected = contract["selected_macro"]
    assert selected["l2_x_retention"] == {
        "bank": 6,
        "bank_name": "MEM_MFU_INITIAL_VRF",
        "required_until": "residual2 after FFN_OUTPUT",
    }


def test_contract_preserves_unit_vector_synthesis_plan(tmp_path):
    macro = _macro("macro-dram-l2-unit-vector-synthesis", "L2")
    macro["allocation"]["synthesis_plan"] = {
        "zero_immediate_fp16": "0x0000",
        "one_immediate_fp16": "0x3c00",
        "lane_selector": "REG_WRITE_VECTOR_MASK = 1 << j",
    }
    _write_evidence(tmp_path, [macro])

    selected = build_contract(tmp_path)["selected_macro"]
    assert selected["synthesis_plan"] == {
        "zero_immediate_fp16": "0x0000",
        "one_immediate_fp16": "0x3c00",
        "lane_selector": "REG_WRITE_VECTOR_MASK = 1 << j",
    }
