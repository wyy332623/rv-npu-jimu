import pytest

from emulator.npu_dag_adapter import get_dag_workload
from emulator.npu_dag_structured import (
    build_cross_config_allocation_proof,
    build_multiseq_dag,
    build_structured_dag,
)
from emulator.npu_micro_op_dag import MicroOp


def _config(seq_len=24):
    return {
        "dim": 4,
        "hidden_size": 4,
        "seq_len": seq_len,
        "num_head": 1,
    }


def test_bert_adapter_preserves_dynamic_low_address_layout():
    adapter = get_dag_workload(
        "bert",
        {"dim": 2, "hidden_size": 4, "seq_len": 6, "num_head": 2},
    )
    assert adapter.classify_dram(0)["slice"] == "X[pos=0,tile=0]"
    assert adapter.classify_dram(4)["slice"] == "X[pos=1,tile=0]"
    assert adapter.classify_dram(28)["tensor"] == "W_Q"
    assert adapter.metadata()["specialized_contracts"] is True


def test_adder_adapter_decodes_weights_control_and_both_scratch_phases():
    adapter = get_dag_workload("adder_140p", _config())
    assert adapter.classify_dram(0xD00) == {
        "tensor": "W_Q_T",
        "role": "weight",
        "position": None,
        "tile": 0,
        "slice": "W_Q_T[tile=0]",
        "position_in_config": True,
        "reuse_family": "weight-stationary",
        "cache_level": "L3",
        "reuse_kind": "weight-stationary-load",
    }
    assert adapter.classify_dram(0x1F00)["tensor"] == "PHASE_FLAG"
    assert adapter.classify_dram(0x2000 + 7 * 4)["slice"] == "X[pos=7]"
    assert adapter.classify_dram(0x3000 + 3 * 4)["slice"] == (
        "FFN_NORM_INPUT[pos=3]"
    )
    assert adapter.metadata()["specialized_contracts"] is False


def test_unknown_adapter_fails_closed_instead_of_using_bert_layout():
    with pytest.raises(ValueError, match="unknown DAG workload"):
        get_dag_workload("not-a-model", _config())


def _adder_graph(seq_len, repeated_weight_reads=2):
    micro_ops = []
    event_index = 0
    for _ in range(repeated_weight_reads):
        micro_ops.append(
            MicroOp(
                kind="MAT_LOAD",
                name="MAT_LOAD",
                event_indices=[event_index],
                defs=[("MRF", event_index)],
                uses=[("DRAM", 0xD00)],
            )
        )
        event_index += 1
    phase1_end = event_index
    micro_ops.extend(
        [
            MicroOp(
                kind="DRAM_LOAD",
                name="DRAM_LOAD",
                event_indices=[event_index],
                defs=[("VRF", 1, 0)],
                uses=[("DRAM", 0x3000)],
            ),
            MicroOp(
                kind="DRAM_STORE",
                name="DRAM_STORE",
                event_indices=[event_index + 1],
                defs=[("DRAM", 0x4000)],
                uses=[("VRF", 1, 0)],
            ),
        ]
    )
    return build_structured_dag(
        micro_ops,
        [],
        dim=4,
        hidden_size=4,
        seq_len=seq_len,
        num_head=1,
        total_events=event_index + 2,
        workload="adder_140p",
        phase_ranges=[
            {
                "kind": "attention_and_residual",
                "label": "phase1",
                "event_start": 0,
                "event_end": phase1_end,
            },
            {
                "kind": "swiglu_ffn_and_residual",
                "label": "phase2",
                "event_start": phase1_end,
                "event_end": event_index + 2,
            },
        ],
    )


def test_structured_dag_records_workload_and_authoritative_phase_boundaries():
    graph = _adder_graph(24)
    assert graph["metadata"]["workload"]["name"] == "adder_140p"
    assert graph["metadata"]["annotation"]["phase_boundaries_authoritative"]
    assert [phase["kind"] for phase in graph["phases"]] == [
        "attention_and_residual",
        "swiglu_ffn_and_residual",
    ]
    assert graph["micro_ops"][0]["uses"][0]["tensor"] == "W_Q_T"
    assert graph["micro_ops"][-1]["defs"][0]["tensor"] == "FINAL_HIDDEN"


def test_non_bert_cross_config_proof_uses_generic_mode_and_blocks_l3():
    seq24 = _adder_graph(24, repeated_weight_reads=2)
    seq25 = _adder_graph(25, repeated_weight_reads=3)
    analysis = build_multiseq_dag({24: seq24, 25: seq25})
    required = [
        "dim4-h4-head1-seq24",
        "dim4-h4-head1-seq25",
    ]
    proof, macros = build_cross_config_allocation_proof(
        seq25,
        [seq24, seq25],
        analysis,
        required_config_ids=required,
    )
    assert proof["proof_mode"] == "generic-conservative"
    assert proof["validation_matrix_complete"] is True
    weight_macro = next(
        macro
        for macro in macros["macros"]
        if macro["id"] == "macro-dram-l3-weight-stationary"
    )
    assert weight_macro["eligible"] is False
    assert weight_macro["status"] == "blocked-workload-schedule-proof-required"
