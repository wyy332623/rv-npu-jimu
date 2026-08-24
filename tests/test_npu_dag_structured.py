import hashlib
import json

from emulator.npu_dag_structured import (
    MULTISEQ_SCHEMA_NAME,
    MULTISEQ_SCHEMA_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_cross_config_allocation_proof,
    build_multiseq_dag,
    build_structured_dag,
    write_multiseq_dag,
    write_structured_dag,
)
from emulator.npu_micro_op_dag import MicroOp, build_micro_op_dag


def _two_position_k_projection():
    micro_ops = [
        MicroOp(
            "MAT_LOAD",
            "MAT_LOAD",
            [0, 1],
            [("MRF", 0)],
            [("DRAM", 0x0C)],
        ),
        MicroOp(
            "DRAM_STORE",
            "DRAM_STORE",
            [2],
            [("DRAM", 0x300)],
            [("MRF", 0)],
        ),
        MicroOp(
            "MAT_LOAD",
            "MAT_LOAD",
            [3, 4],
            [("MRF", 1)],
            [("DRAM", 0x20)],
        ),
        MicroOp(
            "DRAM_STORE",
            "DRAM_STORE",
            [5],
            [("DRAM", 0x310)],
            [("MRF", 1)],
        ),
    ]
    _, edges = build_micro_op_dag(micro_ops, pipe_edges=False)
    return micro_ops, edges


def test_structured_dag_adds_stable_ids_tensor_phase_and_patterns():
    micro_ops, edges = _two_position_k_projection()

    structured = build_structured_dag(
        micro_ops,
        edges,
        dim=2,
        hidden_size=4,
        seq_len=2,
        num_head=2,
        total_events=6,
    )

    metadata = structured["metadata"]
    assert metadata["schema"] == {
        "name": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
    }
    assert metadata["counts"]["micro_ops"] == 4
    assert metadata["counts"]["edges"] == 2

    assert [node["id"] for node in structured["micro_ops"]] == [
        "op-000000",
        "op-000001",
        "op-000002",
        "op-000003",
    ]
    assert structured["edges"][0]["source"] == "op-000000"
    assert structured["edges"][0]["target"] == "op-000001"

    assert {phase["kind"] for phase in structured["phases"]} == {
        "k_projection"
    }
    assert {
        tensor["slice"] for tensor in structured["tensors"]
    } >= {
        "K[pos=0,tile=0]",
        "K[pos=1,tile=0]",
    }

    k_pattern = next(
        pattern
        for pattern in structured["patterns"]
        if pattern["kind"] == "k_projection"
    )
    assert k_pattern["repeated"] is True
    assert k_pattern["instance_count"] == 2
    assert k_pattern["positions"] == [0, 1]


def test_structured_dag_writer_emits_jsonl_json_and_markdown(tmp_path):
    micro_ops, edges = _two_position_k_projection()
    elf = tmp_path / "bert.elf"
    elf.write_bytes(b"test-elf")

    paths = write_structured_dag(
        tmp_path / "dag",
        micro_ops,
        edges,
        dim=2,
        hidden_size=4,
        seq_len=2,
        num_head=2,
        total_events=6,
        elf_path=elf,
    )

    assert set(paths) == {
        "metadata",
        "micro_ops",
        "edges",
        "tensors",
        "phases",
        "patterns",
        "lifetimes",
        "candidates",
        "candidate_summary",
        "macro_candidates",
        "macro_candidate_summary",
        "summary",
    }
    assert all(path.is_file() for path in paths.values())

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["elf"]["sha256"] == hashlib.sha256(b"test-elf").hexdigest()

    node_lines = paths["micro_ops"].read_text(encoding="utf-8").splitlines()
    assert len(node_lines) == 4
    assert json.loads(node_lines[1])["phase_kind"] == "k_projection"

    summary = paths["summary"].read_text(encoding="utf-8")
    assert "# Structured NPU DAG Summary" in summary
    assert "## Repeated phase patterns" in summary
    assert "k_projection" in summary
    assert "`K[pos=0,tile=0]`" in summary
    assert "# DRAM Cache Candidate Summary" in summary


def test_transient_scratch_offset_is_tile_not_sequence_position():
    micro_ops = [
        MicroOp(
            "DRAM_LOAD",
            "DRAM_LOAD",
            [0, 1],
            [("VRF", 1, 0)],
            [("DRAM", 0x510)],
        )
    ]

    structured = build_structured_dag(
        micro_ops,
        [],
        dim=2,
        hidden_size=4,
        seq_len=1,
        num_head=2,
    )

    resource = structured["micro_ops"][0]["uses"][0]
    assert resource["tensor"] == "SCRATCH"
    assert resource["position"] is None
    assert resource["tile"] == 2
    assert resource["slice"] == "SCRATCH[tile=2]"


def _dram_round_trip(*, overwrite_source_vrf=False, overlap_source_vrf=False):
    micro_ops = [
        MicroOp(
            "VREG_LD",
            "VREG_LD",
            [0],
            [("VRF", 7, 0)],
            [],
        ),
        MicroOp(
            "DRAM_STORE",
            "DRAM_STORE",
            [1, 2],
            [("DRAM", 0x300)],
            [("VRF", 7, 0)],
        ),
    ]
    if overwrite_source_vrf:
        micro_ops.append(
            MicroOp(
                "VREG_LD",
                "VREG_LD",
                [3],
                [("VRF", 7, 0)],
                [],
            )
        )
    if overlap_source_vrf:
        micro_ops.append(
            MicroOp(
                "VREG_LD",
                "VREG_LD",
                [3],
                [("VRF", 7, 1)],
                [],
            )
        )
    micro_ops.append(
        MicroOp(
            "DRAM_LOAD",
            "DRAM_LOAD",
            [4, 5],
            [("VRF", 18, 0)],
            [("DRAM", 0x300)],
        )
    )
    _, edges = build_micro_op_dag(micro_ops, pipe_edges=False)
    return micro_ops, edges


def test_pr3_detects_round_trip_and_estimates_saving():
    micro_ops, edges = _dram_round_trip()

    structured = build_structured_dag(
        micro_ops,
        edges,
        dim=2,
        hidden_size=4,
        seq_len=1,
        num_head=2,
    )

    candidate_data = structured["candidates"]
    assert candidate_data["summary"] == {
        "eligible": 1,
        "high_confidence": 1,
        "requires_relocation": 0,
        "rejected": 0,
    }
    candidate = candidate_data["candidates"][0]
    assert candidate["tensor_slice"] == "K[pos=0,tile=0]"
    assert candidate["producer"] == "op-000001"
    assert candidate["consumers"] == ["op-000002"]
    assert candidate["intervening_write"] is False
    assert candidate["proof"]["def_use_edges"]
    assert candidate["vrf_plan"]["reuse_source_resource"] is True
    assert candidate["vrf_plan"]["allocation_proven"] is True
    assert candidate["stable_id"].startswith("candidate-dram-stable-")
    macro = structured["macro_candidates"]["macros"][0]
    assert macro["id"] == "macro-dram-l1-attention"
    assert macro["eligible"] is True
    assert macro["member_candidate_ids"] == [candidate["id"]]
    allocation = macro["allocation"]["regions"][0]
    assert allocation["allocation_proven"] is True
    assert allocation["base"] % 2 == 0
    assert allocation["end"] <= allocation["capacity_elements"]
    assert candidate["estimated_saving"]["observed_graph_bytes"] == 16
    assert candidate["estimated_saving"]["projected_seq2_bytes"] == 48
    assert candidate["estimated_saving"]["projected_seq6_bytes"] == 336
    assert (
        candidate["estimated_saving"]["projection_assumption"]
        == "K/V heuristic: seq stores plus seq^2 attention loads"
    )


def test_pr3_marks_source_vrf_overwrite_as_relocation_required():
    micro_ops, edges = _dram_round_trip(overwrite_source_vrf=True)

    structured = build_structured_dag(
        micro_ops,
        edges,
        dim=2,
        hidden_size=4,
        seq_len=1,
        num_head=2,
    )

    candidate = structured["candidates"]["candidates"][0]
    assert candidate["status"] == "eligible-requires-vrf-relocation"
    assert candidate["confidence"] == "medium"
    assert candidate["vrf_plan"]["reuse_source_resource"] is False
    assert candidate["vrf_plan"]["source_overwrite_nodes"] == [
        "op-000002"
    ]
    assert candidate["vrf_plan"]["required_vector_slots"] == 1
    assert candidate["vrf_plan"]["allocation_proven"] is False
    macro = structured["macro_candidates"]["macros"][0]
    assert macro["eligible"] is True
    allocation = macro["allocation"]["regions"][0]
    assert allocation["bank"] == 6
    assert allocation["reuse_source_resource"] is False
    assert allocation["allocation_proven"] is True


def test_macro_rejects_partially_overlapping_source_vrf_region():
    micro_ops, edges = _dram_round_trip(overlap_source_vrf=True)
    structured = build_structured_dag(
        micro_ops,
        edges,
        dim=2,
        hidden_size=4,
        seq_len=1,
        num_head=2,
    )

    primitive = structured["candidates"]["candidates"][0]
    assert primitive["vrf_plan"]["reuse_source_resource"] is True
    macro = structured["macro_candidates"]["macros"][0]
    assert macro["eligible"] is False
    assert macro["allocation"]["allocation_proven"] is False


def test_pr3_rejects_store_overwritten_before_any_load():
    micro_ops = [
        MicroOp(
            "DRAM_STORE",
            "DRAM_STORE",
            [0],
            [("DRAM", 0x300)],
            [("VRF", 7, 0)],
        ),
        MicroOp(
            "DRAM_STORE",
            "DRAM_STORE",
            [1],
            [("DRAM", 0x300)],
            [("VRF", 8, 0)],
        ),
        MicroOp(
            "DRAM_LOAD",
            "DRAM_LOAD",
            [2, 3],
            [("VRF", 18, 0)],
            [("DRAM", 0x300)],
        ),
    ]
    _, edges = build_micro_op_dag(micro_ops, pipe_edges=False)

    structured = build_structured_dag(
        micro_ops,
        edges,
        dim=2,
        hidden_size=4,
        seq_len=1,
        num_head=2,
    )

    candidate_data = structured["candidates"]
    assert candidate_data["summary"]["eligible"] == 1
    assert candidate_data["summary"]["rejected"] == 1
    rejected = candidate_data["rejected"][0]
    assert rejected["producer"] == "op-000000"
    assert rejected["next_write"] == "op-000001"
    assert rejected["eligible"] is False
    assert rejected["rejection_reasons"] == [
        "no downstream consumer before overwrite or graph end"
    ]


def _readonly_reuse_ops(seq_len):
    dim = 2
    hidden_size = 4
    proj_base = hidden_size * seq_len + 4
    mat_size = hidden_size * hidden_size
    bias_q = proj_base + mat_size
    micro_ops = []
    event_id = 0
    for pos in range(seq_len):
        for duplicate in range(2):
            micro_ops.append(
                MicroOp(
                    "DRAM_LOAD",
                    "DRAM_LOAD",
                    [event_id],
                    [("VRF", 1 + duplicate, 0)],
                    [("DRAM", pos * hidden_size)],
                )
            )
            event_id += 1
        micro_ops.append(
            MicroOp(
                "DRAM_LOAD",
                "DRAM_LOAD",
                [event_id],
                [("VRF", 3, 0)],
                [("DRAM", bias_q)],
            )
        )
        event_id += 1
        micro_ops.append(
            MicroOp(
                "MAT_LOAD",
                "MAT_LOAD",
                [event_id],
                [("MRF", 0)],
                [("DRAM", proj_base)],
            )
        )
        event_id += 1
    return micro_ops


def _readonly_reuse_graph(seq_len):
    micro_ops = _readonly_reuse_ops(seq_len)
    return build_structured_dag(
        micro_ops,
        [],
        dim=2,
        hidden_size=4,
        seq_len=seq_len,
        num_head=2,
        total_events=len(micro_ops),
    )


def _readonly_kv_graph(seq_len):
    dim = 2
    hidden_size = 4
    proj_base = hidden_size * seq_len + 4
    stride = hidden_size * hidden_size + hidden_size
    micro_ops = []
    event_id = 0
    for _pos in range(seq_len):
        for matrix_base in (proj_base + stride, proj_base + 2 * stride):
            for tile in range(4):
                micro_ops.append(
                    MicroOp(
                        "MAT_LOAD",
                        "MAT_LOAD",
                        [event_id],
                        [("MRF", 0)],
                        [("DRAM", matrix_base + tile * dim * dim)],
                    )
                )
                event_id += 1
    return build_structured_dag(
        micro_ops,
        [],
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        num_head=2,
        total_events=len(micro_ops),
    )


def _readonly_q_graph(seq_len):
    dim = 2
    hidden_size = 4
    proj_base = hidden_size * seq_len + 4
    micro_ops = []
    event_id = 0
    for _pos in range(seq_len):
        for tile in range(4):
            micro_ops.append(
                MicroOp(
                    "MAT_LOAD",
                    "MAT_LOAD",
                    [event_id],
                    [("MRF", 0)],
                    [("DRAM", proj_base + tile * dim * dim)],
                )
            )
            event_id += 1
    return build_structured_dag(
        micro_ops,
        [],
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        num_head=2,
        total_events=len(micro_ops),
    )


def _transient_scratch_graph(
    seq_len,
    *,
    bank13_conflict=False,
    include_softmax=True,
    include_layernorm=True,
    softmax_pairs_per_position=2,
):
    micro_ops = []
    event_id = 0
    for _pos in range(seq_len):
        addresses = []
        if include_softmax:
            addresses.extend([0x500] * softmax_pairs_per_position)
        if include_layernorm:
            addresses.extend((0x510, 0x510))
        for address in addresses:
            micro_ops.append(
                MicroOp(
                    "DRAM_STORE",
                    "DRAM_STORE",
                    [event_id],
                    [("DRAM", address)],
                    [],
                )
            )
            event_id += 1
            if bank13_conflict and event_id == 1:
                micro_ops.append(
                    MicroOp(
                        "VREG_LD",
                        "VREG_LD",
                        [event_id],
                        [("VRF", 13, 0)],
                        [],
                    )
                )
                event_id += 1
            micro_ops.append(
                MicroOp(
                    "DRAM_LOAD",
                    "DRAM_LOAD",
                    [event_id],
                    [],
                    [("DRAM", address)],
                )
            )
            event_id += 1
    return build_structured_dag(
        micro_ops,
        [],
        dim=2,
        hidden_size=4,
        seq_len=seq_len,
        num_head=2,
        total_events=len(micro_ops),
    )


def _readonly_self_output_graph(seq_len):
    dim = 2
    hidden_size = 4
    proj_base = hidden_size * seq_len + 4
    stride = hidden_size * hidden_size + hidden_size
    matrix_base = proj_base + 3 * stride
    micro_ops = []
    event_id = 0
    for _pos in range(seq_len):
        for tile in range(4):
            micro_ops.append(
                MicroOp(
                    "MAT_LOAD",
                    "MAT_LOAD",
                    [event_id],
                    [("MRF", 0)],
                    [("DRAM", matrix_base + tile * dim * dim)],
                )
            )
            event_id += 1
    return build_structured_dag(
        micro_ops,
        [],
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        num_head=2,
        total_events=len(micro_ops),
    )


def _readonly_ffn_intermediate_graph(seq_len):
    dim = 2
    hidden_size = 4
    proj_base = hidden_size * seq_len + 4
    stride = hidden_size * hidden_size + hidden_size
    matrix_base = proj_base + 4 * stride
    micro_ops = []
    event_id = 0
    for _pos in range(seq_len):
        for tile in range(4):
            micro_ops.append(
                MicroOp(
                    "MAT_LOAD",
                    "MAT_LOAD",
                    [event_id],
                    [("MRF", 0)],
                    [("DRAM", matrix_base + tile * dim * dim)],
                )
            )
            event_id += 1
    return build_structured_dag(
        micro_ops,
        [],
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        num_head=2,
        total_events=len(micro_ops),
    )


def _readonly_ffn_output_graph(seq_len):
    dim = 2
    hidden_size = 4
    proj_base = hidden_size * seq_len + 4
    stride = hidden_size * hidden_size + hidden_size
    matrix_base = proj_base + 5 * stride
    micro_ops = []
    event_id = 0
    for _pos in range(seq_len):
        for tile in range(4):
            micro_ops.append(
                MicroOp(
                    "MAT_LOAD",
                    "MAT_LOAD",
                    [event_id],
                    [("MRF", 0)],
                    [("DRAM", matrix_base + tile * dim * dim)],
                )
            )
            event_id += 1
    return build_structured_dag(
        micro_ops,
        [],
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        num_head=2,
        total_events=len(micro_ops),
    )


def _unit_vector_graph(seq_len):
    dim = 2
    hidden_size = 4
    micro_ops = [
        MicroOp(
            "DRAM_LOAD",
            "DRAM_LOAD",
            [j],
            [("VRF", 6, 60 + j * dim)],
            [("DRAM", 0x900 + j * dim)],
        )
        for j in range(2)
    ]
    return build_structured_dag(
        micro_ops,
        [],
        dim=dim,
        hidden_size=hidden_size,
        seq_len=seq_len,
        num_head=2,
        total_events=len(micro_ops),
    )


def test_pr5_measures_cross_sequence_loop_invariants_without_projection():
    analysis = build_multiseq_dag(
        {2: _readonly_reuse_graph(2), 6: _readonly_reuse_graph(6)}
    )

    assert analysis["schema"] == {
        "name": MULTISEQ_SCHEMA_NAME,
        "version": MULTISEQ_SCHEMA_VERSION,
    }
    assert analysis["reference_seq_len"] == 6
    assert analysis["summary"] == {
        "candidates": 3,
        "analysis_eligible": 3,
        "implementation_ready": 0,
        "families": 3,
    }

    candidates = {
        candidate["semantic_key"]: candidate
        for candidate in analysis["candidates"]
    }
    input_tile = candidates["X[tile=0]"]
    assert input_tile["per_config"]["seq2"]["read_ops"] == 4
    assert input_tile["per_config"]["seq2"]["unique_values"] == 2
    assert input_tile["per_config"]["seq2"]["removable_read_bytes"] == 16
    assert input_tile["per_config"]["seq6"]["read_ops"] == 12
    assert input_tile["per_config"]["seq6"]["unique_values"] == 6
    assert input_tile["per_config"]["seq6"]["removable_read_bytes"] == 48

    bias = candidates["B_Q[tile=0]"]
    assert bias["per_config"]["seq2"]["removable_read_bytes"] == 8
    assert bias["per_config"]["seq6"]["removable_read_bytes"] == 40
    weight = candidates["W_Q[tile=0]"]
    assert weight["per_config"]["seq2"]["removable_read_bytes"] == 16
    assert weight["per_config"]["seq6"]["removable_read_bytes"] == 80

    assert all(
        candidate["cross_sequence_proof"]["measured_not_projected"]
        for candidate in candidates.values()
    )
    assert not any(
        candidate["implementation_ready"]
        for candidate in candidates.values()
    )
    assert {family["id"] for family in analysis["families"]} == {
        "macro-dram-l2-loop-invariants",
        "macro-dram-l2-sequence-input",
        "macro-dram-l3-weight-stationary",
    }


def test_pr5_rejects_mismatched_graph_geometry():
    seq6 = _readonly_reuse_graph(6)
    seq6["metadata"]["firmware_config"]["hidden_size"] = 8

    try:
        build_multiseq_dag({2: _readonly_reuse_graph(2), 6: seq6})
    except ValueError as exc:
        assert "geometry differs" in str(exc)
    else:
        raise AssertionError("mismatched DAG geometry must be rejected")


def test_pr5_writer_emits_compact_multisequence_evidence(tmp_path):
    sequence_dirs = {}
    for seq_len in (2, 6):
        dag_dir = tmp_path / f"seq{seq_len}"
        micro_ops = _readonly_reuse_ops(seq_len)
        write_structured_dag(
            dag_dir,
            micro_ops,
            [],
            dim=2,
            hidden_size=4,
            seq_len=seq_len,
            num_head=2,
            total_events=len(micro_ops),
        )
        sequence_dirs[seq_len] = dag_dir

    paths = write_multiseq_dag(tmp_path / "merged", sequence_dirs)

    assert set(paths) == {
        "metadata",
        "loop_invariants",
        "summary",
        "candidate_evidence",
        "allocation_proof",
        "allocation_summary",
    }
    assert all(path.is_file() for path in paths.values())
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "Concrete sequence DAGs" in summary
    assert "macro-dram-l3-weight-stationary" in summary
    evidence = [
        json.loads(line)
        for line in paths["candidate_evidence"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(item["evidence_type"] == "node" for item in evidence)
    allocation = json.loads(
        paths["allocation_proof"].read_text(encoding="utf-8")
    )
    assert allocation["validation_matrix_complete"] is True
    assert all(item["cross_config_proven"] for item in allocation["l2"])
    allocation_summary = paths["allocation_summary"].read_text(
        encoding="utf-8"
    )
    assert "Cross-Configuration Allocation Summary" in allocation_summary
    assert "Reference allocation regions" in allocation_summary


def test_pr6_allocates_l2_around_existing_vrf_lifetimes():
    seq2 = _readonly_reuse_graph(2)
    seq6 = _readonly_reuse_graph(6)
    seq6["lifetimes"].append(
        {
            "id": "occupied-cache-row-zero",
            "resource": {"space": "VRF", "bank": 6, "row": 0},
            "interval": {
                "start_node": "op-000000",
                "end_node": "op-999999",
                "start_index": 0,
                "end_index": 999999,
                "span_nodes": 1000000,
            },
            "producer": "op-000000",
            "consumers": [],
            "next_definition": None,
            "phase_id": "phase-0000",
        }
    )
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    assert proof["validation_matrix_complete"] is True
    macro_by_id = {macro["id"]: macro for macro in macros["macros"]}
    for macro_id in (
        "macro-dram-l2-loop-invariants",
        "macro-dram-l2-sequence-input",
    ):
        macro = macro_by_id[macro_id]
        assert macro["eligible"] is True
        assert macro["allocation"]["cross_config_proven"] is True
        assert macro["allocation"]["validation_matrix_complete"] is True
        assert all(
            region["base"] % region["alignment_elements"] == 0
            and region["end"] <= region["capacity_elements"]
            for region in macro["allocation"]["regions"]
        )
    assert all(
        region["base"] >= 2
        for region in macro_by_id["macro-dram-l2-loop-invariants"][
            "allocation"
        ]["regions"]
    )
    l3 = macro_by_id["macro-dram-l3-q-weight-stationary"]
    assert l3["eligible"] is False
    assert l3["allocation"]["cross_config_proven"] is False


def test_pr6_blocks_macro_when_required_configuration_is_missing():
    seq2 = _readonly_reuse_graph(2)
    seq6 = _readonly_reuse_graph(6)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
            "dim4-h4-head2-seq6",
        ],
    )

    assert proof["validation_matrix_complete"] is False
    assert proof["missing_config_ids"] == ["dim4-h4-head2-seq6"]
    assert not any(macro["eligible"] for macro in macros["macros"])


def test_l3_kv_schedule_is_cross_config_proven_and_bounded():
    seq2 = _readonly_kv_graph(2)
    seq6 = _readonly_kv_graph(6)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    macro = by_id["macro-dram-l3-kv-weight-stationary"]
    assert macro["eligible"] is True
    assert macro["implementation_ready"] is True
    resources = macro["expected_dram_resources"]
    assert len(resources) == 8
    assert {item["tensor_slice"].split("[")[0] for item in resources} == {
        "W_K",
        "W_V",
    }
    partial = macro["allocation"]["partial_sum_allocation"]
    assert partial["bank_name"] == "MEM_MVM_ACC_VRF"
    assert partial["end"] == 12
    assert partial["end"] <= partial["capacity_elements"]
    assert any(item["id"] == macro["id"] for item in proof["l3"])


def test_l3_q_schedule_retains_outputs_in_disjoint_bounded_region():
    seq2 = _readonly_q_graph(2)
    seq6 = _readonly_q_graph(6)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    macro = by_id["macro-dram-l3-q-weight-stationary"]
    assert macro["eligible"] is True
    assert macro["implementation_ready"] is True
    assert len(macro["expected_dram_resources"]) == 4
    assert macro["estimated_saving"]["projected_seq2_bytes"] == 64
    assert macro["estimated_saving"]["projected_seq6_bytes"] == 320
    partial = macro["allocation"]["partial_sum_allocation"]
    retained = macro["allocation"]["retained_output_allocation"]
    assert partial["bank_name"] == "MEM_MVM_ACC_VRF"
    assert retained["bank_name"] == "MEM_MVM_ACC_VRF"
    assert partial["end"] == retained["base"] == 12
    assert retained["end"] == 36
    assert retained["end"] <= retained["capacity_elements"]
    assert any(item["id"] == macro["id"] for item in proof["l3"])


def test_transient_scratch_macro_is_exact_bounded_and_cross_config_proven():
    seq2 = _transient_scratch_graph(2)
    seq6 = _transient_scratch_graph(6)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    assert len(by_id) == len(macros["macros"])
    macro = by_id["macro-dram-l1-transient-scratch-bank13"]
    assert macro["eligible"] is True
    assert macro["implementation_ready"] is True
    assert macro["estimated_saving"]["projected_seq2_bytes"] == 128
    assert macro["estimated_saving"]["projected_seq6_bytes"] == 384
    assert {
        (item["tensor_slice"], item["dram_address"])
        for item in macro["expected_dram_resources"]
    } == {
        ("SCRATCH[tile=0]", 0x500),
        ("SCRATCH[tile=2]", 0x510),
    }
    allocation = macro["allocation"]["transient_allocation"]
    assert allocation["bank_name"] == "MEM_MVM_ACC_VRF"
    assert allocation["base"] == 0
    assert allocation["end"] == 2
    assert allocation["retained_q_base"] == 12
    assert allocation["disjoint_from_retained_q"] is True
    assert any(item["id"] == macro["id"] for item in proof["l1"])


def test_transient_scratch_macro_blocks_bank13_conflict():
    seq2 = _transient_scratch_graph(2)
    seq6 = _transient_scratch_graph(6, bank13_conflict=True)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    _proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    assert len(by_id) == len(macros["macros"])
    macro = by_id["macro-dram-l1-transient-scratch-bank13"]
    assert macro["eligible"] is False
    assert macro["allocation"]["cross_config_proven"] is False


def test_transient_scratch_macro_omits_already_eliminated_family():
    seq2 = _transient_scratch_graph(2, include_softmax=False)
    seq6 = _transient_scratch_graph(6, include_softmax=False)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    macro = by_id["macro-dram-l1-transient-scratch-bank13"]
    assert macro["eligible"] is True
    assert macro["implementation_ready"] is True
    assert macro["estimated_saving"]["projected_seq2_bytes"] == 64
    assert macro["estimated_saving"]["projected_seq6_bytes"] == 192
    assert macro["expected_dram_resources"] == [
        {
            "tensor_slice": "SCRATCH[tile=2]",
            "dram_address": 0x510,
            "dram_address_hex": "0x510",
        }
    ]
    scratch_proof = next(
        item
        for item in proof["l1"]
        if item["id"] == "macro-dram-l1-transient-scratch-bank13"
    )
    assert all(
        result["allocation_proven"] for result in scratch_proof["config_results"]
    )
    assert all(
        result["eliminated_regions"]
        == [
            {
                "tensor": "SCRATCH",
                "tensor_slice": None,
                "purpose": "softmax-probability",
                "dram_address": 0x500,
                "dram_address_hex": "0x500",
                "store_ops": 0,
                "load_ops": 0,
                "expected_pairs": result["firmware_config"]["seq_len"]
                * result["firmware_config"]["num_head"],
                "observed_pairs": 0,
                "removable_read_write_bytes": 0,
                "status": "already-eliminated",
            }
        ]
        for result in scratch_proof["config_results"]
    )


def test_transient_scratch_macro_is_absent_when_all_families_are_eliminated():
    seq2 = _transient_scratch_graph(
        2, include_softmax=False, include_layernorm=False
    )
    seq6 = _transient_scratch_graph(
        6, include_softmax=False, include_layernorm=False
    )
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    _proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    assert "macro-dram-l1-transient-scratch-bank13" not in {
        macro["id"] for macro in macros["macros"]
    }


def test_transient_scratch_macro_does_not_accept_partially_observed_family():
    seq2 = _transient_scratch_graph(2, softmax_pairs_per_position=1)
    seq6 = _transient_scratch_graph(6, softmax_pairs_per_position=1)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    _proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    macro = by_id["macro-dram-l1-transient-scratch-bank13"]
    assert macro["eligible"] is False
    assert macro["allocation"]["cross_config_proven"] is False


def test_l3_self_output_uses_two_bounded_state_buffers():
    seq2 = _readonly_self_output_graph(2)
    seq6 = _readonly_self_output_graph(6)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    macro = by_id["macro-dram-l3-self-output-weight-stationary"]
    assert macro["eligible"] is True
    assert macro["estimated_saving"]["projected_seq2_bytes"] == 64
    assert macro["estimated_saving"]["projected_seq6_bytes"] == 320
    states = macro["allocation"]["state_allocations"]
    assert [(state["base"], state["end"]) for state in states] == [
        (12, 36),
        (36, 60),
    ]
    assert all(state["end"] <= 256 for state in states)
    assert any(item["id"] == macro["id"] for item in proof["l3"])


def test_l3_ffn_intermediate_reuses_state_buffers_and_retains_x():
    seq2 = _readonly_ffn_intermediate_graph(2)
    seq6 = _readonly_ffn_intermediate_graph(6)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    macro = by_id[
        "macro-dram-l3-ffn-intermediate-weight-stationary"
    ]
    assert macro["eligible"] is True
    assert macro["estimated_saving"]["projected_seq2_bytes"] == 64
    assert macro["estimated_saving"]["projected_seq6_bytes"] == 320
    states = macro["allocation"]["state_allocations"]
    assert [(state["base"], state["end"]) for state in states] == [
        (12, 36),
        (36, 60),
    ]
    assert macro["allocation"]["l2_x_retention"] == {
        "bank": 6,
        "bank_name": "MEM_MFU_INITIAL_VRF",
        "producer": "macro-dram-l2-sequence-input",
        "required_until": "residual2 after FFN_OUTPUT",
        "disjoint_from_l3_state_bank": True,
        "source_invariant_required": True,
    }
    assert any(item["id"] == macro["id"] for item in proof["l3"])


def test_l3_ffn_output_reuses_state_buffers_and_reaches_weight_floor():
    seq2 = _readonly_ffn_output_graph(2)
    seq6 = _readonly_ffn_output_graph(6)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    macro = by_id["macro-dram-l3-ffn-output-weight-stationary"]
    assert macro["eligible"] is True
    assert macro["estimated_saving"]["projected_seq2_bytes"] == 64
    assert macro["estimated_saving"]["projected_seq6_bytes"] == 320
    states = macro["allocation"]["state_allocations"]
    assert [(state["base"], state["end"]) for state in states] == [
        (12, 36),
        (36, 60),
    ]
    assert macro["allocation"]["l2_x_retention"]["required_until"] == (
        "residual2 after FFN_OUTPUT"
    )
    assert not any(
        item["id"] == "macro-dram-l3-weight-stationary"
        for item in macros["macros"]
    )
    assert any(item["id"] == macro["id"] for item in proof["l3"])


def test_l2_unit_vectors_are_exactly_synthesizable_without_dram():
    seq2 = _unit_vector_graph(2)
    seq6 = _unit_vector_graph(6)
    analysis = build_multiseq_dag({2: seq2, 6: seq6})
    proof, macros = build_cross_config_allocation_proof(
        seq6,
        [seq2, seq6],
        analysis,
        required_config_ids=[
            "dim2-h4-head2-seq2",
            "dim2-h4-head2-seq6",
        ],
    )

    by_id = {macro["id"]: macro for macro in macros["macros"]}
    macro = by_id["macro-dram-l2-unit-vector-synthesis"]
    assert macro["eligible"] is True
    assert macro["estimated_saving"]["projected_seq2_bytes"] == 16
    assert macro["estimated_saving"]["projected_seq6_bytes"] == 16
    assert {
        (item["tensor_slice"], item["dram_address"])
        for item in macro["expected_dram_resources"]
    } == {
        ("UNIT_VEC[tile=0]", 0x900),
        ("UNIT_VEC[tile=1]", 0x902),
    }
    plan = macro["allocation"]["synthesis_plan"]
    assert plan["zero_immediate_fp16"] == "0x0000"
    assert plan["one_immediate_fp16"] == "0x3c00"
    assert plan["lane_selector"] == "REG_WRITE_VECTOR_MASK = 1 << j"
    assert any(item["id"] == macro["id"] for item in proof["l2"])
