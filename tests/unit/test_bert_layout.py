from pathlib import Path

import pytest

from emulator.bert_layout import (
    LEGACY_LAYOUT,
    PACKED_LAYOUT,
    bert_dram_layout,
)
from emulator.workload import WorkloadManifest


def test_legacy_layout_preserves_existing_addresses():
    layout = bert_dram_layout(4, 4, 6, version=LEGACY_LAYOUT)

    assert layout.projection_base == 28
    assert layout.tile_stride == 8
    assert layout.save_q_base == 0x200
    assert layout.save_k_base == 0x300
    assert layout.save_v_base == 0x400
    assert layout.save_out_base == 0x800
    assert layout.unit_vec_base == 0x900


def test_packed_dim16_layout_is_non_overlapping_and_contiguous_per_position():
    layout = bert_dram_layout(16, 16, 16, version=PACKED_LAYOUT)

    assert layout.tile_stride == 16
    assert layout.position_span == 16
    assert layout.sequence_span == 256
    assert layout.position_address(layout.save_out_base, 15) + 16 <= layout.end_address
    ordered = sorted(
        (base, base + length, name)
        for name, (base, length) in layout.named_regions().items()
    )
    assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))
    assert layout.end_address < 524_288


def test_dim16_manifest_matches_layout_and_is_an_order_of_magnitude_larger():
    root = Path(__file__).resolve().parents[2]
    manifest = WorkloadManifest.load(
        root / "jimu-dse/workloads/bert-dim16-h16-seq16.yaml")
    layout = bert_dram_layout(16, 16, 16, version=PACKED_LAYOUT)
    tensors = {tensor.name: tensor for tensor in manifest.tensors}

    assert manifest.metadata["layout"] == PACKED_LAYOUT
    assert tensors["input_sequence"].address == layout.input_base
    assert tensors["input_sequence"].length == 16 * 16
    assert tensors["output_sequence"].address == layout.save_out_base
    assert tensors["output_sequence"].length == layout.sequence_span
    assert tensors["input_sequence"].length / (4 * 6) >= 10


@pytest.mark.parametrize(
    "dim,hidden,seq,error",
    [
        (16, 15, 16, "divisible"),
        (16, 16, 17, "seq_len"),
        (4, 12, 4, "at most two"),
    ],
)
def test_packed_layout_rejects_unsupported_shapes(dim, hidden, seq, error):
    with pytest.raises(ValueError, match=error):
        bert_dram_layout(dim, hidden, seq, version=PACKED_LAYOUT)
