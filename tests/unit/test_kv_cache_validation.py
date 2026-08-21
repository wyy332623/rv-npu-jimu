import numpy as np

from tests.integration.test_bert_e2e import _find_cache_write_sequences


def test_cache_sequence_matching_ignores_unrelated_and_interleaved_writes():
    k_tiles = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ]
    writes = [
        (90, np.array([8.0, 8.0], dtype=np.float32)),
        (10, k_tiles[0].copy()),
        (92, np.array([7.0, 7.0], dtype=np.float32)),
        (12, k_tiles[1].copy()),
    ]

    matches = _find_cache_write_sequences(writes, k_tiles, [0, 2])

    assert [(item["base"], item["indices"]) for item in matches] == [
        (10, [1, 3])
    ]


def test_cache_sequence_matching_requires_values_and_order():
    tiles = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ]
    writes = [
        (12, tiles[1].copy()),
        (10, tiles[0].copy()),
        (12, np.array([9.0, 9.0], dtype=np.float32)),
    ]

    assert _find_cache_write_sequences(writes, tiles, [0, 2]) == []
