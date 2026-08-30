import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VISUALIZE_PATH = REPO_ROOT / "jimu-dse" / "scripts" / "visualize_graph.py"
SPEC = importlib.util.spec_from_file_location("jimu_visualize_graph", VISUALIZE_PATH)
assert SPEC and SPEC.loader
visualize_graph = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = visualize_graph
SPEC.loader.exec_module(visualize_graph)


def test_graph_build_uses_sequence_specific_isolated_directory(
    tmp_path, monkeypatch
):
    firmware_dir = tmp_path / "firmware"
    firmware_dir.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(visualize_graph, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(visualize_graph.subprocess, "run", fake_run)

    elf = visualize_graph.build_firmware(
        dim=2,
        hidden_size=4,
        seq_len=6,
        num_head=2,
    )

    assert calls
    assert "BUILD_DIR=build_graph_dim2_h4_seq6" in calls[0]
    assert "BUILD_DIR=build_dim2" not in calls[0]
    assert elf.endswith("firmware/build_graph_dim2_h4_seq6/bert.elf")


def test_weight_layout_reserves_all_sequence_inputs():
    from emulator.npu_device_mini import MEM_DRAM

    dim = 2
    hidden_size = 4
    seq_len = 6
    dram = np.full(0x1000, -99.0, dtype=np.float32)
    npu = SimpleNamespace(_vrf={MEM_DRAM: dram})
    params = {
        name: {
            "W": np.full((hidden_size, hidden_size), value, dtype=np.float32),
            "b": np.full(hidden_size, value + 0.5, dtype=np.float32),
        }
        for name, value in {
            "Q": 1.0,
            "K": 2.0,
            "V": 3.0,
            "selfoutput": 4.0,
        }.items()
    }
    params.update(
        {
            "W_intmfc": np.full((4, 4), 5.0, dtype=np.float32),
            "b_intmfc": np.full(4, 5.5, dtype=np.float32),
            "W_outfc": np.full((4, 4), 6.0, dtype=np.float32),
            "b_outfc": np.full(4, 6.5, dtype=np.float32),
            "LayerNorm": {
                "W": np.ones((2, 4), dtype=np.float32),
                "b": np.zeros((2, 4), dtype=np.float32),
            },
        }
    )

    visualize_graph.load_weights(npu, params, dim, hidden_size, seq_len)

    input_elements = hidden_size * seq_len
    proj_base = input_elements + 4
    assert np.all(dram[:input_elements] == 0.0)
    assert np.all(dram[input_elements:proj_base] == -99.0)
    assert np.all(dram[proj_base:proj_base + 16] == 1.0)
    assert np.all(dram[proj_base + 16:proj_base + 20] == 1.5)
