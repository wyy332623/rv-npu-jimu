import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "jimu-dse" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

PROBE_PATH = SCRIPTS_DIR / "npu_workload_probe.py"
SPEC = importlib.util.spec_from_file_location("jimu_npu_workload_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def _write_build(tmp_path: Path, workload: str) -> tuple[Path, Path]:
    elf_path = tmp_path / "firmware.elf"
    elf_path.write_bytes(b"test-elf")
    metadata_path = tmp_path / "firmware.elf.jimu-build.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dim": 4,
                "hidden_size": 4,
                "seq_len": 24,
                "num_head": 1,
                "workload": workload,
                "elf_sha256": hashlib.sha256(elf_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return elf_path, metadata_path


def test_bert_probe_rejects_adder_build_metadata(tmp_path):
    elf_path, metadata_path = _write_build(tmp_path, "adder")
    config = probe.WorkloadConfig(4, 4, 24, 1)

    with pytest.raises(RuntimeError, match="firmware workload mismatch"):
        probe._validate_build_metadata(
            metadata_path, elf_path, config, "bert"
        )


def test_adder_probe_accepts_manifest_alias(tmp_path):
    elf_path, metadata_path = _write_build(tmp_path, "adder")
    config = probe.WorkloadConfig(4, 4, 24, 1)

    digest = probe._validate_build_metadata(
        metadata_path, elf_path, config, "adder_140p"
    )

    assert digest == hashlib.sha256(elf_path.read_bytes()).hexdigest()
