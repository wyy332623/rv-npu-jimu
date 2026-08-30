import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLCTL_PATH = REPO_ROOT / "jimu-dse" / "scripts" / "skillctl.py"
SPEC = importlib.util.spec_from_file_location("jimu_skillctl", SKILLCTL_PATH)
assert SPEC and SPEC.loader
skillctl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = skillctl
SPEC.loader.exec_module(skillctl)


def test_skill_tree_is_synchronized_and_versioned():
    skills = skillctl.verify_skills()

    assert "common-constraints" in skills
    assert "dim-optimize" in skills
    assert "vrf-cache" in skills
    assert skills["self-verify"].version == "2.0.0"
    assert all(skillctl.SEMVER_RE.fullmatch(skill.version) for skill in skills.values())


def test_manifest_preserves_effective_skill_order(tmp_path):
    output = tmp_path / "skills_manifest.json"
    names = [
        "common-constraints",
        "dag-analyze",
        "dim-optimize",
        "vrf-cache",
        "self-verify",
    ]

    skillctl.write_manifest(output, names)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert [item["name"] for item in payload["skills"]] == names
    assert all(len(item["sha256"]) == 64 for item in payload["skills"])
    assert payload["skills"][-1]["version"] == "2.0.0"
    vrf_record = next(item for item in payload["skills"] if item["name"] == "vrf-cache")
    assert len(vrf_record["translations"]["zh"]["sha256"]) == 64


def test_pi_bundle_contains_every_effective_skill(tmp_path):
    output = tmp_path / "skills_bundle.md"
    names = [
        "common-constraints",
        "dag-analyze",
        "dim-optimize",
        "vrf-cache",
        "self-verify",
    ]

    skillctl.write_bundle(output, names)
    bundle = output.read_text(encoding="utf-8")

    headings = [line for line in bundle.splitlines() if line.startswith("## Skill:")]
    assert headings == [f"## Skill: {name}" for name in names]
    assert bundle.index("## Skill: common-constraints") < bundle.index(
        "## Skill: dim-optimize"
    )
    assert bundle.index("## Skill: dim-optimize") < bundle.index(
        "## Skill: self-verify"
    )


def test_list_hides_translation_snapshot_names(capsys):
    skillctl.list_versions()
    output = capsys.readouterr().out

    assert ".zh" not in output


def test_vrf_cache_v211_contains_staged_capacity_and_l3_contracts():
    skills = skillctl.discover_skills()
    vrf = skills["vrf-cache"]
    content = vrf.path.read_text(encoding="utf-8")

    assert vrf.version == "2.11.0"
    assert "One Level per Candidate" in content
    assert "Mandatory Preflight Report" in content
    assert "MEM_MVM_ACC_VRF" in content
    assert "seq2 total_bytes: candidate <= iteration input" in content
    assert "seq6 total_bytes: candidate <  iteration input" in content
    assert "instruction_gate=on" in content
    assert "allocation.cross_config_proven=true" in content
    assert "macro-dram-l3-kv-weight-stationary" in content
    assert "macro-dram-l3-q-weight-stationary" in content
    assert "retained_output_allocation" in content
    assert "macro-dram-l1-transient-scratch-bank13" in content
    assert "_SCRATCH + num_tiles*8" in content
    assert "macro-dram-l3-self-output-weight-stationary" in content
    assert "macro-dram-l3-ffn-intermediate-weight-stationary" in content
    assert "macro-dram-l3-ffn-output-weight-stationary" in content
    assert "macro-dram-l2-unit-vector-synthesis" in content
    assert "0x3c00" in content
    assert "L2X_CACHE[pos]" in content
    assert "next_macro_contract.json" in content
    assert "absent from `expected_dram_resources` as already eliminated" in content


def test_dag_analyze_v115_prefers_deterministic_contract_evidence():
    skills = skillctl.discover_skills()
    dag = skills["dag-analyze"]
    content = dag.path.read_text(encoding="utf-8")

    assert dag.version == "1.15.0"
    assert "multiseq_summary.md" in content
    assert "loop_invariants.json" in content
    assert "candidate_evidence.jsonl" in content
    assert "measured_removable_read_bytes" in content
    assert "implementation_ready=true" in content
    assert "deterministic K/V and Q contracts" in content
    assert "macro-dram-l1-transient-scratch-bank13" in content
    assert "SELF_OUTPUT contract" in content
    assert "FFN_INTERMEDIATE contract" in content
    assert "FFN_OUTPUT contract" in content
    assert "UNIT_VEC synthesis contract" in content
    assert "next_macro_contract.json" in content
    assert "do not reopen the full dag" in content.lower()
    assert "allocation_proof.json" in content
    assert "allocation_summary.md" in content
    assert "cross_config_proven" in content
    assert "validation_matrix_complete" in content
    assert "`already-eliminated`" in content
    assert "nonzero but incomplete pair" in content.lower()


def test_version_collision_is_rejected_and_same_version_can_rollback(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "jimu-dse" / "docs" / "skills" / "isa"
    archive_dir = tmp_path / "jimu-dse" / "docs" / "skills" / "versions"
    opencode_dir = tmp_path / ".opencode" / "skills"
    lock_path = tmp_path / "jimu-dse" / "docs" / "skills" / "skills.lock.json"
    source_dir.mkdir(parents=True)

    original = (
        "---\nname: demo\nversion: 1.0.0\n"
        "description: demo skill\n---\n\n# Original\n"
    )
    source = source_dir / "demo.md"
    source.write_text(original, encoding="utf-8")

    monkeypatch.setattr(skillctl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(skillctl, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(skillctl, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(skillctl, "OPENCODE_DIR", opencode_dir)
    monkeypatch.setattr(skillctl, "LOCK_PATH", lock_path)

    skillctl.sync_skills()
    source.write_text(original.replace("# Original", "# Unversioned edit"), encoding="utf-8")

    with pytest.raises(skillctl.SkillError, match="version collision"):
        skillctl.sync_skills()

    skillctl.rollback_skill("demo", "1.0.0")
    assert source.read_text(encoding="utf-8") == original
    skillctl.verify_skills()
