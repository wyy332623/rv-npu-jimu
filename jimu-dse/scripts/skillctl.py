#!/usr/bin/env python3
"""Version, archive, verify, and publish JIMU optimization skills.

The active source of truth is jimu-dse/docs/skills/isa/*.md. English skill
files and their optional *.zh.md translations are archived by semantic version
and published under .opencode/skills/<name>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "jimu-dse" / "docs" / "skills" / "isa"
ARCHIVE_DIR = REPO_ROOT / "jimu-dse" / "docs" / "skills" / "versions"
OPENCODE_DIR = REPO_ROOT / ".opencode" / "skills"
LOCK_PATH = REPO_ROOT / "jimu-dse" / "docs" / "skills" / "skills.lock.json"
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")


class SkillError(RuntimeError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    version: str
    description: str
    path: Path
    sha256: str

    def metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "sha256": self.sha256,
            "source": self.path.relative_to(REPO_ROOT).as_posix(),
            "opencode_path": (
                OPENCODE_DIR / self.name / "SKILL.md"
            ).relative_to(REPO_ROOT).as_posix(),
            "archive": (
                ARCHIVE_DIR / self.name / f"{self.version}.md"
            ).relative_to(REPO_ROOT).as_posix(),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"{path}: missing YAML frontmatter")

    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    else:
        raise SkillError(f"{path}: unterminated YAML frontmatter")
    return metadata


def load_skill(path: Path, *, enforce_filename: bool = True) -> Skill:
    metadata = parse_frontmatter(path)
    missing = [key for key in ("name", "version", "description") if not metadata.get(key)]
    if missing:
        raise SkillError(f"{path}: missing frontmatter fields: {', '.join(missing)}")
    if enforce_filename and metadata["name"] != path.stem.replace("_", "-"):
        raise SkillError(
            f"{path}: skill name {metadata['name']!r} must match filename "
            f"{path.stem!r} (underscores may represent hyphens)"
        )
    if not SEMVER_RE.fullmatch(metadata["version"]):
        raise SkillError(f"{path}: invalid semantic version {metadata['version']!r}")
    return Skill(
        name=metadata["name"],
        version=metadata["version"],
        description=metadata["description"],
        path=path,
        sha256=sha256_file(path),
    )


def discover_skills() -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for path in sorted(SOURCE_DIR.glob("*.md")):
        if path.name.endswith(".zh.md"):
            continue
        skill = load_skill(path)
        if skill.name in skills:
            raise SkillError(f"duplicate skill name: {skill.name}")
        skills[skill.name] = skill
    if not skills:
        raise SkillError(f"no skills found in {SOURCE_DIR}")
    return skills


def discover_translations(skills: dict[str, Skill]) -> dict[str, Path]:
    translations: dict[str, Path] = {}
    for path in sorted(SOURCE_DIR.glob("*.zh.md")):
        metadata = parse_frontmatter(path)
        name = metadata.get("name", "")
        expected_name = path.name.removesuffix(".zh.md").replace("_", "-")
        if not name or name != expected_name:
            raise SkillError(
                f"{path}: translated skill name {name!r} must match {expected_name!r}"
            )
        if name not in skills:
            raise SkillError(f"{path}: translation has no canonical English skill")
        translations[name] = path
    return translations


def select_skills(names: Iterable[str]) -> list[Skill]:
    available = discover_skills()
    selected: list[Skill] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        if name not in available:
            raise SkillError(
                f"unknown skill {name!r}; available: {', '.join(sorted(available))}"
            )
        selected.append(available[name])
        seen.add(name)
    if not selected:
        raise SkillError("at least one skill name is required")
    return selected


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def archive_skill(skill: Skill) -> Path:
    archive = ARCHIVE_DIR / skill.name / f"{skill.version}.md"
    if archive.exists():
        archived_sha = sha256_file(archive)
        if archived_sha != skill.sha256:
            raise SkillError(
                f"version collision for {skill.name} {skill.version}: "
                f"archive={archived_sha}, source={skill.sha256}; bump version first"
            )
    else:
        atomic_copy(skill.path, archive)
    return archive


def translation_metadata(skill: Skill, path: Path) -> dict[str, str]:
    return {
        "source": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "opencode_path": (
            OPENCODE_DIR / skill.name / "SKILL.zh.md"
        ).relative_to(REPO_ROOT).as_posix(),
        "archive": (
            ARCHIVE_DIR / skill.name / f"{skill.version}.zh.md"
        ).relative_to(REPO_ROOT).as_posix(),
    }


def archive_translation(skill: Skill, path: Path) -> Path:
    archive = ARCHIVE_DIR / skill.name / f"{skill.version}.zh.md"
    source_sha = sha256_file(path)
    if archive.exists():
        archived_sha = sha256_file(archive)
        if archived_sha != source_sha:
            raise SkillError(
                f"version collision for {skill.name} {skill.version} zh translation: "
                f"archive={archived_sha}, source={source_sha}; bump version first"
            )
    else:
        atomic_copy(path, archive)
    return archive


def expected_lock(skills: dict[str, Skill]) -> dict[str, object]:
    translations = discover_translations(skills)
    records: dict[str, dict[str, object]] = {}
    for name, skill in sorted(skills.items()):
        record: dict[str, object] = skill.metadata()
        if name in translations:
            record["translations"] = {
                "zh": translation_metadata(skill, translations[name])
            }
        records[name] = record
    return {
        "schema_version": 1,
        "source_of_truth": "jimu-dse/docs/skills/isa/*.md",
        "skills": records,
    }


def sync_skills() -> dict[str, Skill]:
    skills = discover_skills()
    translations = discover_translations(skills)
    for skill in skills.values():
        archive_skill(skill)
        destination = OPENCODE_DIR / skill.name / "SKILL.md"
        if not destination.exists() or sha256_file(destination) != skill.sha256:
            atomic_copy(skill.path, destination)
        translation = translations.get(skill.name)
        if translation is not None:
            archive_translation(skill, translation)
            translated_destination = OPENCODE_DIR / skill.name / "SKILL.zh.md"
            if (
                not translated_destination.exists()
                or sha256_file(translated_destination) != sha256_file(translation)
            ):
                atomic_copy(translation, translated_destination)
    lock_text = json.dumps(expected_lock(skills), indent=2, sort_keys=True) + "\n"
    if not LOCK_PATH.exists() or LOCK_PATH.read_text(encoding="utf-8") != lock_text:
        atomic_write_text(LOCK_PATH, lock_text)
    return skills


def verify_skills() -> dict[str, Skill]:
    skills = discover_skills()
    translations = discover_translations(skills)
    errors: list[str] = []
    for skill in skills.values():
        archive = ARCHIVE_DIR / skill.name / f"{skill.version}.md"
        published = OPENCODE_DIR / skill.name / "SKILL.md"
        if not archive.exists():
            errors.append(f"missing archive: {archive.relative_to(REPO_ROOT)}")
        elif sha256_file(archive) != skill.sha256:
            errors.append(f"archive differs from source: {archive.relative_to(REPO_ROOT)}")
        if not published.exists():
            errors.append(f"missing OpenCode skill: {published.relative_to(REPO_ROOT)}")
        elif sha256_file(published) != skill.sha256:
            errors.append(f"OpenCode skill is stale: {published.relative_to(REPO_ROOT)}")
        translation = translations.get(skill.name)
        if translation is not None:
            translated_archive = ARCHIVE_DIR / skill.name / f"{skill.version}.zh.md"
            translated_published = OPENCODE_DIR / skill.name / "SKILL.zh.md"
            translated_sha = sha256_file(translation)
            if not translated_archive.exists():
                errors.append(
                    f"missing translation archive: {translated_archive.relative_to(REPO_ROOT)}"
                )
            elif sha256_file(translated_archive) != translated_sha:
                errors.append(
                    "translation archive differs from source: "
                    f"{translated_archive.relative_to(REPO_ROOT)}"
                )
            if not translated_published.exists():
                errors.append(
                    f"missing OpenCode translation: {translated_published.relative_to(REPO_ROOT)}"
                )
            elif sha256_file(translated_published) != translated_sha:
                errors.append(
                    f"OpenCode translation is stale: {translated_published.relative_to(REPO_ROOT)}"
                )

    expected = expected_lock(skills)
    if not LOCK_PATH.exists():
        errors.append(f"missing lock file: {LOCK_PATH.relative_to(REPO_ROOT)}")
    else:
        try:
            actual = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid lock file: {exc}")
        else:
            if actual != expected:
                errors.append(f"lock file is stale: {LOCK_PATH.relative_to(REPO_ROOT)}")
    if errors:
        raise SkillError("\n".join(errors))
    return skills


def write_manifest(output: Path, names: list[str]) -> None:
    selected = select_skills(names)
    translations = discover_translations(discover_skills())
    records = []
    for skill in selected:
        record: dict[str, object] = skill.metadata()
        if skill.name in translations:
            record["translations"] = {
                "zh": translation_metadata(skill, translations[skill.name])
            }
        records.append(record)
    payload = {
        "schema_version": 1,
        "skills": records,
    }
    atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_bundle(output: Path, names: list[str]) -> None:
    selected = select_skills(names)
    version = "+".join(f"{skill.name}@{skill.version}" for skill in selected)
    descriptions = "; ".join(skill.description for skill in selected)
    chunks = [
        "---",
        "name: jimu-run-skill-bundle",
        f'version: "{version}"',
        f'description: "{descriptions}"',
        "---",
        "",
        "# JIMU Run Skill Bundle",
        "",
        "Apply every skill below. Earlier skills define mandatory constraints for later ones.",
        "",
    ]
    for skill in selected:
        chunks.extend(
            [
                f"## Skill: {skill.name}",
                "",
                f"- Version: `{skill.version}`",
                f"- SHA256: `{skill.sha256}`",
                f"- Source: `{skill.path.relative_to(REPO_ROOT).as_posix()}`",
                "",
                skill.path.read_text(encoding="utf-8").rstrip(),
                "",
            ]
        )
    atomic_write_text(output, "\n".join(chunks).rstrip() + "\n")


def rollback_skill(name: str, version: str) -> None:
    skills = discover_skills()
    translations = discover_translations(skills)
    if name not in skills:
        raise SkillError(f"unknown skill {name!r}")
    current = skills[name]
    target = ARCHIVE_DIR / name / f"{version}.md"
    current_translation = translations.get(name)
    target_translation = ARCHIVE_DIR / name / f"{version}.zh.md"
    if current.version != version:
        archive_skill(current)
        if current_translation is not None:
            archive_translation(current, current_translation)
    elif not target.exists():
        archive_skill(current)
    if not target.exists():
        versions = sorted(path.stem for path in (ARCHIVE_DIR / name).glob("*.md"))
        raise SkillError(
            f"version {version!r} not found for {name}; available: {', '.join(versions)}"
        )
    if current_translation is not None and not target_translation.exists():
        raise SkillError(
            f"translation archive not found for {name} {version}: {target_translation}"
        )
    restored = load_skill(target, enforce_filename=False)
    if restored.name != name or restored.version != version:
        raise SkillError(f"invalid archive metadata in {target}")
    atomic_copy(target, current.path)
    if current_translation is not None:
        atomic_copy(target_translation, current_translation)
    sync_skills()


def list_versions() -> None:
    skills = discover_skills()
    for name, skill in sorted(skills.items()):
        versions = sorted(
            path.stem
            for path in (ARCHIVE_DIR / name).glob("*.md")
            if not path.name.endswith(".zh.md")
        )
        suffix = ", ".join(versions) if versions else "(not snapshotted)"
        print(f"{name}: active={skill.version} sha256={skill.sha256} archived=[{suffix}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sync", help="archive active skills and publish to OpenCode")
    subparsers.add_parser("verify", help="verify source, archive, lock, and OpenCode copies")
    subparsers.add_parser("list", help="list active and archived skill versions")

    rollback = subparsers.add_parser("rollback", help="restore an archived version")
    rollback.add_argument("name")
    rollback.add_argument("version")

    manifest = subparsers.add_parser("manifest", help="write metadata for selected skills")
    manifest.add_argument("--output", required=True, type=Path)
    manifest.add_argument("skills", nargs="+")

    bundle = subparsers.add_parser("bundle", help="combine selected skills for one agent run")
    bundle.add_argument("--output", required=True, type=Path)
    bundle.add_argument("skills", nargs="+")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "sync":
            skills = sync_skills()
            print(f"synced {len(skills)} skills")
        elif args.command == "verify":
            skills = verify_skills()
            print(f"verified {len(skills)} skills")
        elif args.command == "list":
            list_versions()
        elif args.command == "rollback":
            rollback_skill(args.name, args.version)
            print(f"rolled back {args.name} to {args.version}")
        elif args.command == "manifest":
            write_manifest(args.output, args.skills)
        elif args.command == "bundle":
            write_bundle(args.output, args.skills)
        else:
            raise AssertionError(args.command)
    except (OSError, SkillError) as exc:
        print(f"skillctl: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
