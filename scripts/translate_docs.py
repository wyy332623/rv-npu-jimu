from __future__ import annotations

import re
from pathlib import Path

import argostranslate.translate


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".pytest_cache"}
PROTECT_RE = re.compile(
    r"`[^`\n]+`|https?://[^\s)]+|(?<=\]\()[^)]+(?=\))|"
    r"\b[A-Za-z]:\\[^\s)]+|(?:\b[\w.-]+/)+[\w./-]+"
)
FENCE_RE = re.compile(r"(?ms)^(```.*?^```|~~~.*?^~~~)")


def get_translator():
    installed = argostranslate.translate.get_installed_languages()
    source = next(lang for lang in installed if lang.code == "en")
    target = next(lang for lang in installed if lang.code == "zh")
    return source.get_translation(target)


def should_translate(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_PARTS for part in rel.parts):
        return False
    if path.suffix == ".md" and path.name.endswith(".zh.md"):
        return False
    return path.suffix == ".md" or path.name == "LICENSE"


def output_path(path: Path) -> Path:
    if path.suffix == ".md":
        return path.with_name(f"{path.stem}.zh.md")
    return path.with_name(f"{path.name}.zh")


def protect(text: str):
    protected: list[str] = []

    def repl(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"XQZPROT{len(protected) - 1:05d}XQZ"

    return PROTECT_RE.sub(repl, text), protected


def restore(text: str, protected: list[str]) -> str:
    for idx, value in enumerate(protected):
        text = text.replace(f"XQZPROT{idx:05d}XQZ", value)
        text = text.replace(f"XQZ PROT{idx:05d}XQZ", value)
    return text


def translate_fragment(fragment: str, translator, cache: dict[str, str]) -> str:
    if not re.search(r"[A-Za-z]", fragment):
        return fragment
    if fragment.strip() == "":
        return fragment
    if fragment in cache:
        return cache[fragment]

    protected_fragment, protected = protect(fragment)
    try:
        translated = translator.translate(protected_fragment)
    except Exception:
        translated = protected_fragment
    translated = restore(translated, protected)
    cache[fragment] = translated
    return translated


def translate_line(line: str, translator, cache: dict[str, str]) -> str:
    if not line.strip():
        return line
    if re.match(r"^\s*[-*_]{3,}\s*$", line):
        return line
    if re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", line):
        return line

    if "|" in line and line.strip().startswith("|") and line.strip().endswith("|"):
        cells = line.split("|")
        for i in range(1, len(cells) - 1):
            cells[i] = translate_fragment(cells[i], translator, cache)
        return "|".join(cells)

    match = re.match(
        r"^(\s{0,3}(?:(?:#{1,6}\s+)|(?:[-*+]\s+(?:\[[ xX]\]\s+)?)|(?:\d+\.\s+)|(?:>\s+))*)(.*)$",
        line,
    )
    if match and match.group(1):
        return match.group(1) + translate_fragment(match.group(2), translator, cache)
    return translate_fragment(line, translator, cache)


def translate_markdown(text: str, translator) -> str:
    cache: dict[str, str] = {}
    pieces: list[str] = []
    last = 0
    for match in FENCE_RE.finditer(text):
        pieces.append(translate_plain_markdown(text[last : match.start()], translator, cache))
        pieces.append(match.group(0))
        last = match.end()
    pieces.append(translate_plain_markdown(text[last:], translator, cache))
    return "".join(pieces)


def translate_plain_markdown(text: str, translator, cache: dict[str, str]) -> str:
    return "\n".join(translate_line(line, translator, cache) for line in text.split("\n"))


def translate_license(text: str, translator) -> str:
    cache: dict[str, str] = {}
    return "\n".join(translate_fragment(line, translator, cache) for line in text.split("\n"))


def main() -> None:
    translator = get_translator()
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and should_translate(path))
    for src in files:
        dst = output_path(src)
        text = src.read_text(encoding="utf-8")
        if src.name == "LICENSE":
            translated = translate_license(text, translator)
            preface = "本文件是 LICENSE 的中文参考译文；授权条款以英文原文为准。\n\n"
        else:
            translated = translate_markdown(text, translator)
            preface = "> 本文件由自动翻译生成，仅供参考；以英文原文为准。\n\n"
        dst.write_text(preface + translated, encoding="utf-8", newline="\n")
        print(src.relative_to(ROOT), "->", dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
