#!/usr/bin/env bash
# Remove reproducible local outputs without touching source, venv, or run history.
set -euo pipefail

MODE=dry-run
RUN_DIR=""

usage() {
    cat <<'EOF'
Usage: clean_workspace.sh [--apply] [--run-dir PATH]

Without --apply the script only prints the verified cleanup plan.
The default cleanup removes build outputs, Python/test caches, local edit
backups, and generated ELF files. It preserves virtual environments and every
closed-loop run unless one exact run directory is passed with --run-dir.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)
            MODE=apply
            shift
            ;;
        --run-dir)
            [[ $# -ge 2 ]] || {
                echo "--run-dir requires a path" >&2
                exit 2
            }
            RUN_DIR=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
REPO_ROOT=$(realpath -m -- "$REPO_ROOT")
declare -a TARGETS=()

add_target() {
    local candidate=$1
    local lexical resolved
    # Keep the lexical path for deletion so a symlink itself is removed rather
    # than the file it points to. Resolve separately only for the boundary check.
    lexical=$(realpath -ms -- "$candidate")
    case "$lexical" in
        "${REPO_ROOT}"/*) ;;
        *)
            echo "refusing cleanup target outside repository: ${candidate}" >&2
            exit 1
            ;;
    esac
    [[ "$lexical" != "$REPO_ROOT" ]] || {
        echo "refusing to clean repository root" >&2
        exit 1
    }
    if [[ -e "$lexical" || -L "$lexical" ]]; then
        resolved=$(realpath -m -- "$lexical")
        case "$resolved" in
            "${REPO_ROOT}"/*) ;;
            *)
                echo "refusing symlink target outside repository: ${candidate}" >&2
                exit 1
                ;;
        esac
        TARGETS+=("$lexical")
    fi
}

for path in \
    "$REPO_ROOT/_build" \
    "$REPO_ROOT/_out" \
    "$REPO_ROOT/.pytest_cache" \
    "$REPO_ROOT/libnpukernels.so"; do
    add_target "$path"
done

while IFS= read -r -d '' path; do
    add_target "$path"
done < <(
    find "$REPO_ROOT/firmware" -mindepth 1 -maxdepth 1 \
        -type d -name 'build*' -print0 2>/dev/null
)

while IFS= read -r -d '' path; do
    add_target "$path"
done < <(
    find "$REPO_ROOT/adderboard/firmware" -mindepth 1 -maxdepth 1 \
        -type d -name 'build_*' -print0 2>/dev/null
)

for search_root in .opencode emulator iss tests jimu-dse scripts kernels firmware adderboard; do
    [[ -d "$REPO_ROOT/$search_root" ]] || continue
    while IFS= read -r -d '' path; do
        add_target "$path"
    done < <(
        find "$REPO_ROOT/$search_root" \
            \( -type d -name __pycache__ -o \
               -type f \( -name '*.pyc' -o -name '*.pyo' -o \
                            -name '*.bak' -o -name '*.new' -o \
                            -name '*.orig' -o -name '*.saved' -o \
                            -name '*.before_manual_validation' \) \) \
            -print0 2>/dev/null
    )
done

while IFS= read -r -d '' path; do
    add_target "$path"
done < <(
    find "$REPO_ROOT/firmware" -mindepth 1 -maxdepth 1 -type f \
        \( -name '*.elf' -o -name '*.elf.*' \) -print0 2>/dev/null
)

if [[ -n "$RUN_DIR" ]]; then
    RUN_DIR=$(realpath -m -- "$RUN_DIR")
    case "$RUN_DIR" in
        "$REPO_ROOT/jimu-dse/results/"run-*) ;;
        *)
            echo "--run-dir must select one run-* directory below jimu-dse/results" >&2
            exit 1
            ;;
    esac
    add_target "$RUN_DIR"
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "workspace already clean"
    exit 0
fi

mapfile -t TARGETS < <(printf '%s\n' "${TARGETS[@]}" | sort -u)
echo "mode=${MODE} targets=${#TARGETS[@]}"
printf '  %s\n' "${TARGETS[@]}"

if [[ "$MODE" != apply ]]; then
    echo "dry-run only; pass --apply to remove these targets"
    exit 0
fi

# Remove deeper paths first so parent build directories do not hide failures.
mapfile -t TARGETS < <(printf '%s\n' "${TARGETS[@]}" | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2-)
for target in "${TARGETS[@]}"; do
    if [[ -d "$target" && ! -L "$target" ]]; then
        rm -rf -- "$target"
    else
        rm -f -- "$target"
    fi
done
echo "cleanup complete"
