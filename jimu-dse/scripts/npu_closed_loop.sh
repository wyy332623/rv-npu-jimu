#!/usr/bin/env bash
# NPU Closed-Loop Firmware-Hardware Co-Optimization
#
#   PROBE → ANALYZE → AGENT → VALIDATE → DEPLOY → LOOP
#
# Supports multiple optimization goals via --goal:
#   dram-optimization   (G1): Reduce DRAM bytes at dim=2 via VRF cache
#   compute-optimization (G2): Refactor firmware for dim=4 single-tile
#   combined            (G3): Both G1 + G2 at dim=4
#
# Usage:
#   ./jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization
#   ./jimu-dse/scripts/npu_closed_loop.sh --goal compute-optimization --agent opencode
#   ./jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode \
#       --start-from jimu-dse/results/run-<timestamp>
#   ./jimu-dse/scripts/npu_closed_loop.sh --goal dram-optimization --validation-dim dim2
set -uo pipefail

JIMU_MAX_ITER="${JIMU_MAX_ITER:-5}"
JIMU_THRESHOLD="${JIMU_THRESHOLD:-0.15}"
JIMU_INSTR_GATE="${JIMU_INSTR_GATE:-on}"
JIMU_INSTR_REGRESSION_LIMIT="${JIMU_INSTR_REGRESSION_LIMIT:-0.10}"
JIMU_DAG_EVIDENCE_GATE="${JIMU_DAG_EVIDENCE_GATE:-on}"
# opencode agents need more time for multi-step file analysis and editing
JIMU_AGENT_TIMEOUT="${JIMU_AGENT_TIMEOUT:-1800}"
JIMU_AGENT_RETRIES="${JIMU_AGENT_RETRIES:-2}"
RUN_TAG="run-$(date +%Y%m%d-%H%M%S)-$$"
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
CC="${CC:-riscv64-unknown-elf-gcc}"

RESUME_DIR=""
START_FROM=""
AGENT="pi"
OPENCODE_MODEL="${OPENCODE_MODEL:-opencode/big-pickle}"
GOAL="dram-optimization"
VALIDATION_DIM_REQUEST="${JIMU_VALIDATION_DIM:-all}"
PREPARE_ONLY=0

usage() {
    cat <<'EOF'
Usage: npu_closed_loop.sh [options]

Options:
  --goal NAME                Optimization goal.
  --agent pi|opencode        Agent backend.
  --model NAME               OpenCode model.
  --workload NAME            Workload manifest name.
  --start-from PATH          Optimization starting source. PATH may be:
                             - a firmware .c file (including candidate_N.c)
                             - a run directory containing candidate_best.c
                             - "baseline" for the canonical unoptimized source
  --resume DIR               Compatibility alias for:
                             --start-from DIR/candidate_best.c
  --prepare-only             Generate prompt/manifests without probing,
                             invoking an agent, or modifying firmware.
  --instruction-gate STATE   Enable or disable the G1 seq6 instruction gate:
                             on (default) or off.
  --instruction-regression-limit RATIO
                             Maximum G1 seq6 instruction regression ratio.
                             The value "off" also disables the gate.
  --dag-evidence-gate STATE  Require the G1 Agent candidate declaration and
                             measured/structural DAG consistency: on (default)
                             or off.
  --validation-dim SCOPE     Correctness gate scope:
                             dim2  - validate all dim2 BERT cases
                             dim4  - validate all dim4 BERT cases
                             all   - validate dim2 and dim4 (default)
                             goal  - validate the DIM selected by the goal
  -h, --help                 Show this help.

Environment:
  JIMU_VALIDATION_DIM        Default for --validation-dim.
  JIMU_INSTR_GATE            on (default) or off.
  JIMU_INSTR_REGRESSION_LIMIT
                             Maximum G1 seq6 instruction regression ratio
                             (default: 0.10); "off" disables the gate.
  JIMU_DAG_EVIDENCE_GATE     on (default) or off.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            [[ $# -ge 2 ]] || {
                echo "--resume requires a run directory" >&2
                exit 1
            }
            RESUME_DIR="$2"
            shift 2
            ;;
        --start-from)
            [[ $# -ge 2 ]] || {
                echo "--start-from requires a source file, run directory, or baseline" >&2
                exit 1
            }
            START_FROM="$2"
            shift 2
            ;;
        --prepare-only) PREPARE_ONLY=1; shift ;;
        --instruction-gate)
            [[ $# -ge 2 ]] || {
                echo "--instruction-gate requires on or off" >&2
                exit 1
            }
            JIMU_INSTR_GATE="$2"
            shift 2
            ;;
        --instruction-regression-limit)
            [[ $# -ge 2 ]] || {
                echo "--instruction-regression-limit requires a ratio or off" >&2
                exit 1
            }
            JIMU_INSTR_REGRESSION_LIMIT="$2"
            shift 2
            ;;
        --dag-evidence-gate)
            [[ $# -ge 2 ]] || {
                echo "--dag-evidence-gate requires on or off" >&2
                exit 1
            }
            JIMU_DAG_EVIDENCE_GATE="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
            if [[ "$AGENT" != "pi" && "$AGENT" != "opencode" ]]; then
                echo "Unknown agent: $AGENT (use pi or opencode)" >&2
                exit 1
            fi
            shift 2 ;;
        --model) OPENCODE_MODEL="$2"; shift 2 ;;
        --workload) WORKLOAD="$2"; shift 2 ;;
        --validation-dim)
            [[ $# -ge 2 ]] || {
                echo "--validation-dim requires dim2, dim4, all, or goal" >&2
                exit 1
            }
            VALIDATION_DIM_REQUEST="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --goal)
            GOAL="$2"
            GOAL_DIR="${REPO_ROOT}/jimu-dse/goals/${GOAL}"
            if [[ ! -f "${GOAL_DIR}/goal.sh" ]]; then
                echo "Unknown goal: $GOAL" >&2
                echo "Available goals:" >&2
                for g in "${REPO_ROOT}"/jimu-dse/goals/*/goal.sh; do
                    gname=$(basename "$(dirname "$g")")
                    desc=$(grep -m1 '^# ' "$g" 2>/dev/null | sed 's/^# //')
                    echo "  $gname${desc:+ ($desc)}" >&2
                done
                exit 1
            fi
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ! "${JIMU_AGENT_RETRIES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "JIMU_AGENT_RETRIES must be a positive integer" >&2
    exit 1
fi

case "${JIMU_INSTR_REGRESSION_LIMIT,,}" in
    off|none|disabled)
        JIMU_INSTR_GATE="off"
        JIMU_INSTR_REGRESSION_LIMIT="0.10"
        ;;
esac

case "${JIMU_INSTR_GATE,,}" in
    on|true|1|enabled)
        JIMU_INSTR_GATE="on"
        INSTR_GATE_ENABLED=1
        python3 -c 'import sys; assert float(sys.argv[1]) >= 0' \
            "${JIMU_INSTR_REGRESSION_LIMIT}" 2>/dev/null || {
            echo "Instruction regression limit must be a non-negative ratio" >&2
            exit 1
        }
        ;;
    off|false|0|disabled)
        JIMU_INSTR_GATE="off"
        INSTR_GATE_ENABLED=0
        # The numeric value is ignored while disabled but retained for CLI shape.
        JIMU_INSTR_REGRESSION_LIMIT="0.10"
        ;;
    *)
        echo "Unknown instruction gate state: ${JIMU_INSTR_GATE} (use on or off)" >&2
        exit 1
        ;;
esac

case "${JIMU_DAG_EVIDENCE_GATE,,}" in
    on|true|1|enabled)
        JIMU_DAG_EVIDENCE_GATE="on"
        DAG_EVIDENCE_GATE_ENABLED=1
        ;;
    off|false|0|disabled)
        JIMU_DAG_EVIDENCE_GATE="off"
        DAG_EVIDENCE_GATE_ENABLED=0
        ;;
    *)
        echo "Unknown DAG evidence gate state: ${JIMU_DAG_EVIDENCE_GATE} (use on or off)" >&2
        exit 1
        ;;
esac

log() { echo "[$(date -Iseconds)] $*" >&2; }
die() { log "[FATAL] $*"; exit 1; }

cd "${REPO_ROOT}"

# ── Load goal configuration ──────────────────────────────────────────
WORKLOAD="${WORKLOAD:-bert}"
GOAL_DIR="${REPO_ROOT}/jimu-dse/goals/${GOAL}"
source "${GOAL_DIR}/goal.sh"

WORKLOAD_MANIFEST="${REPO_ROOT}/jimu-dse/workloads/${WORKLOAD}.sh"
if [[ -f "${WORKLOAD_MANIFEST}" ]]; then
    source "${WORKLOAD_MANIFEST}"
else
    # Fallback to bert if manifest not found
    WORKLOAD_NAME="bert"
    MAKE_TARGET="bert"
    TARGET_FILE="firmware/bert/bert_layer.c"
    TEST_VERIFY_CMD='python3 -m pytest tests/integration/test_bert_e2e.py -k "dim${DIM} and h${HIDDEN}" --no-header -q -rs'
    TEST_CONVERGE_CMD="${TEST_VERIFY_CMD}"
    TEST_GATE_CMD='python3 -m pytest tests/integration/test_bert_e2e.py --no-header -q -rs'
    EXPECTED_GATE_TESTS_ALL=6
    EXPECTED_GATE_TESTS_DIM2=2
    EXPECTED_GATE_TESTS_DIM4=4
fi

case "${VALIDATION_DIM_REQUEST}" in
    all|both)
        VALIDATION_DIM_REQUEST="all"
        VALIDATION_SCOPE="all"
        ;;
    goal)
        VALIDATION_SCOPE="dim${DIM}"
        ;;
    2|dim2)
        VALIDATION_DIM_REQUEST="dim2"
        VALIDATION_SCOPE="dim2"
        ;;
    4|dim4)
        VALIDATION_DIM_REQUEST="dim4"
        VALIDATION_SCOPE="dim4"
        ;;
    *)
        die "Unknown validation dimension: ${VALIDATION_DIM_REQUEST} (use dim2, dim4, all, or goal)"
        ;;
esac

if [[ "${WORKLOAD_NAME}" == "bert" ]]; then
    if [[ "${VALIDATION_SCOPE}" != "all" ]]; then
        TEST_GATE_CMD+=" -k \"${VALIDATION_SCOPE}\""
    fi
elif [[ "${VALIDATION_SCOPE}" != "all" ]] && \
     [[ "${VALIDATION_SCOPE}" != "dim${DIM}" ]]; then
    die "${WORKLOAD_NAME} goal uses dim${DIM}; cannot validate ${VALIDATION_SCOPE}"
fi

case "${VALIDATION_SCOPE}" in
    all)
        EXPECTED_GATE_TESTS="${EXPECTED_GATE_TESTS_ALL:-0}"
        VALIDATION_COVERAGE="full"
        ;;
    dim2)
        EXPECTED_GATE_TESTS="${EXPECTED_GATE_TESTS_DIM2:-0}"
        VALIDATION_COVERAGE="partial"
        ;;
    dim4)
        EXPECTED_GATE_TESTS="${EXPECTED_GATE_TESTS_DIM4:-0}"
        VALIDATION_COVERAGE="partial"
        ;;
    *)
        EXPECTED_GATE_TESTS=0
        VALIDATION_COVERAGE="workload-defined"
        ;;
esac
# Agent self-verification uses the same unfiltered command as the independent gate.
TEST_VERIFY_CMD="${TEST_GATE_CMD}"

# BASELINE_FILE is the canonical, known-correct, unoptimized firmware source.
# The optimization starting point is selected independently so continuing from
# an earlier candidate never mutates or redefines the canonical baseline.
CANONICAL_BASELINE_FILE="${BASELINE_FILE}"
[[ -s "${CANONICAL_BASELINE_FILE}" ]] || \
    die "No canonical baseline file at ${CANONICAL_BASELINE_FILE}"

if [[ -n "${RESUME_DIR}" && -n "${START_FROM}" ]]; then
    die "--resume and --start-from are mutually exclusive"
fi

START_SOURCE_KIND="canonical-baseline"
OPTIMIZATION_START_ORIGIN="${CANONICAL_BASELINE_FILE}"
OPTIMIZATION_START_SOURCE="${CANONICAL_BASELINE_FILE}"

if [[ -n "${RESUME_DIR}" ]]; then
    [[ -d "${RESUME_DIR}" ]] || die "Resume directory not found: ${RESUME_DIR}"
    OPTIMIZATION_START_SOURCE="${RESUME_DIR}/candidate_best.c"
    OPTIMIZATION_START_ORIGIN="${RESUME_DIR}"
    START_SOURCE_KIND="resume-run"
elif [[ -n "${START_FROM}" ]]; then
    case "${START_FROM,,}" in
        baseline|canonical|unoptimized)
            OPTIMIZATION_START_SOURCE="${CANONICAL_BASELINE_FILE}"
            OPTIMIZATION_START_ORIGIN="${CANONICAL_BASELINE_FILE}"
            START_SOURCE_KIND="canonical-baseline"
            ;;
        *)
            if [[ -d "${START_FROM}" ]]; then
                OPTIMIZATION_START_SOURCE="${START_FROM}/candidate_best.c"
                START_SOURCE_KIND="run-directory"
            else
                OPTIMIZATION_START_SOURCE="${START_FROM}"
                START_SOURCE_KIND="source-file"
            fi
            OPTIMIZATION_START_ORIGIN="${START_FROM}"
            ;;
    esac
fi

[[ -s "${OPTIMIZATION_START_SOURCE}" ]] || \
    die "Optimization starting source not found or empty: ${OPTIMIZATION_START_SOURCE}"
OPTIMIZATION_START_SOURCE=$(realpath "${OPTIMIZATION_START_SOURCE}")
if [[ "${OPTIMIZATION_START_SOURCE}" == "${REPO_ROOT}/"* ]]; then
    OPTIMIZATION_START_FILE="${OPTIMIZATION_START_SOURCE#${REPO_ROOT}/}"
else
    OPTIMIZATION_START_FILE="${OPTIMIZATION_START_SOURCE}"
fi

# DIM, HIDDEN, NUM_HEAD, SEQ_LENS, SKILLS, and PRIMARY_METRIC are set
# from the goal config.
read -ra SEQ_ARRAY <<< "${SEQ_LENS}"
SL2="${SEQ_ARRAY[0]}"
SL6="${SEQ_ARRAY[1]}"

SKILLCTL="${REPO_ROOT}/jimu-dse/scripts/skillctl.py"
[[ -f "${SKILLCTL}" ]] || die "Skill manager not found: ${SKILLCTL}"
python3 "${SKILLCTL}" sync > /dev/null || \
    die "Skill synchronization failed; fix skill metadata/version conflicts first"

read -ra GOAL_SKILL_ARRAY <<< "${SKILLS}"
EFFECTIVE_SKILL_NAMES=("common-constraints" "dag-analyze")
for sk in "${GOAL_SKILL_ARRAY[@]}"; do
    already_selected=0
    for selected in "${EFFECTIVE_SKILL_NAMES[@]}"; do
        [[ "${selected}" == "${sk}" ]] && already_selected=1 && break
    done
    [[ ${already_selected} -eq 0 ]] && EFFECTIVE_SKILL_NAMES+=("${sk}")
done
if [[ ! " ${EFFECTIVE_SKILL_NAMES[*]} " =~ [[:space:]]self-verify[[:space:]] ]]; then
    EFFECTIVE_SKILL_NAMES+=("self-verify")
fi

RESULTS="${REPO_ROOT}/jimu-dse/results/${RUN_TAG}"
mkdir -p "${RESULTS}"
OPTIMIZATION_BASELINE="${RESULTS}/optimization_baseline.c"
cp "${OPTIMIZATION_START_SOURCE}" "${OPTIMIZATION_BASELINE}"
OPTIMIZATION_START_SHA256=$(sha256sum "${OPTIMIZATION_BASELINE}" | awk '{print $1}')
SKILL_MANIFEST="${RESULTS}/skills_manifest.json"
SKILL_BUNDLE="${RESULTS}/skills_bundle.md"
python3 "${SKILLCTL}" manifest --output "${SKILL_MANIFEST}" \
    "${EFFECTIVE_SKILL_NAMES[@]}" || die "Failed to record skill manifest"
python3 "${SKILLCTL}" bundle --output "${SKILL_BUNDLE}" \
    "${EFFECTIVE_SKILL_NAMES[@]}" || die "Failed to build combined skill bundle"

log "===== NPU Closed Loop ====="
log "Goal:       ${GOAL} (dim=${DIM}, hidden=${HIDDEN}, heads=${NUM_HEAD})"
log "Agent:      ${AGENT}"
log "Run tag:    ${RUN_TAG}"
log "Reference:  ${CANONICAL_BASELINE_FILE} (canonical unoptimized)"
log "Start from: ${OPTIMIZATION_START_FILE} (${START_SOURCE_KIND})"
log "Skills:     ${EFFECTIVE_SKILL_NAMES[*]}"
log "Metric:     ${PRIMARY_METRIC}"
log "Validation: ${VALIDATION_SCOPE}"
log "Coverage:   ${VALIDATION_COVERAGE}"
log "Expected:   ${EXPECTED_GATE_TESTS:-0} passed, 0 skipped"
if [[ "${JIMU_INSTR_GATE}" == "on" ]]; then
    log "Instr gate: on (max regression ${JIMU_INSTR_REGRESSION_LIMIT})"
else
    log "Instr gate: off"
fi
log "DAG gate:   ${JIMU_DAG_EVIDENCE_GATE}"
log "Agent retry: ${JIMU_AGENT_RETRIES} attempt(s), ${JIMU_AGENT_TIMEOUT}s each"
if [[ "${VALIDATION_COVERAGE}" == "partial" ]]; then
    log "[WARN] Partial validation is for debugging; it is not cross-DIM verification"
fi
[[ ${PREPARE_ONLY} -eq 1 ]] && log "Mode:       prepare-only"

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_STATUS_SHA256=$(git status --porcelain=v1 2>/dev/null | sha256sum | awk '{print $1}')
BASELINE_SHA256=$(sha256sum "${CANONICAL_BASELINE_FILE}" | awk '{print $1}')
python3 - "${RESULTS}/run_manifest.json" \
    "$(date -Iseconds)" "${GIT_COMMIT}" "${GIT_STATUS_SHA256}" \
    "${CANONICAL_BASELINE_FILE}" "${BASELINE_SHA256}" \
    "${OPTIMIZATION_START_FILE}" "${OPTIMIZATION_START_SHA256}" \
    "${START_SOURCE_KIND}" "${OPTIMIZATION_BASELINE}" \
    "${GOAL}" "${WORKLOAD}" \
    "${AGENT}" "${OPENCODE_MODEL}" "${JIMU_AGENT_TIMEOUT}" \
    "${JIMU_AGENT_RETRIES}" "${JIMU_MAX_ITER}" "${JIMU_THRESHOLD}" \
    "${JIMU_INSTR_GATE}" "${JIMU_INSTR_REGRESSION_LIMIT}" \
    "${JIMU_DAG_EVIDENCE_GATE}" \
    "${VALIDATION_DIM_REQUEST}" "${VALIDATION_SCOPE}" "${TEST_GATE_CMD}" \
    "${SKILL_MANIFEST}" "${SKILL_BUNDLE}" "${PREPARE_ONLY}" \
    "${EXPECTED_GATE_TESTS:-0}" "${VALIDATION_COVERAGE}" <<'PY'
import json
import sys

(
    path,
    created_at,
    commit,
    status_sha256,
    baseline_file,
    baseline_sha256,
    optimization_start_file,
    optimization_start_sha256,
    optimization_start_kind,
    optimization_baseline_snapshot,
    goal,
    workload,
    agent,
    model,
    agent_timeout_seconds,
    agent_retries,
    max_iter,
    threshold,
    instruction_gate,
    instruction_regression_limit,
    dag_evidence_gate,
    validation_dim_request,
    validation_scope,
    gate_command,
    skill_manifest_path,
    skill_bundle_path,
    prepare_only,
    expected_gate_tests,
    validation_coverage,
) = sys.argv[1:]

with open(skill_manifest_path, encoding="utf-8") as skill_file:
    skill_manifest = json.load(skill_file)

with open(path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "created_at": created_at,
            "git_commit": commit,
            "git_status_sha256": status_sha256,
            # baseline_file is retained for compatibility and always names the
            # canonical unoptimized reference, never a promoted candidate.
            "baseline_file": baseline_file,
            "baseline_sha256": baseline_sha256,
            "canonical_baseline_file": baseline_file,
            "canonical_baseline_sha256": baseline_sha256,
            "optimization_start_file": optimization_start_file,
            "optimization_start_sha256": optimization_start_sha256,
            "optimization_start_kind": optimization_start_kind,
            "optimization_baseline_snapshot": optimization_baseline_snapshot,
            "goal": goal,
            "workload": workload,
            "agent": agent,
            "model": model,
            "agent_timeout_seconds": int(agent_timeout_seconds),
            "agent_retries": int(agent_retries),
            "max_iterations": int(max_iter),
            "convergence_threshold_ratio": float(threshold),
            "instruction_gate_enabled": instruction_gate == "on",
            "instruction_regression_limit": (
                float(instruction_regression_limit)
                if instruction_gate == "on"
                else None
            ),
            "dag_evidence_gate_enabled": dag_evidence_gate == "on",
            "validation_dim_request": validation_dim_request,
            "validation_scope": validation_scope,
            "validation_coverage": validation_coverage,
            "correctness_gate_command": gate_command,
            "expected_gate_tests": int(expected_gate_tests),
            "require_zero_skipped": True,
            "skills": skill_manifest["skills"],
            "skill_bundle": skill_bundle_path,
            "prepare_only": bool(int(prepare_only)),
        },
        f,
        indent=2,
    )
    f.write("\n")
PY

if [[ ${PREPARE_ONLY} -eq 1 ]]; then
    PREPARE_SOURCE="${OPTIMIZATION_BASELINE}"

    PROMPT="${RESULTS}/prompt_1.txt"
    {
        echo "You are preparing to optimize NPU firmware at ${TARGET_FILE}."
        echo ""
        echo "Mode: prepare-only (do not edit files and do not invoke an agent)"
        echo "Goal: ${GOAL} (dim=${DIM}, hidden=${HIDDEN}, num_head=${NUM_HEAD})"
        echo "Canonical unoptimized reference: ${CANONICAL_BASELINE_FILE}"
        echo "Read-only optimization start snapshot: ${PREPARE_SOURCE}"
        echo "Optimization start origin: ${OPTIMIZATION_START_FILE} (${START_SOURCE_KIND})"
        echo "Primary metric: ${PRIMARY_METRIC}"
        echo "Validation scope: ${VALIDATION_SCOPE}"
        echo "Validation coverage: ${VALIDATION_COVERAGE}"
        echo "Expected gate result: ${EXPECTED_GATE_TESTS:-0} passed, 0 skipped"
        echo ""
        echo "=== Mandatory Skills, in order ==="
        skill_num=1
        for sk in "${EFFECTIVE_SKILL_NAMES[@]}"; do
            echo "${skill_num}. jimu-dse/docs/skills/isa/${sk}.md"
            skill_num=$((skill_num + 1))
        done
        echo ""
        echo "=== Constraints ==="
        echo "- Only ${TARGET_FILE} may be modified in a real optimization run"
        echo "- Never modify ${CANONICAL_BASELINE_FILE} or ${OPTIMIZATION_BASELINE}"
        echo "- Never modify tests, emulator, ISS, trace recorder, or hardware model"
        echo "- Never run git stash, git reset, git checkout, git restore, or git clean"
        echo "- Never weaken, skip, filter, or edit validation"
        echo ""
        echo "=== Independent Acceptance Gate (${VALIDATION_SCOPE}) ==="
        echo "${TEST_GATE_CMD}"
        if [[ "${PRIMARY_METRIC}" == "total_bytes" ]]; then
            echo ""
            echo "=== G1 Metric Gate ==="
            echo "seq${SL2} bytes must not increase; seq${SL6} bytes must strictly decrease."
            if [[ "${JIMU_INSTR_GATE}" == "on" ]]; then
                echo "seq${SL6} instructions may regress by at most ${JIMU_INSTR_REGRESSION_LIMIT}."
            else
                echo "seq${SL6} instruction gate is disabled for this run."
            fi
            echo "Apply exactly one new vrf-cache level per candidate."
        fi
        echo ""
        echo "Dynamic probe and DAG data are intentionally omitted in prepare-only mode."
    } > "${PROMPT}"

    log "[PREPARE] Prompt:         ${PROMPT}"
    log "[PREPARE] Skill bundle:   ${SKILL_BUNDLE}"
    log "[PREPARE] Skill manifest: ${SKILL_MANIFEST}"
    log "[PREPARE] Run manifest:   ${RESULTS}/run_manifest.json"
    log "[PREPARE] Firmware was not modified; no probe or agent was run"
    exit 0
fi

BASELINE_VALUE=""
BASELINE_VALUE_SEQ2=""
BEST_VALUE=""
BEST_VALUE_SEQ2=""
BEST_CANDIDATE=""

read_fw_header() {
    head -1 "$1" 2>/dev/null | sed 's/^\/* //;s/ \*\/$//' || echo "unknown"
}

# ── Set starting firmware ─────────────────────────────────────────────
cp "${OPTIMIZATION_BASELINE}" "${TARGET_FILE}"
HEADER=$(read_fw_header "${TARGET_FILE}")
log "[START] ${START_SOURCE_KIND}: ${OPTIMIZATION_START_FILE} (${HEADER})"

restore_baseline() {
    if ! diff -q "${CANONICAL_BASELINE_FILE}" "${TARGET_FILE}" > /dev/null 2>&1; then
        cp "${CANONICAL_BASELINE_FILE}" "${TARGET_FILE}"
        log "[BASELINE] Restored canonical source from ${CANONICAL_BASELINE_FILE}"
    fi
}

restore_best() {
    local source_file="${BEST_CANDIDATE:-${OPTIMIZATION_BASELINE}}"
    cp "${source_file}" "${TARGET_FILE}"
    log "[ROLLBACK] Restored last accepted firmware from ${source_file}"
}

on_exit() {
    restore_baseline
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── Initialize: build C kernel library ────────────────────────────
log "[INIT] Building C kernel library..."
if [[ ! -f "_build/kernels/libnpukernels.so" ]]; then
    make kernels > /dev/null 2>&1
fi
log "[INIT] Kernels ready"

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

build_firmware() {
    local SL=$1
    if type build_firmware_workload &>/dev/null; then
        build_firmware_workload "$SL"
        return $?
    fi
    # Fallback to bert build
    local dim=${DIM} hidden=${HIDDEN} num_head=${NUM_HEAD}
    local proj_base=$((hidden * SL + 4))
    local mat_size=$((hidden * hidden))
    local stride=$((mat_size + hidden))
    local num_tiles=$((hidden / dim))
    local ln_base=$((proj_base + 6 * stride))
    local ln_size=$((num_tiles * 8))
    CC="${CC}" \
    NATIVE_DIM=${dim} SEQ_LEN=${SL} \
    _HIDDEN_SIZE=${hidden} _PROJ_BASE=${proj_base} \
    _MAT_SIZE=${mat_size} _STRIDE=${stride} \
    _NUM_TILES=${num_tiles} \
    _LN1_GAMMA=${ln_base} _LN1_BETA=$((ln_base + ln_size)) \
    _LN2_GAMMA=$((ln_base + 2*ln_size)) _LN2_BETA=$((ln_base + 3*ln_size)) \
    _SCRATCH=1280 NUM_HEAD=${num_head} \
    make -C firmware BUILD_DIR=build_dim${dim} TARGET=${MAKE_TARGET:-bert} clean all > /dev/null 2>&1
    local elf_path="firmware/build_dim${dim}/${MAKE_TARGET:-bert}.elf"
    local build_meta="${elf_path}.jimu-build.json"
    if [[ ! -f "${elf_path}" ]]; then
        log "[BUILD] FAILED for seq=${SL}"
        return 1
    fi
    python3 - "${elf_path}" "${build_meta}" \
        "${dim}" "${hidden}" "${SL}" "${num_head}" \
        "${WORKLOAD_NAME}" <<'PY' || return 1
import hashlib
import json
import sys

elf_path, output_path, dim, hidden, seq_len, num_head, workload = sys.argv[1:]
with open(elf_path, "rb") as elf_file:
    elf_sha256 = hashlib.sha256(elf_file.read()).hexdigest()
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(
        {
            "elf_path": elf_path,
            "elf_sha256": elf_sha256,
            "dim": int(dim),
            "hidden_size": int(hidden),
            "seq_len": int(seq_len),
            "num_head": int(num_head),
            "workload": workload,
        },
        output_file,
        indent=2,
    )
    output_file.write("\n")
PY
    log "[BUILD] seq=${SL} OK"
    return 0
}

run_correctness_gate() {
    local label=$1
    local gate_log="${RESULTS}/validation_${label}.log"
    local gate_json="${RESULTS}/validation_${label}.json"
    local status

    log "[GATE] Running independent correctness suite..."
    eval "${TEST_GATE_CMD}" > "${gate_log}" 2>&1
    status=$?

    python3 jimu-dse/scripts/validation_gate.py \
        --log "${gate_log}" \
        --output "${gate_json}" \
        --pytest-returncode "${status}" \
        --expected-passed "${EXPECTED_GATE_TESTS:-0}" \
        --scope "${VALIDATION_SCOPE}" \
        --command "${TEST_GATE_CMD}"
    status=$?

    if [[ ${status} -ne 0 ]]; then
        log "[GATE] REJECT: $(python3 -c "
import json
d=json.load(open('${gate_json}'))
print(d.get('failure_reason') or f'return code {d.get(\"returncode\")}')
" 2>/dev/null || echo "validation failure")"
        tail -20 "${gate_log}" >&2
        return 1
    fi

    local passed
    passed=$(python3 -c "
import json
d=json.load(open('${gate_json}'))
print(d.get('passed', 0))
" 2>/dev/null || echo 0)
    log "[GATE] PASS (${passed}/${EXPECTED_GATE_TESTS:-0}, skipped=0)"
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════
# PROBE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

generate_agent_dag() {
    local output_dir=$1
    local short_dir="${output_dir}/seq${SL2}"
    local proof_root="${output_dir}/proof"

    mkdir -p "${short_dir}" "${proof_root}"
    python3 jimu-dse/scripts/visualize_graph.py --phase micro \
        --workload "${WORKLOAD_NAME}" \
        --dim ${DIM} --hidden ${HIDDEN} --seq-len ${SL2} \
        --num-head ${NUM_HEAD} -o "${short_dir}" --no-render || return 1
    python3 jimu-dse/scripts/visualize_graph.py --phase micro \
        --workload "${WORKLOAD_NAME}" \
        --dim ${DIM} --hidden ${HIDDEN} --seq-len ${SL6} \
        --num-head ${NUM_HEAD} -o "${output_dir}" --no-render || return 1

    local -a proof_specs=(
        "${DIM},${HIDDEN},${SL2},${NUM_HEAD}"
        "${DIM},${HIDDEN},${SL6},${NUM_HEAD}"
    )
    if [[ "${WORKLOAD_NAME}" == "bert" ]]; then
        if [[ "${VALIDATION_SCOPE}" == "all" || \
              "${VALIDATION_SCOPE}" == "dim2" ]]; then
            proof_specs+=("2,4,${SL2},2" "2,4,${SL6},2")
        fi
        if [[ "${VALIDATION_SCOPE}" == "all" || \
              "${VALIDATION_SCOPE}" == "dim4" ]]; then
            proof_specs+=(
                "4,4,${SL2},2" "4,4,${SL6},2"
                "4,8,${SL2},2" "4,8,${SL6},2"
            )
        fi
    fi

    local -a merge_args=(
        python3 jimu-dse/scripts/merge_dag_sequences.py
        --dag "${SL2}=${short_dir}"
        --dag "${SL6}=${output_dir}"
        -o "${output_dir}"
    )
    local -A seen_proof_specs=()
    local proof_spec proof_dim proof_hidden proof_seq proof_heads
    local proof_id proof_dir
    for proof_spec in "${proof_specs[@]}"; do
        IFS=, read -r proof_dim proof_hidden proof_seq proof_heads <<< "${proof_spec}"
        proof_id="dim${proof_dim}-h${proof_hidden}-head${proof_heads}-seq${proof_seq}"
        [[ -n "${seen_proof_specs[${proof_id}]:-}" ]] && continue
        seen_proof_specs[${proof_id}]=1
        merge_args+=(--required-config "${proof_id}")

        if [[ "${proof_dim}" == "${DIM}" && \
              "${proof_hidden}" == "${HIDDEN}" && \
              "${proof_heads}" == "${NUM_HEAD}" ]]; then
            continue
        fi
        proof_dir="${proof_root}/${proof_id}"
        python3 jimu-dse/scripts/visualize_graph.py --phase micro \
            --workload "${WORKLOAD_NAME}" \
            --dim "${proof_dim}" --hidden "${proof_hidden}" \
            --seq-len "${proof_seq}" --num-head "${proof_heads}" \
            -o "${proof_dir}" --no-render || return 1
        merge_args+=(--proof-dag "${proof_dir}")
    done
    "${merge_args[@]}" || return 1
}

probe() {
    local SL=$1
    local dag_mode="${2:-refresh}"
    local dim=${DIM} hidden=${HIDDEN} nh=${NUM_HEAD}
    case "${dag_mode}" in
        refresh|metric-only) ;;
        *)
            log "[PROBE] Unknown DAG mode: ${dag_mode}"
            return 1
            ;;
    esac
    build_firmware ${SL} || return 1

    local elf_path="firmware/build_dim${dim}/${MAKE_TARGET:-bert}.elf"

    # Generate concrete seq2/seq6 DAGs and cross-sequence reuse evidence.
    if [[ "${SL}" == "${SL6}" ]] && [[ "${dag_mode}" == "refresh" ]]; then
        log "[PROBE] Generating seq${SL2}/seq${SL6} DAG evidence..."
        if ! generate_agent_dag "${RESULTS}/dag_agent" > /dev/null 2>&1; then
            log "[PROBE] DAG evidence generation failed"
            return 1
        fi
    fi

    if ! python3 jimu-dse/scripts/npu_workload_probe.py \
        --workload "${WORKLOAD_NAME}" \
        --dim "${dim}" --hidden "${hidden}" --seq-len "${SL}" \
        --num-head "${nh}" --elf "${elf_path}" \
        --build-metadata "${elf_path}.jimu-build.json" \
        --output "${RESULTS}/p${SL}_probe.json"; then
        return 1
    fi
    python3 -c "import json; print(json.load(open('${RESULTS}/p${SL}_probe.json')).get('total_bytes',0))"
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP — always incremental within a run
# ═══════════════════════════════════════════════════════════════════════════

if ! run_correctness_gate "baseline"; then
    die "Starting firmware failed the independent correctness gate"
fi
BEST_CANDIDATE="${RESULTS}/candidate_best.c"
cp "${TARGET_FILE}" "${BEST_CANDIDATE}"
log "[GATE] Starting firmware recorded as the first accepted candidate"

for ((iter=1; iter<=JIMU_MAX_ITER; iter++)); do
    log "--- Iteration ${iter} ---"

    # ---- PHASE 1: PROBE --------------------------------------------------
    log "[PROBE] seq=${SL2}..."
    if ! B2=$(probe "${SL2}" metric-only); then
        die "Probe failed for seq=${SL2}"
    fi
    log "[PROBE] seq=${SL6}..."
    if ! B6=$(probe "${SL6}" refresh); then
        die "Probe failed for seq=${SL6}"
    fi

    # Also extract tile info for display
    PROBE_TILES=$(python3 -c "
import json
d = json.load(open('${RESULTS}/p${SL6}_probe.json'))
ts = d.get('tile_structure', {})
n = ts.get('num_tiles', 0)
mvpp = ts.get('mv_mul_per_projection')
mv = d.get('mv_mul_count', 0)
mrd = d.get('mat_rd_ops', 0)
detail = f'tile={n}x{n} MV/proj={mvpp}' if mvpp is not None else f'native-tiles={n}'
print(f'MV={mv} MRD={mrd} {detail}')
" 2>/dev/null)
    log "  seq=${SL2}: ${B2}B  seq=${SL6}: ${B6}B (${PROBE_TILES})"
    if [[ -z "${BASELINE_VALUE}" ]] && [[ "${B6}" != "0" ]]; then
        BASELINE_VALUE="${B6}"
        BASELINE_VALUE_SEQ2="${B2}"
        BEST_VALUE="${B6}"
        BEST_VALUE_SEQ2="${B2}"
        log "  Baseline for this run set to ${BASELINE_VALUE}B"
    fi

    # ---- PHASE 2: ANALYZE ------------------------------------------------
    RATIO=$(python3 -c "print(f'{$B6/$B2:.1f}x')" 2>/dev/null || echo "0")
    log "[ANALYZE] seq${SL6}/seq${SL2} ratio: ${RATIO}"

    # Extract probe data for prompt
    PROBE_DATA=$(python3 -c "
import json
d = json.load(open('${RESULTS}/p${SL6}_probe.json'))
ts = d.get('tile_structure', {})
dram = d.get('total_bytes', 0)
mv = d.get('mv_mul_count', 0)
mrd = d.get('mat_rd_ops', 0)
tiles = ts.get('num_tiles', 0)
mvpp = ts.get('mv_mul_per_projection')
detail = f'tiles={tiles}x{tiles} MV/proj={mvpp}' if mvpp is not None else f'native-tiles={tiles}'
print(f'DRAM={dram}B MV_MUL={mv} M_RD={mrd} {detail}')
" 2>/dev/null)

    METRIC_BEFORE_SEQ2="${RESULTS}/metric_before_${iter}_seq${SL2}.json"
    METRIC_BEFORE_SEQ6="${RESULTS}/metric_before_${iter}_seq${SL6}.json"
    cp "${RESULTS}/p${SL2}_probe.json" "${METRIC_BEFORE_SEQ2}"
    cp "${RESULTS}/p${SL6}_probe.json" "${METRIC_BEFORE_SEQ6}"

    # Freeze the exact DAG that produced this iteration's candidate IDs.
    # Candidate metric probes must never mutate this evidence snapshot.
    DAG_BEFORE_DIR="${RESULTS}/dag_before_iter${iter}"
    mkdir -p "${DAG_BEFORE_DIR}"
    cp -a "${RESULTS}/dag_agent/." "${DAG_BEFORE_DIR}/"
    for required_dag_file in candidates.json macro_candidates.json \
        micro_ops.jsonl edges.jsonl \
        lifetimes.json run_metadata.json multiseq_metadata.json \
        loop_invariants.json multiseq_summary.md candidate_evidence.jsonl \
        allocation_proof.json allocation_summary.md \
        next_macro_contract.json next_macro_contract.md; do
        [[ -s "${DAG_BEFORE_DIR}/${required_dag_file}" ]] || \
            die "Missing immutable before-DAG artifact: ${required_dag_file}"
    done
    log "[DAG] Iteration input snapshot: ${DAG_BEFORE_DIR}"

    CONTRACT_STATUS=$(python3 -c "import json; print(json.load(open('${DAG_BEFORE_DIR}/next_macro_contract.json'))['status'])")
    if [[ "${CONTRACT_STATUS}" != "ready" ]]; then
        log "[DAG] No eligible deterministic implementation contract remains"
        break
    fi
    CONTRACT_ID=$(python3 -c "import json; print(json.load(open('${DAG_BEFORE_DIR}/next_macro_contract.json'))['selected_macro']['id'])")
    log "[DAG] Selected deterministic contract: ${CONTRACT_ID}"

    # ---- PHASE 4: AGENT --------------------------------------------------
    PROMPT="${RESULTS}/prompt_${iter}.txt"

    FW_HEADER=$(read_fw_header "${TARGET_FILE}")
    {
        echo "You are optimizing NPU firmware at ${TARGET_FILE}."
        echo ""
        echo "Goal: ${GOAL} (dim=${DIM}, hidden=${HIDDEN}, num_head=${NUM_HEAD})"
        echo "Current header: ${FW_HEADER}"
        echo "Canonical unoptimized reference: ${CANONICAL_BASELINE_FILE}"
        echo "Run optimization baseline: ${OPTIMIZATION_BASELINE}"
        echo "Run start origin: ${OPTIMIZATION_START_FILE} (${START_SOURCE_KIND})"
        echo "Primary metric: ${PRIMARY_METRIC}"
        echo "Validation coverage: ${VALIDATION_COVERAGE}"
        echo "Instruction count (seq=${SL6}): $(python3 -c "import json; print(json.load(open('${METRIC_BEFORE_SEQ6}')).get('instr_count',0))" 2>/dev/null)"
        echo "MV_MUL count: $(python3 -c "import json; print(json.load(open('${RESULTS}/p${SL6}_probe.json')).get('mv_mul_count',0))" 2>/dev/null)"
        echo "M_RD_DRAM ops (weight tile loads): $(python3 -c "import json; print(json.load(open('${RESULTS}/p${SL6}_probe.json')).get('mat_rd_ops',0))" 2>/dev/null)"
        echo "Current DRAM: seq=${SL2}=${B2}B, seq=${SL6}=${B6}B (ratio ${RATIO})"
        echo ""
        echo "=== Deterministic DAG Contract ==="
        echo "Contract JSON: ${DAG_BEFORE_DIR}/next_macro_contract.json"
        cat "${DAG_BEFORE_DIR}/next_macro_contract.md"
        echo ""
        echo "=== Skills to Apply ==="
        skill_num=1
        for sk in "${EFFECTIVE_SKILL_NAMES[@]}"; do
            echo "${skill_num}. Read and obey jimu-dse/docs/skills/isa/${sk}.md."
            skill_num=$((skill_num + 1))
        done
        echo "All listed skills are mandatory and must be applied in order."
        echo "Common constraints override transformations; self-verify governs validation."
        echo "next_macro_contract.json is the authoritative selection and proof."
        echo "Implement exactly ${CONTRACT_ID}; do not select another macro, regenerate"
        echo "the DAG, reopen full evidence, or re-prove a valid generated contract."
        echo "Declare the exact marker supplied by the contract. The independent gate"
        echo "requires every exact Tensor/address member to reduce and rejects positive"
        echo "DRAM reductions outside the selected scope."
        echo ""
        if [[ "${GOAL}" == "compute-optimization" || "${GOAL}" == "combined" ]]; then
            echo "=== Focus: Correct Single-Tile Operation (dim=${DIM}) ==="
            echo "The baseline firmware was designed for dim=2 (multi-tile). You are upgrading"
            echo "it to run correctly at dim=${DIM} (single-tile, num_tiles=1)."
            echo "Rewrite mvm_tiled_q, attention, layer_norm, and gelu to use straight-line"
            echo "single-tile operations. Fix heads_per_tile=2 attention masking."
            echo "The primary metric is test_pass — make dim${DIM}-h${HIDDEN} tests pass."
            echo "DRAM reduction is secondary and will be optimized in a separate pass."
            echo ""
            echo "=== Structural Statistics ==="
            echo "MV_MUL count: currently optimal (num_tiles=1 at runtime)."
            echo "The goal is CODE CORRECTNESS: remove tile loop overhead and fix masking."
        elif [[ "${SKILLS}" == *"vrf-cache"* ]]; then
            echo "=== G1 Staged Metric Gate ==="
            echo "- Apply exactly one new vrf-cache level in this candidate."
            echo "- seq${SL2} total_bytes must not increase."
            echo "- seq${SL6} total_bytes must strictly decrease."
            if [[ "${JIMU_INSTR_GATE}" == "on" ]]; then
                echo "- seq${SL6} instruction count may increase by at most ${JIMU_INSTR_REGRESSION_LIMIT}."
            else
                echo "- seq${SL6} instruction-count rejection is disabled for this run."
            fi
            echo "- Print the required VRF allocation and lifetime proof before editing."
            echo "- Declare exactly one chosen macro, or one primitive only as fallback."
            echo "- Keep all existing functions intact — apply VRF cache transformations as instructed."
        else
            echo "- Follow the skill instructions exactly as provided."
        fi
        echo "=== Constraints ==="
        echo "- Only modify ${TARGET_FILE}"
        echo "- Do NOT modify ${CANONICAL_BASELINE_FILE} or ${OPTIMIZATION_BASELINE}"
        echo "- Do NOT modify tests, the emulator, ISS, trace recorder, hardware model, or any other file"
        echo "- Do NOT inspect or modify closed-loop, gate, DAG-generator, or skill source"
        echo "- Do NOT regenerate or independently reconstruct DAG/allocation evidence"
        echo "- Do NOT run git stash, git reset, git checkout, git restore, or git clean"
        echo "- Do NOT weaken, skip, filter, or edit validation"
        echo "- File must compile: gcc for RISC-V (NATIVE_DIM=${DIM})"
        echo ""
        echo "=== Self-Verify ==="
        echo "${TEST_VERIFY_CMD}"
        echo "Expected: ${EXPECTED_GATE_TESTS:-0} passed, 0 skipped"
        echo ""
        echo "=== Independent Acceptance Gate (${VALIDATION_SCOPE}) ==="
        echo "${TEST_GATE_CMD}"
    } > "${PROMPT}"

    log "[AGENT] Prompt: $(wc -c < ${PROMPT}) bytes"
    CANDIDATE="${RESULTS}/candidate_${iter}.c"
    PRE_AGENT="${RESULTS}/pre_agent_${iter}.c"
    cp "${TARGET_FILE}" "${PRE_AGENT}"
    AGENT_STATUS=0

    if [[ "${AGENT}" == "opencode" ]]; then
        if command -v opencode &>/dev/null; then
            log "[AGENT] Invoking opencode (model: ${OPENCODE_MODEL})..."
            OPENCODE_SKILL_ARGS=()
            for sk in "${EFFECTIVE_SKILL_NAMES[@]}"; do
                OPENCODE_SKILL_ARGS+=(
                    -f "${REPO_ROOT}/.opencode/skills/${sk}/SKILL.md"
                )
            done
            OPENCODE_DAG_ARGS=()
            for dag_file in next_macro_contract.json next_macro_contract.md; do
                if [[ -s "${DAG_BEFORE_DIR}/${dag_file}" ]]; then
                    OPENCODE_DAG_ARGS+=(
                        -f "${DAG_BEFORE_DIR}/${dag_file}"
                    )
                fi
            done
            AGENT_STATUS=1
            for ((agent_attempt=1; agent_attempt<=JIMU_AGENT_RETRIES; agent_attempt++)); do
                if (( agent_attempt > 1 )); then
                    cp "${PRE_AGENT}" "${TARGET_FILE}"
                fi
                log "[AGENT] OpenCode attempt ${agent_attempt}/${JIMU_AGENT_RETRIES}"
                timeout "${JIMU_AGENT_TIMEOUT}" opencode run --model "${OPENCODE_MODEL}" \
                    "${OPENCODE_SKILL_ARGS[@]}" \
                    "${OPENCODE_DAG_ARGS[@]}" \
                    -f "${RESULTS}/p${SL6}_probe.json" \
                    -f "${TARGET_FILE}" \
                    --dangerously-skip-permissions \
                    "$(cat ${PROMPT})" 2>&1 | tee -a "${RESULTS}/opencode_output.log" | tail -20
                AGENT_STATUS=${PIPESTATUS[0]}
                (( AGENT_STATUS == 0 )) && break
                log "[AGENT] Attempt ${agent_attempt} failed with status ${AGENT_STATUS}"
            done
            [[ -f "${REPO_ROOT}/${TARGET_FILE}" ]] && \
                cp "${REPO_ROOT}/${TARGET_FILE}" "${CANDIDATE}"
        else
            log "[AGENT] opencode not available — using unmodified firmware"
            cp "${TARGET_FILE}" "${CANDIDATE}"
        fi
    elif command -v pi &>/dev/null; then
        log "[AGENT] Invoking pi (timeout: ${JIMU_AGENT_TIMEOUT}s)..."
        # Pi accepts one --skill file, so skillctl composes every effective skill.
        pi --skill "${SKILL_BUNDLE}" \
           -p "$(cat ${PROMPT})" 2>/dev/null
        AGENT_STATUS=$?
        [[ -f "${REPO_ROOT}/${TARGET_FILE}" ]] && \
            cp "${REPO_ROOT}/${TARGET_FILE}" "${CANDIDATE}"
        [[ ! -s "${CANDIDATE}" ]] && pi -p "$(cat ${PROMPT})" 2>/dev/null | \
            sed -n '/^```/,/^```/{/^```/d;p}' > "${CANDIDATE}"
        [[ ! -s "${CANDIDATE}" ]] && cp "${TARGET_FILE}" "${CANDIDATE}"
    else
        log "[AGENT] No agent available — using unmodified firmware"
        cp "${TARGET_FILE}" "${CANDIDATE}"
    fi

    if [[ ${AGENT_STATUS} -ne 0 ]]; then
        log "[AGENT] Failed with status ${AGENT_STATUS}; rejecting iteration"
        restore_best
        continue
    fi

    [[ ! -f "${CANDIDATE}" ]] && log "[AGENT] No candidate" && restore_best && continue
    log "[AGENT] Candidate: $(wc -l < ${CANDIDATE}) lines"

    # ---- Check: agent actually changed anything? ----
    if diff -q "${CANDIDATE}" "${PRE_AGENT}" > /dev/null 2>&1; then
        log "[AGENT] Candidate unchanged from iteration input"
        restore_best
        continue
    fi

    # ---- PHASE 5: VALIDATE -----------------------------------------------
    cp "${CANDIDATE}" "${TARGET_FILE}"
    if ! run_correctness_gate "${iter}"; then
        log "[VALIDATE] Candidate rejected by correctness gate"
        restore_best
        continue
    fi

    log "[VALIDATE] Correctness passed; measuring emulator traffic..."
    if [[ "${PRIMARY_METRIC}" == "total_bytes" ]]; then
        METRIC_AFTER_SEQ2="${RESULTS}/metric_after_${iter}_seq${SL2}.json"
        METRIC_AFTER_SEQ6="${RESULTS}/metric_after_${iter}_seq${SL6}.json"

        if ! B2_NEW=$(probe "${SL2}" metric-only); then
            log "[VALIDATE] Candidate seq${SL2} metric probe failed"
            restore_best
            continue
        fi
        cp "${RESULTS}/p${SL2}_probe.json" "${METRIC_AFTER_SEQ2}"

        if ! B6_NEW=$(probe "${SL6}" metric-only); then
            log "[VALIDATE] Candidate seq${SL6} metric probe failed"
            restore_best
            continue
        fi
        cp "${RESULTS}/p${SL6}_probe.json" "${METRIC_AFTER_SEQ6}"

        python3 jimu-dse/scripts/metric_gate.py \
            --before-seq2 "${METRIC_BEFORE_SEQ2}" \
            --before-seq6 "${METRIC_BEFORE_SEQ6}" \
            --after-seq2 "${METRIC_AFTER_SEQ2}" \
            --after-seq6 "${METRIC_AFTER_SEQ6}" \
            --instruction-gate "${JIMU_INSTR_GATE}" \
            --instruction-regression-limit "${JIMU_INSTR_REGRESSION_LIMIT}" \
            --output "${RESULTS}/val_${iter}.json" \
            2> "${RESULTS}/val_${iter}.stderr"
        METRIC_STATUS=$?

        if [[ ${METRIC_STATUS} -ne 0 ]]; then
            log "[VALIDATE] G1 metric gate rejected candidate:"
            python3 -c "
import json
d=json.load(open('${RESULTS}/val_${iter}.json'))
for reason in d.get('failure_reasons', []):
    print(f'  - {reason}')
" >&2
            restore_best
            continue
        fi

        I6_BEFORE=$(python3 -c "import json; print(json.load(open('${METRIC_BEFORE_SEQ6}')).get('instr_count',0))")
        I6_NEW=$(python3 -c "import json; print(json.load(open('${METRIC_AFTER_SEQ6}')).get('instr_count',0))")
        SAVED=$((${BASELINE_VALUE:-0} - B6_NEW))
        log "[VALIDATE] G1 gate PASS: seq${SL2} ${B2}B -> ${B2_NEW}B; seq${SL6} ${B6}B -> ${B6_NEW}B"
        log "[VALIDATE] seq${SL6} instructions: ${I6_BEFORE} -> ${I6_NEW}"
    else
        if ! build_firmware ${SL6} > /dev/null 2>&1; then
            log "[VALIDATE] Candidate failed metric build"
            restore_best
            continue
        fi

        python3 -c "
import sys, json
sys.path.insert(0, '.')
from emulator.npu_device_mini import NpuDeviceMini, MEM_DRAM
from emulator.trace_recorder import TraceRecorder
from iss.mini_rv64 import MiniRV64
import numpy as np

dim=${DIM}; h=${HIDDEN}; sl=${SL6}; nh=${NUM_HEAD}
npu = NpuDeviceMini(native_dim=dim)
npu.set_hidden_size(h); npu.set_seq_len(sl)
npu._vrf[MEM_DRAM][0:h] = np.zeros(h, dtype=np.float32)
rec = TraceRecorder(npu)
cpu = MiniRV64()
cpu.set_mmio_device(rec)
cpu.load_elf('firmware/build_dim${DIM}/${MAKE_TARGET:-bert}.elf')
cpu.run(cycles=300000)
ds = npu.get_dram_stats()
t_el = ds.get('vec_rd_elements',0)+ds.get('vec_wr_elements',0)+ds.get('mat_rd_elements',0)+ds.get('mat_wr_elements',0)
t_b = t_el * 4
print(json.dumps({'total_bytes': t_b, 'dram_stats': dict(ds), 'instr_count': len(rec.inst_trace)}))
" 2>"${RESULTS}/val_${iter}.stderr" > "${RESULTS}/val_${iter}.json"
        METRIC_STATUS=$?

        if [[ ${METRIC_STATUS} -eq 0 ]] && [[ -s "${RESULTS}/val_${iter}.json" ]]; then
            B6_NEW=$(python3 -c "import json; print(json.load(open('${RESULTS}/val_${iter}.json')).get('total_bytes',0))" 2>/dev/null || echo 0)
            SAVED=$((${BASELINE_VALUE:-0} - B6_NEW))
            log "[VALIDATE] DRAM: ${B6_NEW}B (saved ${SAVED}B vs run-start ${BASELINE_VALUE:-0}B)"
        else
            log "[VALIDATE] Metric collection failed; rejecting candidate"
            restore_best
            continue
        fi
    fi

    # Save diff against baseline file
    diff "${OPTIMIZATION_BASELINE}" "${CANDIDATE}" > "${RESULTS}/diff_${iter}.patch" 2>/dev/null || true
    log "[SAVE] Diff saved ($(wc -l < "${RESULTS}/diff_${iter}.patch" 2>/dev/null || echo 0) lines)"

    # ---- DAG: regenerate concrete seq2/seq6 evidence for the candidate ----
    log "[DAG] Generating seq${SL2}/seq${SL6} DAG evidence..."
    DAG_AFTER_DIR="${RESULTS}/dag_iter${iter}"
    if ! generate_agent_dag "${DAG_AFTER_DIR}" > /dev/null 2>&1; then
        log "[DAG] Candidate DAG generation failed"
        if [[ "${PRIMARY_METRIC}" == "total_bytes" ]] && \
           [[ "${JIMU_DAG_EVIDENCE_GATE}" == "on" ]]; then
            log "[DAG] Evidence gate requires a candidate DAG; rejecting candidate"
            restore_best
            continue
        fi
    else
        log "[DAG] DAG graphs saved to ${DAG_AFTER_DIR}"

        DAG_CANDIDATE_REQUIRED="off"
        DAG_DIFF_ARGS=(
            python3 jimu-dse/scripts/dag_diff_gate.py
            --before-dag "${DAG_BEFORE_DIR}"
            --after-dag "${DAG_AFTER_DIR}"
            --candidate-source "${CANDIDATE}"
            --candidate-required "${DAG_CANDIDATE_REQUIRED}"
            --gate "${JIMU_DAG_EVIDENCE_GATE}"
            --output "${RESULTS}/dag_diff_${iter}.json"
            --summary "${RESULTS}/dag_diff_${iter}.md"
        )
        if [[ "${PRIMARY_METRIC}" == "total_bytes" ]]; then
            DAG_CANDIDATE_REQUIRED="on"
            DAG_DIFF_ARGS=(
                python3 jimu-dse/scripts/dag_diff_gate.py
                --before-dag "${DAG_BEFORE_DIR}"
                --after-dag "${DAG_AFTER_DIR}"
                --candidate-source "${CANDIDATE}"
                --before-seq2 "${METRIC_BEFORE_SEQ2}"
                --before-seq6 "${METRIC_BEFORE_SEQ6}"
                --after-seq2 "${METRIC_AFTER_SEQ2}"
                --after-seq6 "${METRIC_AFTER_SEQ6}"
                --candidate-required "${DAG_CANDIDATE_REQUIRED}"
                --gate "${JIMU_DAG_EVIDENCE_GATE}"
                --output "${RESULTS}/dag_diff_${iter}.json"
                --summary "${RESULTS}/dag_diff_${iter}.md"
            )
        fi

        if ! "${DAG_DIFF_ARGS[@]}"; then
            log "[DAG] PR4 evidence gate rejected candidate:"
            python3 - "${RESULTS}/dag_diff_${iter}.json" <<'PY' >&2
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
for reason in result.get("failure_reasons", []):
    print(f"  - {reason}")
PY
            restore_best
            continue
        fi
        log "[DAG] PR4 evidence report: ${RESULTS}/dag_diff_${iter}.md"
    fi

    # ---- PHASE 6: DEPLOY ------------------------------------------------
    if [[ "${PRIMARY_METRIC}" == "test_pass" ]]; then
        cp "${CANDIDATE}" "${BEST_CANDIDATE}"
        [[ -d "${DAG_AFTER_DIR}" ]] && \
            cp -a "${DAG_AFTER_DIR}/." "${RESULTS}/dag_agent/"
        BEST_VALUE=1
        log "[DEPLOY] Candidate iter${iter} passed the full correctness suite"
        log "Converged: all required tests pass"
        break
    fi

    if [[ -z "${BEST_VALUE}" ]] || [[ ${B6_NEW} -lt ${BEST_VALUE} ]]; then
        BEST_VALUE=${B6_NEW}
        BEST_VALUE_SEQ2=${B2_NEW}
        cp "${CANDIDATE}" "${BEST_CANDIDATE}"
        [[ -d "${DAG_AFTER_DIR}" ]] && \
            cp -a "${DAG_AFTER_DIR}/." "${RESULTS}/dag_agent/"
        log "[DEPLOY] Accepted iter${iter}: seq${SL2}=${B2_NEW}B, seq${SL6}=${B6_NEW}B (saved ${SAVED}B)"
    else
        log "[DEPLOY] Rejected iter${iter}: ${B6_NEW}B is not better than ${BEST_VALUE}B"
        restore_best
        continue
    fi

    IMPROV_RATIO=$(python3 -c "print(max(0.0, (${B6}-${B6_NEW})/max(${B6},1)))" 2>/dev/null || echo "0")
    IMPROV_PCT=$(python3 -c "print(f'{${IMPROV_RATIO}*100.0:.2f}')" 2>/dev/null || echo "0")
    log "[CONVERGE] Iteration improvement: ${IMPROV_PCT}% (${PROBE_DATA})"
    CONVERGED=$(python3 -c "print('1' if ${IMPROV_RATIO} < ${JIMU_THRESHOLD} else '0')" 2>/dev/null || echo "0")
    if [[ "${CONVERGED}" == "1" ]]; then
        if [[ "${SKILLS}" == *"vrf-cache"* ]]; then
            log "[CONVERGE] Improvement is below threshold, but staged VRF optimization continues"
        else
            log "Converged: ${IMPROV_RATIO} < threshold ${JIMU_THRESHOLD}"
            break
        fi
    fi
    continue
done

# ── Save best candidate for --resume ────────────────────────────────────
if [[ -f "${BEST_CANDIDATE}" ]]; then
    log "[SAVE] Best accepted candidate (${BEST_VALUE:-baseline}) saved to ${BEST_CANDIDATE}"
fi

# ── Final restore to baseline ──────────────────────────────────────────
restore_baseline

log ""
log "===== Done ====="
log "Goal:       ${GOAL}"
log "Agent:      ${AGENT}"
log "Validation: ${VALIDATION_SCOPE}"
log "Results:    ${RESULTS}"
log "Reference:  ${CANONICAL_BASELINE_FILE}"
log "Run start:  ${OPTIMIZATION_START_FILE}"
if [[ "${PRIMARY_METRIC}" == "test_pass" ]]; then
    TEST_RESULT="FAIL"
    [[ "${BEST_VALUE}" == "1" ]] && TEST_RESULT="PASS"
    log "Baseline:   ${OPTIMIZATION_START_FILE}"
    log "Target:     dim${DIM}-h${HIDDEN} tests pass"
    log "Result:     ${TEST_RESULT}"
else
    log "Baseline:   seq${SL2}=${BASELINE_VALUE_SEQ2}B, seq${SL6}=${BASELINE_VALUE}B"
    log "Best:       seq${SL2}=${BEST_VALUE_SEQ2}B, seq${SL6}=${BEST_VALUE}B"
    if [[ -n "${BASELINE_VALUE}" ]] && [[ -n "${BEST_VALUE}" ]] && [[ ${BEST_VALUE} -lt ${BASELINE_VALUE} ]]; then
        SAVED=$((BASELINE_VALUE - BEST_VALUE))
        PCT=$(python3 -c "print(f'{(${SAVED}*100.0/${BASELINE_VALUE}):.1f}')" 2>/dev/null || echo "0")
        log "Improvement: ${SAVED}B (${PCT}%)"
    fi
fi
log "Continue:   ./jimu-dse/scripts/npu_closed_loop.sh --goal ${GOAL} --validation-dim ${VALIDATION_DIM_REQUEST} --start-from ${RESULTS}/candidate_best.c"
