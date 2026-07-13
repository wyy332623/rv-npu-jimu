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
#   ./jimu-dse/scripts/npu_closed_loop.sh --goal combined --agent opencode --resume <dir>
set -uo pipefail

JIMU_MAX_ITER="${JIMU_MAX_ITER:-5}"
JIMU_THRESHOLD="${JIMU_THRESHOLD:-0.15}"
# opencode agents need more time for multi-step file analysis and editing
JIMU_AGENT_TIMEOUT="${JIMU_AGENT_TIMEOUT:-1800}"
RUN_TAG="run-$(date +%Y%m%d-%H%M%S)-$$"
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
CC="${CC:-riscv64-unknown-elf-gcc}"

RESUME_DIR=""
AGENT="pi"
OPENCODE_MODEL="${OPENCODE_MODEL:-opencode/big-pickle}"
GOAL="dram-optimization"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME_DIR="$2"; shift 2 ;;
        --agent)
            AGENT="$2"
            if [[ "$AGENT" != "pi" && "$AGENT" != "opencode" ]]; then
                echo "Unknown agent: $AGENT (use pi or opencode)" >&2
                exit 1
            fi
            shift 2 ;;
        --model) OPENCODE_MODEL="$2"; shift 2 ;;
        --workload) WORKLOAD="$2"; shift 2 ;;
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
    TEST_VERIFY_CMD='python3 -m pytest tests/integration/test_bert_e2e.py -k "dim${DIM} and h${HIDDEN}" -s --no-header 2>&1 | grep max_diff'
    TEST_CONVERGE_CMD='python3 -m pytest tests/integration/test_bert_e2e.py -k "dim${DIM} and h${HIDDEN}" --no-header -q 2>&1 | tail -3'
fi

# DIM, HIDDEN, NUM_HEAD, SEQ_LENS, BASELINE_FILE, SKILLS, PRIMARY_METRIC
# are now set from the goal config.
read -ra SEQ_ARRAY <<< "${SEQ_LENS}"
SL2="${SEQ_ARRAY[0]}"
SL6="${SEQ_ARRAY[1]}"

RESULTS="${REPO_ROOT}/jimu-dse/results/${RUN_TAG}"
mkdir -p "${RESULTS}"

log "===== NPU Closed Loop ====="
log "Goal:       ${GOAL} (dim=${DIM}, hidden=${HIDDEN}, heads=${NUM_HEAD})"
log "Agent:      ${AGENT}"
log "Run tag:    ${RUN_TAG}"
log "Baseline:   ${BASELINE_FILE}"
log "Skills:     ${SKILLS}"
log "Metric:     ${PRIMARY_METRIC}"

BASELINE_VALUE=""
BEST_VALUE=""

read_fw_header() {
    head -1 "$1" 2>/dev/null | sed 's/^\/* //;s/ \*\/$//' || echo "unknown"
}

# ── Set starting firmware ─────────────────────────────────────────────
if [[ -n "${RESUME_DIR}" ]]; then
    CANDIDATE_FILE="${RESUME_DIR}/candidate_best.c"
    [[ ! -f "${CANDIDATE_FILE}" ]] && die "No candidate_best.c found in ${RESUME_DIR}"
    cp "${CANDIDATE_FILE}" "${TARGET_FILE}"
    log "[RESUME] Resuming from ${CANDIDATE_FILE}"
elif [[ -f "${BASELINE_FILE}" ]]; then
    cp "${BASELINE_FILE}" "${TARGET_FILE}"
    HEADER=$(read_fw_header "${TARGET_FILE}")
    log "[START] Starting from baseline — ${HEADER}"
else
    die "No baseline file at ${BASELINE_FILE}"
fi

# ── Initialize: build C kernel library ────────────────────────────
log "[INIT] Building C kernel library..."
if [[ ! -f "_build/kernels/libnpukernels.so" ]]; then
    make kernels > /dev/null 2>&1
fi
rm -rf firmware/build_dim*
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
    if [[ ! -f "firmware/build_dim${dim}/${MAKE_TARGET:-bert}.elf" ]]; then
        log "[BUILD] FAILED for seq=${SL}"
        return 1
    fi
    log "[BUILD] seq=${SL} OK"
    return 0
}

restore_baseline() {
    if ! diff -q "${BASELINE_FILE}" "${TARGET_FILE}" > /dev/null 2>&1; then
        cp "${BASELINE_FILE}" "${TARGET_FILE}"
        log "[BASELINE] Restored from ${BASELINE_FILE}"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# PROBE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

probe() {
    local SL=$1
    local dim=${DIM} hidden=${HIDDEN} nh=${NUM_HEAD}
    build_firmware ${SL} || return 1

    local elf_path="firmware/build_dim${dim}/${MAKE_TARGET:-bert}.elf"

    # Generate DAG + cluster graphs (seq=1 for visual clarity)
    if [[ "${SL}" == "${SL6}" ]]; then
        log "[PROBE] Generating DAG and cluster graphs..."
        python3 jimu-dse/scripts/visualize_graph.py --phase micro \
            --dim ${dim} --hidden ${hidden} --seq-len 1 --num-head ${nh} \
            -o "${RESULTS}/dag_agent" --no-render > /dev/null 2>&1
        python3 jimu-dse/scripts/visualize_graph.py --phase cluster \
            --dim ${dim} --hidden ${hidden} --seq-len 1 --num-head ${nh} \
            -o "${RESULTS}/dag_agent" > /dev/null 2>&1
    fi

    python3 -c "
import sys, json
sys.path.insert(0, '.')
from emulator.npu_device_mini import NpuDeviceMini, MEM_DRAM
from emulator.npu_event_trace import EventTracer
from emulator.trace_recorder import TraceRecorder
from emulator.npu_micro_op_dag import (
    collapse_to_micro_ops, build_micro_op_dag,
    extract_clusters, clusters_to_text,
    check_dag_connectivity
)
from iss.mini_rv64 import MiniRV64
import numpy as np

dim=${dim}; h=${hidden}; sl=${SL}; nh=${nh}
npu = NpuDeviceMini(native_dim=dim)
npu.set_hidden_size(h); npu.set_seq_len(sl)
npu._vrf[MEM_DRAM][0:h] = np.zeros(h, dtype=np.float32)
tracer = EventTracer(npu)
rec = TraceRecorder(npu)
cpu = MiniRV64()
cpu.set_mmio_device(rec)
cpu.load_elf('${elf_path}')
cpu.run(cycles=300000)
ds = npu.get_dram_stats()
t_el = ds.get('vec_rd_elements',0)+ds.get('vec_wr_elements',0)+ds.get('mat_rd_elements',0)+ds.get('mat_wr_elements',0)
t_b = t_el * 4

# MV_MUL count
mv_mul_count = sum(1 for e in tracer.events if ((e['raw'] if isinstance(e, dict) else e.inst) >> 24) & 0xFF in (7, 27))

# DAG clusters
micro_ops = collapse_to_micro_ops(tracer.events)
nodes, edges = build_micro_op_dag(micro_ops)
clusters = extract_clusters(nodes, edges, dim=dim, hidden_size=h, seq_len=sl)
cluster_lines = []
for i,c in enumerate(clusters):
    ai = c.compute_flops / max(c.dram_load_bytes + c.dram_store_bytes, 1)
    cluster_lines.append(f'  [{i:2d}] {c.label:30s} Load={c.dram_load_bytes:3d}B Store={c.dram_store_bytes:3d}B FLOPs={c.compute_flops:3d} AI={ai:.1f}')

# Compute tile structure metrics
num_tiles = h // dim
mv_mul_per_proj = num_tiles * num_tiles  # MV_MULs per projection (tile loop)
num_projections = 6  # Q, K, V, SO, Wi, Wo
total_proj_mv_mul = mv_mul_per_proj * num_projections * sl  # per position

result = {
    'total_bytes': t_b,
    'dram_stats': dict(ds),
    'instr_count': len(rec.inst_trace),
    'mv_mul_count': mv_mul_count,
    'mat_rd_ops': ds.get('mat_rd_ops', 0),
    'clusters': cluster_lines,
    'num_micro_ops': len(nodes),
    'tile_structure': {
        'num_tiles': num_tiles,
        'mv_mul_per_projection': mv_mul_per_proj,
        'num_projections': num_projections,
        'total_projection_mv_mul': total_proj_mv_mul,
        'heads_per_tile': dim // (h // nh) if nh > 0 else 1,
    },
}
print(json.dumps(result))
tracer.unpatch()
" 2>/dev/null > "${RESULTS}/p${SL}_probe.json"

    if [[ -s "${RESULTS}/p${SL}_probe.json" ]]; then
        python3 -c "import json; print(json.load(open('${RESULTS}/p${SL}_probe.json')).get('total_bytes',0))"
    else
        echo 0
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP — always incremental within a run
# ═══════════════════════════════════════════════════════════════════════════

for ((iter=1; iter<=JIMU_MAX_ITER; iter++)); do
    log "--- Iteration ${iter} ---"

    # ---- PHASE 1: PROBE --------------------------------------------------
    log "[PROBE] seq=${SL2}..."
    B2=$(probe ${SL2})
    log "[PROBE] seq=${SL6}..."
    B6=$(probe ${SL6})

    # Also extract tile info for display
    PROBE_TILES=$(python3 -c "
import json
d = json.load(open('${RESULTS}/p${SL6}_probe.json'))
ts = d.get('tile_structure', {})
n = ts.get('num_tiles', 0)
mvpp = ts.get('mv_mul_per_projection', 0)
mv = d.get('mv_mul_count', 0)
mrd = d.get('mat_rd_ops', 0)
print(f'MV={mv} MRD={mrd} tile={n}x{n} MV/proj={mvpp}')
" 2>/dev/null)
    log "  seq=${SL2}: ${B2}B  seq=${SL6}: ${B6}B (${PROBE_TILES})"
    if [[ -z "${BASELINE_VALUE}" ]] && [[ "${B6}" != "0" ]]; then
        BASELINE_VALUE="${B6}"
        BEST_VALUE="${B6}"
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
mvpp = ts.get('mv_mul_per_projection', 0)
print(f'DRAM={dram}B MV_MUL={mv} M_RD={mrd} tiles={tiles}x{tiles} MV/proj={mvpp}')
" 2>/dev/null)

    # ---- PHASE 4: AGENT --------------------------------------------------
    PROMPT="${RESULTS}/prompt_${iter}.txt"
    CLUSTER_TXT=$(python3 -c "
import json
d = json.load(open('${RESULTS}/p${SL6}_probe.json'))
for l in d.get('clusters', []):
    print(l)
" 2>/dev/null)

    FW_HEADER=$(read_fw_header "${TARGET_FILE}")
    {
        echo "You are optimizing NPU firmware at ${TARGET_FILE}."
        echo ""
        echo "Goal: ${GOAL} (dim=${DIM}, hidden=${HIDDEN}, num_head=${NUM_HEAD})"
        echo "Current header: ${FW_HEADER}"
        echo "Primary metric: ${PRIMARY_METRIC}"
        echo "MV_MUL count: $(python3 -c "import json; print(json.load(open('${RESULTS}/p${SL6}_probe.json')).get('mv_mul_count',0))" 2>/dev/null)"
        echo "M_RD_DRAM ops (weight tile loads): $(python3 -c "import json; print(json.load(open('${RESULTS}/p${SL6}_probe.json')).get('mat_rd_ops',0))" 2>/dev/null)"
        echo "Current DRAM: seq=${SL2}=${B2}B, seq=${SL6}=${B6}B (ratio ${RATIO})"
        echo ""
        echo "=== DRAM Cluster Analysis (seq=${SL6}) ==="
        echo "${CLUSTER_TXT}"
        echo ""
        # Tile structure analysis from probe data
        TILE_INFO=$(python3 -c "
import json
d = json.load(open('${RESULTS}/p${SL6}_probe.json'))
ts = d.get('tile_structure', {})
print(f'num_tiles={ts.get(\"num_tiles\",0)}, MV_MUL per projection={ts.get(\"mv_mul_per_projection\",0)}, heads_per_tile={ts.get(\"heads_per_tile\",0)}')
print(f'Each projection loops over {ts.get(\"num_tiles\",0)}x{ts.get(\"num_tiles\",0)} tiles = {ts.get(\"mv_mul_per_projection\",0)} MV_MUL calls')
print(f'6 projections = {ts.get(\"total_projection_mv_mul\",0)} MV_MUL total')
" 2>/dev/null)
        echo "=== Tile Structure Analysis ==="
        echo "${TILE_INFO}"
        echo ""
        echo "=== Skills to Apply ==="
        skill_num=1
        for sk in ${SKILLS}; do
            echo "${skill_num}. Read jimu-dse/results/dag_agent/micro_op_dag.txt"
            echo "   and jimu-dse/docs/skills/isa/${sk}.md to apply the ${sk} transformation."
            skill_num=$((skill_num + 1))
        done
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
            echo "- Keep all existing functions intact — apply VRF cache transformations as instructed."
        else
            echo "- Follow the skill instructions exactly as provided."
        fi
        echo "=== Constraints ==="
        echo "- Only modify ${TARGET_FILE}"
        echo "- Do NOT modify the emulator or any other file"
        echo "- File must compile: gcc for RISC-V (NATIVE_DIM=${DIM})"
        echo ""
        echo "=== Self-Verify ==="
        echo "${TEST_VERIFY_CMD}"
    } > "${PROMPT}"

    log "[AGENT] Prompt: $(wc -c < ${PROMPT}) bytes"
    CANDIDATE="${RESULTS}/candidate_${iter}.c"

    if [[ "${AGENT}" == "opencode" ]]; then
        if command -v opencode &>/dev/null; then
            log "[AGENT] Invoking opencode (model: ${OPENCODE_MODEL})..."
            timeout "${JIMU_AGENT_TIMEOUT}" opencode run --model "${OPENCODE_MODEL}" \
                -f "${RESULTS}/dag_agent/micro_op_dag.txt" \
                -f "${RESULTS}/p${SL6}_probe.json" \
                -f "${TARGET_FILE}" \
                --dangerously-skip-permissions \
                "$(cat ${PROMPT})" 2>&1 | tee -a "${RESULTS}/opencode_output.log" | tail -20 || true
            [[ -f "${REPO_ROOT}/${TARGET_FILE}" ]] && \
                cp "${REPO_ROOT}/${TARGET_FILE}" "${CANDIDATE}"
        else
            log "[AGENT] opencode not available — using unmodified firmware"
            cp "${TARGET_FILE}" "${CANDIDATE}"
        fi
    elif command -v pi &>/dev/null; then
        log "[AGENT] Invoking pi (timeout: ${JIMU_AGENT_TIMEOUT}s)..."
        # Pi only supports one skill via --skill right now, pass the first one
        FIRST_SKILL=$(echo ${SKILLS} | awk '{print $1}')
        pi --skill "${REPO_ROOT}/jimu-dse/docs/skills/isa/${FIRST_SKILL}.md" \
           -p "$(cat ${PROMPT})" 2>/dev/null || true
        [[ -f "${REPO_ROOT}/${TARGET_FILE}" ]] && \
            cp "${REPO_ROOT}/${TARGET_FILE}" "${CANDIDATE}"
        [[ ! -s "${CANDIDATE}" ]] && pi -p "$(cat ${PROMPT})" 2>/dev/null | \
            sed -n '/^```/,/^```/{/^```/d;p}' > "${CANDIDATE}"
        [[ ! -s "${CANDIDATE}" ]] && cp "${TARGET_FILE}" "${CANDIDATE}"
    else
        log "[AGENT] No agent available — using unmodified firmware"
        cp "${TARGET_FILE}" "${CANDIDATE}"
    fi

    [[ ! -f "${CANDIDATE}" ]] && log "[AGENT] No candidate" && continue
    log "[AGENT] Candidate: $(wc -l < ${CANDIDATE}) lines"

    # ---- Check: agent actually changed anything? ----
    if diff -q "${CANDIDATE}" "${BASELINE_FILE}" > /dev/null 2>&1; then
        log "[AGENT] Candidate unchanged from baseline — agent made no changes"
        continue
    fi

    # ---- PHASE 5: VALIDATE -----------------------------------------------
    log "[VALIDATE] Running emulator pipeline..."
    cp "${CANDIDATE}" "${TARGET_FILE}"
    build_firmware ${SL6} > /dev/null 2>&1 || true

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
" 2>/dev/null > "${RESULTS}/val_${iter}.json" || true

    if [[ -s "${RESULTS}/val_${iter}.json" ]]; then
        B6_NEW=$(python3 -c "import json; print(json.load(open('${RESULTS}/val_${iter}.json')).get('total_bytes',0))" 2>/dev/null || echo 0)
        SAVED=$((${BASELINE_VALUE:-0} - B6_NEW))
        log "[VALIDATE] DRAM: ${B6_NEW}B (saved ${SAVED}B vs run-start ${BASELINE_VALUE:-0}B)"
    else
        B6_NEW=0; SAVED=0
        log "[VALIDATE] Could not validate"
    fi

    # Save diff against baseline file
    diff "${BASELINE_FILE}" "${CANDIDATE}" > "${RESULTS}/diff_${iter}.patch" 2>/dev/null || true
    log "[SAVE] Diff saved ($(wc -l < "${RESULTS}/diff_${iter}.patch" 2>/dev/null || echo 0) lines)"

    # ---- DAG: generate updated graph visualization (seq=1) ----
    log "[DAG] Generating micro-op DAG and cluster graphs..."
    python3 jimu-dse/scripts/visualize_graph.py --phase all \
        --dim ${DIM} --hidden ${HIDDEN} --seq-len 1 --num-head ${NUM_HEAD} \
        -o "${RESULTS}/dag_iter${iter}" > /dev/null 2>&1
    log "[DAG] DAG graphs saved to ${RESULTS}/dag_iter${iter}"

    # ---- PHASE 6: DEPLOY ------------------------------------------------
    if [[ ${SAVED} -gt 0 ]]; then
        log "[DEPLOY] Candidate iter${iter}: ${B6_NEW}B (saved ${SAVED}B)"
        [[ ${B6_NEW} -lt ${BEST_VALUE} ]] && BEST_VALUE=${B6_NEW}
    fi
    # ---- PHASE 7: CONVERGENCE (after AGENT) ------------------------------
    if [[ "${PRIMARY_METRIC}" == "test_pass" ]]; then
        log "[CONVERGE] Running dim${DIM}-h${HIDDEN} tests..."
        TEST_OUT=$(eval "${TEST_CONVERGE_CMD}")
        TEST_OK="$?"
        echo "${TEST_OUT}"
        if [[ "${TEST_OK}" == "0" ]]; then
            log "[CONVERGE] dim${DIM}-h${HIDDEN} tests PASS (${PROBE_DATA})"
            BEST_VALUE=1
            log "Converged — all tests pass"
            break
        else
            log "[CONVERGE] dim${DIM}-h${HIDDEN} tests FAIL (${PROBE_DATA})"
        fi
    elif [[ ${SAVED} -gt 0 || ${BASELINE_VALUE:-0} -eq 0 ]]; then
        # Standard DRAM-based convergence
        IMPROV=$(python3 -c "p=((${BASELINE_VALUE:-1}-${B6})*100.0/max(${BASELINE_VALUE:-1},1)); print(f'{p:.1f}')" 2>/dev/null || echo "0")
        log "[CONVERGE] ${B6}B vs run-start ${BASELINE_VALUE:-0}B = ${IMPROV}% (${PROBE_DATA})"
        if [[ ${iter} -gt 1 ]]; then
            CONVERGED=$(python3 -c "p=((${BASELINE_VALUE:-1}-${B6})*100.0/max(${BASELINE_VALUE:-1},1)); print('1' if p < ${JIMU_THRESHOLD} else '0')" 2>/dev/null || echo "0")
            if [[ "${CONVERGED}" == "1" ]]; then
                log "Converged (${IMPROV}% < ${JIMU_THRESHOLD}%)"
                break
            fi
        fi
    else
        log "[CONVERGE] Skipping"
    fi
done

# ── Save best candidate for --resume ────────────────────────────────────
if [[ -n "${BEST_VALUE}" ]] && [[ -f "${TARGET_FILE}" ]]; then
    cp "${TARGET_FILE}" "${RESULTS}/candidate_best.c"
    log "[SAVE] Best candidate (${BEST_VALUE}B) saved to ${RESULTS}/candidate_best.c"
fi

# ── Final restore to baseline ──────────────────────────────────────────
restore_baseline

log ""
log "===== Done ====="
log "Goal:       ${GOAL}"
log "Agent:      ${AGENT}"
log "Results:    ${RESULTS}"
if [[ "${PRIMARY_METRIC}" == "test_pass" ]]; then
    TEST_RESULT="FAIL"
    [[ "${BEST_VALUE}" == "1" ]] && TEST_RESULT="PASS"
    log "Baseline:   ${BASELINE_FILE}"
    log "Target:     dim${DIM}-h${HIDDEN} tests pass"
    log "Result:     ${TEST_RESULT}"
else
    log "Baseline:   ${BASELINE_VALUE}B"
    log "Best:       ${BEST_VALUE}B"
    if [[ -n "${BASELINE_VALUE}" ]] && [[ -n "${BEST_VALUE}" ]] && [[ ${BEST_VALUE} -lt ${BASELINE_VALUE} ]]; then
        SAVED=$((BASELINE_VALUE - BEST_VALUE))
        PCT=$(python3 -c "print(f'{(${SAVED}*100.0/${BASELINE_VALUE}):.1f}')" 2>/dev/null || echo "0")
        log "Improvement: ${SAVED}B (${PCT}%)"
    fi
fi
log "To resume:  ./jimu-dse/scripts/npu_closed_loop.sh --goal ${GOAL} --resume ${RESULTS}"
