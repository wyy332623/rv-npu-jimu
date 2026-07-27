/* ── 03_softmax_chain.c ────────────────────────────────────────────
 *
 * Softmax chaining example: demonstrates V_FUNC(SOFTMAX) integrated
 * into a single chain alongside MV_MUL, using the implicit pipeline
 * to pass values between attention stages without DRAM saves.
 *
 * In attention, the computation is:
 *   scores = Q × K.T          (row vector × matrix → row vector)
 *   attn   = softmax(scores)  (row vector, per-position distribution)
 *   out    = attn × V         (row vector × matrix → row vector)
 *
 * The MV_MUL instruction does MRF × pipeline.  So:
 *   Chain 1: MRF = K.T,  pipeline = Q      → scores
 *   Chain 2: MRF = V,    pipeline = attn   → out
 *
 * Key insight: V_FUNC(SOFTMAX) reads the pipeline (scores from MV_MUL),
 * applies softmax, writes the result back to the pipeline — no DRAM
 * save-load roundtrip between scoring and softmax.  The V_WR saves
 * the result to VRF for the next chain after INST_ISSUE clears
 * the pipeline.
 *
 * Pipeline flow for Chain 1 (score + softmax):
 *
 *   instr │ op                │ MRF state │ pipeline       │ notes
 *   ──────┼───────────────────┼───────────┼────────────────┼────────────────
 *    [1]  │ M_RD_DRAM(K.T)    │ = K.T     │ (prev)         │ explicit
 *    [2]  │ M_WR              │ = K.T     │ (prev)         │ commit
 *    [3]  │ V_RD_DRAM(Q)      │ = K.T     │ = Q            │ implicit
 *    [4]  │ MV_MUL            │ = K.T     │ = scores       │ K.T × Q
 *    [5]  │ V_FUNC(SOFTMAX)   │ = K.T     │ = attn         │ softmax(scores)
 *    [6]  │ V_WR/ADDSUB_VRF   │ = K.T     │ = attn         │ save (pipe kept)
 *    [7]  │ INST_ISSUE        │ (clear)   │ (clear)        │ commit
 *
 * Pipeline flow for Chain 2 (context = attn × V):
 *
 *   instr │ op                │ MRF state │ pipeline       │ notes
 *   ──────┼───────────────────┼───────────┼────────────────┼────────────────
 *    [8]  │ M_RD_DRAM(V)      │ = V       │ (prev)         │ explicit
 *    [9]  │ M_WR              │ = V       │ (prev)         │ commit
 *   [10]  │ V_RD/attn         │ = V       │ = attn         │ load from VRF
 *   [11]  │ MV_MUL            │ = V       │ = context      │ V × attn
 *   [12]  │ V_WR/ADDSUB_VRF   │ = V       │ = context      │ save (pipe kept)
 *   [13]  │ INST_ISSUE        │ (clear)   │ (clear)        │ commit
 */

#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"
#include "npu_driver.h"

#define NATIVE_DIM      4

#define KTILE_ADDR      0x600    /* K.T matrix in DRAM (for Q × K.T) */
#define VTILE_ADDR      0x700    /* V  matrix in DRAM (for attn × V) */
#define Q_ADDR          0x2000   /* Q vector in DRAM */
#define CONTEXT_ADDR    0x2100   /* context output */

#define SEND_SI(op, opd0, opd1) npu_send_inst(SI(op, opd0, opd1))
#define SEND_LO(op, adr)        npu_send_inst(LO(op, adr))


void softmax_chain_example(void)
{
    /* ── Configuration ─────────────────────────────────────────── */
    SEND_SI(OP_S_WR, REG_TILE_ROWS, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS, 1);

    /* ═════════════════════════════════════════════════════════════
     * Chain 1: Q × K.T → scores → softmax → attention weights
     *
     *   [1] M_RD_DRAM(K.T)       → MRF
     *   [2] M_WR                   commit
     *   [3] V_RD_DRAM(Q)          → pipeline
     *   [4] MV_MUL                 K.T × Q  = scores  (MRF × pipe)
     *   [5] V_FUNC(SOFTMAX)        softmax(scores) = attn  (pipe → pipe)
     *   [6] V_WR(ADDSUB_VRF_0)    save attn for chain 2 (pipe kept)
     *   [7] INST_ISSUE             commit, pipeline discarded
     * ═════════════════════════════════════════════════════════════ */

    SEND_LO(OP_M_RD_DRAM, KTILE_ADDR);              /* [1] MRF = K.T */
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);             /* [2] commit */
    SEND_LO(OP_V_RD_DRAM, Q_ADDR);                  /* [3] pipe = Q */
    SEND_SI(OP_MV_MUL, 0, 0);                       /* [4] pipe = scores */
    SEND_SI(OP_V_FUNC, SUB_SOFTMAX, 0);             /* [5] pipe = attn (softmax(scores)) */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);          /* [6] save attn (pipe kept) */
    SEND_SI(OP_INST_ISSUE, 0, 0);                   /* [7] commit */
    npu_wait_chain();


    /* ═════════════════════════════════════════════════════════════
     * Chain 2: attn × V → context
     *
     *   [8]  M_RD_DRAM(V)        → MRF
     *   [9]  M_WR                  commit
     *  [10]  V_RD(attn)           → pipeline (from VRF saved by chain 1)
     *  [11]  MV_MUL                V × attn  = context  (MRF × pipe)
     *  [12]  V_WR(ADDSUB_VRF_0)   save context (pipe kept)
     *  [13]  INST_ISSUE            commit
     *
     * Note: MRF = V (not V.T).  Since MV_MUL does MRF × pipeline,
     * and pipeline = attn (row vector), this computes V × attn,
     * which is the correct context vector.
     * ═════════════════════════════════════════════════════════════ */

    SEND_LO(OP_M_RD_DRAM, VTILE_ADDR);              /* [8]  MRF = V */
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);             /* [9]  commit */
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);          /* [10] pipe = attn (from VRF) */
    SEND_SI(OP_MV_MUL, 0, 0);                       /* [11] pipe = context (V × attn) */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);          /* [12] save context */
    SEND_SI(OP_INST_ISSUE, 0, 0);                   /* [13] commit */
    npu_wait_chain();


    /* ═════════════════════════════════════════════════════════════
     * Chain 3: Context → DRAM (final write-out)
     * ═════════════════════════════════════════════════════════════ */
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_LO(OP_V_WR_DRAM, CONTEXT_ADDR);
    SEND_SI(OP_INST_ISSUE, 0, 0);
    npu_wait_chain();
}


/* ═══════════════════════════════════════════════════════════════════
 * Variant: Single-chain attention (Q × K.T → softmax → attn × V)
 *
 * Demonstrates the entire attention computation in ONE chain without
 * intermediate INST_ISSUE.  The pipeline holds:
 *
 *    V_RD_DRAM(Q)      → pipeline(Q)
 *    MV_MUL            → pipeline(K.T × Q = scores)
 *    V_FUNC(SOFTMAX)   → pipeline(softmax(scores) = attn)
 *    V_WR              save attn to VRF (pipe kept, broadcast)
 *    M_RD_DRAM(V)      → MRF (overwrites K.T)
 *    M_WR              commit
 *    V_RD(VRF)         → pipeline(attn) — reload from VRF
 *    MV_MUL            → pipeline(V × attn = context)
 *    V_WR(ADDSUB)      save context
 *    INST_ISSUE        commit
 * ═══════════════════════════════════════════════════════════════════ */
void single_chain_attention(void)
{
    SEND_SI(OP_S_WR, REG_TILE_ROWS, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS, 1);

    /* ── Q × K.T → scores → softmax ── */
    SEND_LO(OP_M_RD_DRAM, KTILE_ADDR);
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);
    SEND_LO(OP_V_RD_DRAM, Q_ADDR);
    SEND_SI(OP_MV_MUL, 0, 0);                       /* pipe = scores */
    SEND_SI(OP_V_FUNC, SUB_SOFTMAX, 0);             /* pipe = attn (softmax(scores)) */

    /* ── Save attn, load V into MRF, reload attn, multiply ── */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);          /* save attn (pipe kept, broadcast) */
    SEND_LO(OP_M_RD_DRAM, VTILE_ADDR);              /* MRF = V (overwrites K.T) */
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);             /* commit */
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);          /* pipe = attn (reload from VRF) */
    SEND_SI(OP_MV_MUL, 0, 0);                       /* pipe = context (V × attn) */

    /* ── Save and commit ── */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
    SEND_SI(OP_INST_ISSUE, 0, 0);
    npu_wait_chain();

    /* ── Final write-out to DRAM ── */
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_LO(OP_V_WR_DRAM, CONTEXT_ADDR);
    SEND_SI(OP_INST_ISSUE, 0, 0);
    npu_wait_chain();
}


void main(void)
{
    while (npu_read_reg(NPU_STATUS) & NPU_STATUS_BUSY);
    softmax_chain_example();
    single_chain_attention();
    npu_set_done();
    while (1);
}
