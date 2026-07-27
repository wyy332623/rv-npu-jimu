/* ── 02_multi_chain.c ─────────────────────────────────────────────
 *
 * Multi-chain example: independent operations split across multiple
 * INST_ISSUE groups.  Demonstrates explicit vs implicit operands
 * across compute patterns.
 *
 * Chain 1 (MVM):     Load weight (MRF, explicit), load vector
 *                     (pipeline, implicit), MV_MUL, store to VRF
 * Chain 2 (bias):    Load W×X from VRF, load bias (pipeline →
 *                     vpipe_a), VV_ADD (vpipe_a + pipeline), store
 *
 * Also includes silu_mvm_residual_chain(): one position's FFN from
 * adder_140p.c phase2 as a single chain, showing SiLU via V_SIGM +
 * VV_MUL, then MVM, then residual VV_ADD — all through the pipeline.
 *
 * Reference: adder_140p.c adder_phase1() — the tiled attention
 *            loop builds K.T MRF tiles across many positions
 *            within one chain before issuing.
 *
 * Reference: bert_layer.c — the comment at line 18-26 describes
 *            SMC with 3+ concurrent INST_ISSUE chains.
 *
 * Key points:
 *   - MV_MUL takes MRF (explicit) and pipeline (implicit).
 *   - VV_ADD takes vpipe_a (implicit, saved by V_RD) + pipeline
 *     (implicit) — both from the pipe, neither from instruction word.
 *   - V_WR / V_WR_DRAM broadcast: write to target AND keep
 *     the value in the pipeline for the next instruction.
 *   - INST_ISSUE discards pipeline.  Next chain reloads from VRF.
 */

#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"
#include "npu_driver.h"

#define NATIVE_DIM      4
#define MAT_SIZE        (NATIVE_DIM * NATIVE_DIM)

#define WEIGHT_ADDR     0x400    /* W matrix in DRAM */
#define VECTOR_ADDR     0x2000   /* X vector in DRAM  */
#define BIAS_ADDR       0x500    /* bias vector in DRAM */
#define RESULT_ADDR     0x2100   /* output location   */

#define SEND_SI(op, opd0, opd1) npu_send_inst(SI(op, opd0, opd1))
#define SEND_LO(op, adr)        npu_send_inst(LO(op, adr))

void multi_chain_example(void)
{
    /* ── Configuration (scalar, not part of any chain) ─────────── */
    SEND_SI(OP_S_WR, REG_TILE_ROWS, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS, 1);

    /* ═════════════════════════════════════════════════════════════
     * Chain 1: MVM (W × X → MULTIPLY_VRF)
     *
     *   instr │ op          │ MRF state │ pipeline │ notes
     *   ──────┼─────────────┼───────────┼──────────┼────────────────
     *    [1]  │ M_RD_DRAM/W │ = W       │ (keep)   │ explicit
     *    [2]  │ M_WR        │ = W       │ (keep)   │ commit (no-op)
     *    [3]  │ V_RD_DRAM/X │ = W       │ = X      │ implicit
     *    [4]  │ MV_MUL      │ = W       │ = W×X    │ MRF × pipe
     *    [5]  │ V_WR/MPV    │ = W       │ = W×X    │ store (kept)
     *    [6]  │ INST_ISSUE  │ (clear)   │ (clear)  │ commit + poll
     *
     * Note: no V_WR(IVRF)/V_RD(IVRF) round-trip — the vector X
     * stays in the pipeline from [3] straight into MV_MUL at [4].
     * ═════════════════════════════════════════════════════════════ */
    SEND_LO(OP_M_RD_DRAM, WEIGHT_ADDR);            /* [1] explicit: MRF = W */
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);            /* [2] commit */
    SEND_LO(OP_V_RD_DRAM, VECTOR_ADDR);             /* [3] implicit: pipe = X */
    SEND_SI(OP_MV_MUL, 0, 0);                       /* [4] MRF × pipe → pipe */
    SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);          /* [5] pipe → VRF (kept) */
    SEND_SI(OP_INST_ISSUE, 0, 0);                   /* [6] commit */
    npu_wait_chain();

    /* ═════════════════════════════════════════════════════════════
     * Chain 2: Bias add (W×X + bias → DRAM)
     *
     *   instr │ op           │ pipeline │ vpipe_a  │ notes
     *   ──────┼──────────────┼──────────┼──────────┼────────────────
     *    [7]  │ V_RD/MPV     │ = W×X    │ (none)   │ load from VRF
     *    [8]  │ V_RD_DRAM/b  │ = bias   │ = W×X    │ implicit pair
     *    [9]  │ VV_ADD       │ = W×X+b  │ (clear)  │ vpipe_a + pipe
     *   [10]  │ V_WR_DRAM    │ = W×X+b  │ (clear)  │ store (kept)
     *   [11]  │ INST_ISSUE   │ (clear)  │ (clear)  │ commit + poll
     *
     * VV_ADD's operands are both implicit: vpipe_a (saved by [8])
     * and pipeline (set by [8]).  Neither is in the instruction word.
     * ═════════════════════════════════════════════════════════════ */
    SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);          /* [7] load W×X from VRF */
    SEND_LO(OP_V_RD_DRAM, BIAS_ADDR);               /* [8] load bias (W×X → vpipe_a) */
    SEND_SI(OP_VV_ADD, 0, 0);                       /* [9] vpipe_a + pipe */
    SEND_LO(OP_V_WR_DRAM, RESULT_ADDR);             /* [10] store to DRAM */
    SEND_SI(OP_INST_ISSUE, 0, 0);                   /* [11] commit */
    npu_wait_chain();
}

/* ── Variant: SiLU × up → W_down → residual (from adder_140p.c) ─
 *
 * One position of adder_phase2()'s FFN as a single chain group.
 * Demonstrates: scalar activation (V_SIGM) feeding VV_MUL, then
 * MVM, then residual VV_ADD — all through the implicit pipeline.
 *
 *   V_RD_DRAM/gate → pipeline
 *   V_SIGM         → sigmoid(gate)
 *   V_WR/MPV       save sigmoid       (pipeline kept for reload)
 *   V_RD_DRAM/gate → gate              (saves sigmoid → vpipe_a)
 *   V_RD/MPV       → sigmoid          (saves gate → vpipe_a)
 *   VV_MUL         gate × sigmoid     = SiLU(gate)
 *   V_WR/MPV       save SiLU          (kept)
 *   V_RD_DRAM/up   → up               (saves SiLU → vpipe_a)
 *   V_RD/MPV       → SiLU             (saves up → vpipe_a)
 *   VV_MUL         SiLU × up
 *   V_WR/IVRF      save SiLU×up
 *   M_RD_DRAM/Wdn  → MRF             (explicit)
 *   V_RD/IVRF      → SiLU×up         (implicit, into pipeline)
 *   MV_MUL         Wdn × SiLU×up     (MRF explicit + pipe implicit)
 *   V_WR/MPV       save FFN_out      (kept)
 *   V_RD_DRAM/res  → attn_res        (saves FFN_out → vpipe_a)
 *   V_RD/MPV       → FFN_out         (saves attn_res → vpipe_a)
 *   VV_ADD         attn_res + FFN_out
 *   V_WR_DRAM/out  store to DRAM
 *   INST_ISSUE     commit
 *   wait_chain
 */
void silu_mvm_residual_chain(uint32_t base_addr, uint32_t pos)
{
    uint32_t gate_addr = base_addr + pos * NATIVE_DIM;
    uint32_t up_addr   = base_addr + pos * NATIVE_DIM + 0x400;
    uint32_t res_addr  = base_addr + pos * NATIVE_DIM + 0x800;
    uint32_t out_addr  = base_addr + pos * NATIVE_DIM + 0xC00;

    /* ── SiLU: sigmoid(gate) × gate ── */
    SEND_LO(OP_V_RD_DRAM, gate_addr);              /* pipe = gate */
    SEND_SI(OP_V_SIGM, 0, 0);                      /* pipe = sigmoid(gate) */
    SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);         /* save sigmoid (pipe kept) */
    SEND_LO(OP_V_RD_DRAM, gate_addr);               /* pipe = gate  (sigmoid → vpipe_a) */
    SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);         /* pipe = sigmoid (gate → vpipe_a) */
    SEND_SI(OP_VV_MUL, 0, 0);                      /* pipe = gate × sigmoid = SiLU(gate) */

    /* ── Scale by up ── */
    SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);         /* save SiLU (pipe kept) */
    SEND_LO(OP_V_RD_DRAM, up_addr);                 /* pipe = up (SiLU → vpipe_a) */
    SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);         /* pipe = SiLU (up → vpipe_a) */
    SEND_SI(OP_VV_MUL, 0, 0);                      /* pipe = SiLU × up */

    /* ── W_down projection: M_RD_DRAM (explicit) + V_RD (implicit) + MV_MUL ── */
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);      /* save SiLU×up (pipe kept) */
    SEND_LO(OP_M_RD_DRAM, 0xA00);                   /* explicit: MRF = W_down */
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);             /* commit */
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);      /* pipe = SiLU×up (implicit) */
    SEND_SI(OP_MV_MUL, 0, 0);                      /* pipe = W_down × SiLU×up */

    /* ── Residual: attn_res + FFN_out ── */
    SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);         /* save FFN_out (pipe kept) */
    SEND_LO(OP_V_RD_DRAM, res_addr);                /* pipe = attn_res (FFN_out → vpipe_a) */
    SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);         /* pipe = FFN_out (attn_res → vpipe_a) */
    SEND_SI(OP_VV_ADD, 0, 0);                      /* pipe = attn_res + FFN_out */
    SEND_LO(OP_V_WR_DRAM, out_addr);               /* store to DRAM (pipe kept) */

    /* Commit all 15 instructions as one atomic chain */
    SEND_SI(OP_INST_ISSUE, 0, 0);
    npu_wait_chain();
}

void main(void)
{
    while (npu_read_reg(NPU_STATUS) & NPU_STATUS_BUSY);
    multi_chain_example();
    npu_set_done();
    while (1);
}
