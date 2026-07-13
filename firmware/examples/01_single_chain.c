/* ── 01_single_chain.c ─────────────────────────────────────────────
 *
 * Single-chain example: one INST_ISSUE group contains all instructions.
 *
 * MV_MUL has TWO operands:
 *   [explicit] MRF  — loaded by the preceding M_RD_DRAM
 *   [implicit] pipeline — the live vector in the pipe register,
 *              set by the most recent V_RD or V_RD_DRAM
 *
 * The pipeline operand is "implicit" because it is NOT encoded in the
 * MV_MUL instruction word — it is whatever value happens to be in the
 * pipe register from the previous instruction.  This is the essence of
 * chaining: values flow through the pipe without explicit source/dest
 * fields in every instruction.
 *
 * Chain pattern (load-compute-store):
 *   M_RD_DRAM   load weight tile        → MRF  (explicit operand for MV_MUL)
 *   M_WR        no-op (MRF already set)
 *   V_RD_DRAM   load vector from DRAM   → pipeline (implicit operand for MV_MUL)
 *   MV_MUL      MRF × pipeline          → pipeline (consumes both)
 *   V_WR        pipeline → VRF          (keeps pipeline — broadcast)
 *   INST_ISSUE  commit chain            → pipeline discarded, CHAIN_STATUS=0
 *
 * Reference: adder_140p.c's mvm() inline.
 */

#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"
#include "npu_driver.h"

#define NATIVE_DIM      4
#define MAT_SIZE        (NATIVE_DIM * NATIVE_DIM)

#define WEIGHT_ADDR     0x400
#define VECTOR_ADDR     0x2000
#define RESULT_ADDR     0x2100

#define SEND_SI(op, opd0, opd1) npu_send_inst(SI(op, opd0, opd1))
#define SEND_LO(op, adr)        npu_send_inst(LO(op, adr))

void single_chain_example(void)
{
    /* ── Configuration (scalar writes, not part of any chain) ──── */
    SEND_SI(OP_S_WR, REG_TILE_ROWS, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS, 1);

    /* ═════════════════════════════════════════════════════════════
     * Chain: MVM (single chain, 5 instructions + issue)
     *
     *   instr │ op         │ MRF state  │ pipeline  │ description
     *   ──────┼────────────┼────────────┼───────────┼─────────────────
     *    [1]  │ M_RD_DRAM  │ = W        │ (unchanged)│ load weight
     *    [2]  │ M_WR       │ = W        │ (unchanged)│ commit (no-op)
     *    [3]  │ V_RD_DRAM  │ = W        │ = X       │ load vector
     *    [4]  │ MV_MUL     │ = W        │ = W×X     │ explicit: MRF
     *        │            │            │           │ implicit: pipe
     *    [5]  │ V_WR/MPV   │ = W        │ = W×X     │ store (pipe kept)
     *    [6]  │ INST_ISSUE │ (cleared)  │ (cleared) │ commit + poll
     * ═════════════════════════════════════════════════════════════ */

    /* [1] Load weight tile — sets MRF as explicit MV_MUL operand */
    SEND_LO(OP_M_RD_DRAM, WEIGHT_ADDR);

    /* [2] Commit to MRF (M_WR is a no-op in the emulator) */
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

    /* [3] Load input vector — sets pipeline as implicit MV_MUL operand.
     *     V_RD_DRAM saves any previous pipeline to vpipe_a (not used here). */
    SEND_LO(OP_V_RD_DRAM, VECTOR_ADDR);

    /* [4] MV_MUL: MRF (explicit) × pipeline (implicit) → pipeline */
    SEND_SI(OP_MV_MUL, 0, 0);

    /* [5] Store result to VRF.  V_WR broadcasts: writes to VRF AND keeps
     *     the value in the pipeline for any subsequent instructions. */
    SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);

    /* [6] Commit: dispatch [1-5] as one parallel group, discard pipeline */
    SEND_SI(OP_INST_ISSUE, 0, 0);
    npu_wait_chain();

    /* ── Second chain: load result from VRF, write to DRAM ───────
     * Pipeline was discarded by INST_ISSUE, so we reload from VRF.
     */
    SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
    SEND_LO(OP_V_WR_DRAM, RESULT_ADDR);
    SEND_SI(OP_INST_ISSUE, 0, 0);
    npu_wait_chain();
}

void main(void)
{
    while (npu_read_reg(NPU_STATUS) & NPU_STATUS_BUSY);
    single_chain_example();
    npu_set_done();
    while (1);
}
