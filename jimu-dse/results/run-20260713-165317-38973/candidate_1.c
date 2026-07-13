/* NPU — Trained 140p Qwen3 Forward Pass Firmware
 *
 * Architecture: 1L Qwen3 decoder, d=4, 1h, hd=4, ff=4, RoPE θ=3,
 *               SwiGLU, RMSNorm, tied K=V, O=Q^T, LM head=embed.T.
 *
 * ISS pre-fills (before firmware runs):
 *   - Embedding at S_X
 *   - Q (norm+RoPE applied) at S_Q
 *   - K (norm+RoPE applied) at S_K  
 *   - V (=K) at S_V
 *   - Causal mask table at S_MASK_TABLE
 *   - Transposed V tiles at S_VTT
 *   - Scale in SRF[6]
 *
 *   - norm2(attn_res) at S_H2  (ISS precomputes? No — needs attn_res)
 *     Actually ISS writes to S_H2/S_GATE/S_UP AFTER firmware, then runs phase2.
 *
 * Firmware does (single entry, reads phase from DRAM):
 *   Phase 1 (phase==0): Tiled attention → CTX → O=Q^T → residual → S_ATTN_RES
 *   Phase 2 (phase==1): SiLU(gate) → silu*up → W_down → residual → LAST_H → FW_LAST_H
 *
 * ISS reads FW_LAST_H, computes norm_final + embedding^T → logits.
 *
 * Phase selection: firmware reads a phase flag from a magic DRAM address
 * (S_FLAGS = 0x1F00). ISS writes 0 or 1 before booting the firmware.
 */

#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"
#include "npu_driver.h"

#define MODEL_DIM  4
#define MAX_SEQ    35
#define HEAD_DIM   4
#define FF_DIM     4

#define ADDR_W_Q           0x400
#define ADDR_W_KV          0x500
#define ADDR_W_Q_T         0xD00
#define ADDR_W_DOWN        0xA00

/* Main scratch */
#define SCR_BASE      0x2000
#define S_X            SCR_BASE
#define S_Q           (SCR_BASE + MAX_SEQ * MODEL_DIM)
#define S_K           (S_Q + MAX_SEQ * MODEL_DIM)
#define S_V           (S_K + MAX_SEQ * MODEL_DIM)
#define S_CTX         (S_V + MAX_SEQ * MODEL_DIM)
#define S_ATTN_OUT    (S_CTX + MAX_SEQ * MODEL_DIM)
#define S_ATTN_RES    (S_ATTN_OUT + MAX_SEQ * MODEL_DIM)
#define S_SCORE       (S_ATTN_RES + MAX_SEQ * MODEL_DIM)
#define S_PROB        (S_SCORE + MAX_SEQ)
#define S_TEMP        (S_PROB + MAX_SEQ)
#define S_MASK_TABLE  (S_TEMP + MAX_SEQ)
/* V^T tiles after mask_table (address computed at runtime) */

/* Phase 2 scratch (ISS fills this after Phase 1) */
#define S_BASE2      0x3000
#define S_H2          S_BASE2
#define S_GATE        (S_H2 + MAX_SEQ * MODEL_DIM)
#define S_UP          (S_GATE + MAX_SEQ * MODEL_DIM)
#define S_FFN_RES     (S_UP + MAX_SEQ * MODEL_DIM)
#define S_LAST_H      (S_FFN_RES + MAX_SEQ * MODEL_DIM)
#define S_TEMP2       (S_LAST_H + MAX_SEQ * MODEL_DIM)

#define FW_LAST_H     0x4000
#define S_FLAGS       0x1F00  /* DRAM scratch: 0=phase1, 1=phase2 */

/* VRF cache offsets (MFU_INITIAL_VRF = bank 6, 4096 elems) */
#define VRF_SCORE_OFF  0x000
#define VRF_PROB_OFF   0x100
#define VRF_TEMP2_OFF  0x200
#define VRF_FFNR_OFF   0x400

#define SEND_SI(op, opd0, opd1) npu_send_inst(SI(op, opd0, opd1))
#define SEND_LO(op, adr)        npu_send_inst(LO(op, adr))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline void mvm(uint32_t mat, uint32_t vec, uint32_t sink)
{
    SEND_LO(OP_M_RD_DRAM, mat);
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);
    SEND_LO(OP_V_RD_DRAM, vec);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_MV_MUL, 0, 0);
    SEND_SI(OP_V_WR, sink, 0);
}

static inline void mvm_vrf(uint32_t mat, uint32_t vrf_bank, uint32_t vrf_off, uint32_t sink)
{
    SEND_LO(OP_M_RD_DRAM, mat);
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);
    SEND_SI(OP_V_RD, vrf_bank, vrf_off);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_MV_MUL, 0, 0);
    SEND_SI(OP_V_WR, sink, 0);
}

/* Read phase flag from DRAM via MMIO */
static inline uint32_t read_phase_flag(void)
{
    return npu_read_reg(NPU_DRAM_BASE + S_FLAGS * 4);
}

void adder_phase1(void)
{
    uint32_t sl = npu_read_reg(NPU_REG_SEQ_LEN);
    if (sl == 0) sl = 1; if (sl > MAX_SEQ) sl = MAX_SEQ;
    uint32_t nt = (sl + 3) / 4;
    uint32_t mt = S_MASK_TABLE;
    uint32_t svt = mt + MAX_SEQ * nt * MODEL_DIM;
    uint32_t i, q, tc, p, d;

    SEND_SI(OP_S_WR, REG_TILE_ROWS, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS, 1);

    for (i = 0; i < sl; i++) {
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        SEND_LO(OP_V_WR_DRAM, S_CTX + i * MODEL_DIM);
    }

    for (q = 0; q < sl; q++) {
        SEND_SI(OP_V_RD, MEM_FILL, 0xFC00);
        SEND_SI(OP_V_WR, MEM_SPU_ADD_REDUCE, 0);

        for (tc = 0; tc < nt; tc++) {
            uint32_t base = tc * 4;
            uint32_t vld = MIN(4, sl - base);

            for (p = 0; p < 4; p++) {
                if (p < vld)
                    SEND_LO(OP_V_RD_DRAM, S_K + (base + p) * MODEL_DIM);
                else
                    SEND_SI(OP_V_RD, MEM_FILL, 0);
                SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
            }
            SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

            SEND_LO(OP_V_RD_DRAM, S_Q + q * MODEL_DIM);
            SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_MV_MUL, 0, 0);

            SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_V_RD, MEM_SPU_BROADCAST, 6);
            SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_VV_MUL, 0, 0);

            SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
            SEND_LO(OP_V_RD_DRAM, mt + (q * nt + tc) * MODEL_DIM);
            SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_VV_ADD, 0, 0);

            SEND_SI(OP_V_WR, 6, VRF_SCORE_OFF + base);
            SEND_SI(OP_V_WR, MEM_SPU_MAX_REDUCE, 0);
        }

        // Zero SRF[1] via MMIO (SPU_ADD_REDUCE is cumulative)
        npu_write_reg(NPU_SRF_BASE + 1 * 4, 0);

        for (tc = 0; tc < nt; tc++) {
            uint32_t base = tc * 4;
            SEND_SI(OP_V_RD, 6, VRF_SCORE_OFF + base);
            SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_V_RD, MEM_SPU_BROADCAST, 0);
            SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_VV_B_SUB_A, 0, 0);
            SEND_SI(OP_V_EXP, 0, 0);
            SEND_SI(OP_V_WR, MEM_SPU_ADD_REDUCE, 1);
        }

        SEND_SI(OP_S_RECIP, 2, 1);

        for (tc = 0; tc < nt; tc++) {
            uint32_t base = tc * 4;

            SEND_SI(OP_V_RD, 6, VRF_SCORE_OFF + base);
            SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_V_RD, MEM_SPU_BROADCAST, 0);
            SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_VV_B_SUB_A, 0, 0);
            SEND_SI(OP_V_EXP, 0, 0);
            SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_V_RD, MEM_SPU_BROADCAST, 2);
            SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_VV_MUL, 0, 0);
            SEND_SI(OP_V_WR, 6, VRF_PROB_OFF + base);

            for (d = 0; d < MODEL_DIM; d++) {
                SEND_LO(OP_V_RD_DRAM, svt + tc * MODEL_DIM * MODEL_DIM + d * MODEL_DIM);
                SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
            }
            SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

            SEND_SI(OP_V_RD, 6, VRF_PROB_OFF + base);
            SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_MV_MUL, 0, 0);

            SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
            SEND_LO(OP_V_RD_DRAM, S_CTX + q * MODEL_DIM);
            SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
            SEND_SI(OP_VV_ADD, 0, 0);
            SEND_LO(OP_V_WR_DRAM, S_CTX + q * MODEL_DIM);
        }
    }

    for (i = 0; i < sl; i++) {
        mvm(ADDR_W_Q_T, S_CTX + i * MODEL_DIM, MEM_MULTIPLY_VRF);
        SEND_LO(OP_V_WR_DRAM, S_ATTN_OUT + i * MODEL_DIM);
        SEND_SI(OP_V_RD, MEM_FILL, 0);
    }

    for (i = 0; i < sl; i++) {
        SEND_LO(OP_V_RD_DRAM, S_X + i * MODEL_DIM);
        SEND_LO(OP_V_RD_DRAM, S_ATTN_OUT + i * MODEL_DIM);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_LO(OP_V_WR_DRAM, S_ATTN_RES + i * MODEL_DIM);
    }
    npu_wait_done();
}

void adder_phase2(void)
{
    uint32_t sl = npu_read_reg(NPU_REG_SEQ_LEN);
    if (sl == 0) sl = 1; if (sl > MAX_SEQ) sl = MAX_SEQ;
    uint32_t i;

    for (i = 0; i < sl; i++) {
        SEND_LO(OP_V_RD_DRAM, S_GATE + i * MODEL_DIM);
        SEND_SI(OP_V_SIGM, 0, 0);
        SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
        SEND_LO(OP_V_RD_DRAM, S_GATE + i * MODEL_DIM);
        SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
        SEND_SI(OP_VV_MUL, 0, 0);

        SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
        SEND_LO(OP_V_RD_DRAM, S_UP + i * MODEL_DIM);
        SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
        SEND_SI(OP_VV_MUL, 0, 0);
        SEND_SI(OP_V_WR, 6, VRF_TEMP2_OFF + i * MODEL_DIM);

        mvm_vrf(ADDR_W_DOWN, 6, VRF_TEMP2_OFF + i * MODEL_DIM, MEM_MULTIPLY_VRF);
        SEND_SI(OP_V_WR, 6, VRF_FFNR_OFF + i * MODEL_DIM);

        SEND_LO(OP_V_RD_DRAM, S_ATTN_RES + i * MODEL_DIM);
        SEND_SI(OP_V_RD, 6, VRF_FFNR_OFF + i * MODEL_DIM);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_LO(OP_V_WR_DRAM, S_LAST_H + i * MODEL_DIM);
    }

    uint32_t last = S_LAST_H + (sl - 1) * MODEL_DIM;
    SEND_LO(OP_V_RD_DRAM, last);
    SEND_LO(OP_V_WR_DRAM, FW_LAST_H);
    npu_wait_done();
}

void main(void)
{
    while (npu_read_reg(NPU_STATUS) & NPU_STATUS_BUSY);

    uint32_t phase = read_phase_flag();

    if (phase == 1) {
        adder_phase2();
    } else {
        adder_phase1();
    }

    npu_set_done();
    while (1);
}
