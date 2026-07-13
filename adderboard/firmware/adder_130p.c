/* NPU — AdderBoard cosminscn_130p Forward Pass Firmware
 *
 * Single-phase firmware using MMIO data exchange:
 *   - S_RECIP: SRF[dst] = 1/SRF[src] for inv_sum
 *   - SRF window: zero SRF[1] between queries (SPU_ADD_REDUCE is cumulative)
 *   - DRAM window: read dot product for rank-1, write mlp_out
 */

#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"
#include "npu_driver.h"

#define MODEL_DIM  4
#define VOCAB_SIZE 10
#define MAX_SEQ    33
#define HEAD_DIM   2

#define ADDR_PE_TABLE     0x000
#define ADDR_C_ATTN       0x092
#define ADDR_C_PROJ       0x0C2
#define ADDR_C_FC         0x0D2
#define ADDR_C_FC_BIAS    0x0E2
#define ADDR_C_PROJ_U     0x0E6
#define ADDR_C_PROJ_V     0x0EA

#define SCR_BASE      0x1000
#define S_X           (SCR_BASE)
#define S_Q           (S_X + MAX_SEQ * MODEL_DIM)
#define S_K           (S_Q + MAX_SEQ * MODEL_DIM)
#define S_V           (S_K + MAX_SEQ * MODEL_DIM)
#define S_CTX         (S_V + MAX_SEQ * MODEL_DIM)
#define S_ATTN_OUT    (S_CTX + MAX_SEQ * MODEL_DIM)
#define S_AFTER_ATTN  (S_ATTN_OUT + MAX_SEQ * MODEL_DIM)
#define S_FC          (S_AFTER_ATTN + MAX_SEQ * MODEL_DIM)
#define S_MLP_OUT     (S_FC + MAX_SEQ * MODEL_DIM)
#define S_AFTER_MLP   (S_MLP_OUT + MAX_SEQ * MODEL_DIM)
#define S_SCORE       (S_AFTER_MLP + MAX_SEQ * MODEL_DIM)
#define S_PROB        (S_SCORE + MAX_SEQ)
#define S_TEMP        (S_PROB + MAX_SEQ)
#define S_MASK_H0     (S_TEMP + 0)
#define S_MASK_H1     (S_TEMP + 4)
#define S_MASK_TABLE  (S_TEMP + MAX_SEQ)

#define FW_LAST_H     0x2000

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

static inline void mask_head(uint32_t off)
{
    SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
    SEND_LO(OP_V_RD_DRAM, (off == 0) ? S_MASK_H0 : S_MASK_H1);
    SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
    SEND_SI(OP_VV_MUL, 0, 0);
}

void adder_forward(void)
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

    // ── Phase 1: Embed + PE ──
    for (i = 0; i < sl; i++) {
        uint32_t xa = S_X + i * MODEL_DIM;
        SEND_LO(OP_V_RD_DRAM, xa);
        SEND_LO(OP_V_RD_DRAM, ADDR_PE_TABLE + i * MODEL_DIM);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_LO(OP_V_WR_DRAM, xa);
    }

    // ── Phase 2: QKV ──
    for (i = 0; i < sl; i++) {
        uint32_t xa = S_X + i * MODEL_DIM;
        mvm(ADDR_C_ATTN, xa, MEM_MULTIPLY_VRF);
        SEND_LO(OP_V_WR_DRAM, S_Q + i * MODEL_DIM);
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        mvm(ADDR_C_ATTN + 4 * MODEL_DIM, xa, MEM_MULTIPLY_VRF);
        SEND_LO(OP_V_WR_DRAM, S_K + i * MODEL_DIM);
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        mvm(ADDR_C_ATTN + 8 * MODEL_DIM, xa, MEM_MULTIPLY_VRF);
        SEND_LO(OP_V_WR_DRAM, S_V + i * MODEL_DIM);
        SEND_SI(OP_V_RD, MEM_FILL, 0);
    }

    // ── Phase 3: Tiled Attention ──
    // Zero CTX accumulators
    for (i = 0; i < sl; i++) {
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        SEND_LO(OP_V_WR_DRAM, S_CTX + i * MODEL_DIM);
    }

    for (q = 0; q < sl; q++) {
        for (uint32_t ho = 0; ho < MODEL_DIM; ho += HEAD_DIM) {

            // ── Pass 1: scores → max ──
            // Init SRF[0] = -inf
            SEND_SI(OP_V_RD, MEM_FILL, 0xFC00);
            SEND_SI(OP_V_WR, MEM_SPU_ADD_REDUCE, 0);

            // Zero SRF[1] via MMIO write (SPU_ADD_REDUCE is cumulative)
            npu_write_reg(NPU_SRF_BASE + 1 * 4, 0);

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
                mask_head(ho);

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

                SEND_LO(OP_V_WR_DRAM, S_SCORE + base);
                SEND_SI(OP_V_WR, MEM_SPU_MAX_REDUCE, 0);
            }

            // ── Pass 2: sum exp(score - max) → SRF[1] ──
            for (tc = 0; tc < nt; tc++) {
                uint32_t base = tc * 4;
                SEND_LO(OP_V_RD_DRAM, S_SCORE + base);
                SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_V_RD, MEM_SPU_BROADCAST, 0);
                SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_VV_B_SUB_A, 0, 0);
                SEND_SI(OP_V_EXP, 0, 0);
                SEND_SI(OP_V_WR, MEM_SPU_ADD_REDUCE, 1);
            }

            // ── S_RECIP: SRF[2] = 1 / SRF[1] ──
            SEND_SI(OP_S_RECIP, 2, 1);

            // ── Pass 3: probs = exp(s-m) * inv_sum → V.T context ──
            for (tc = 0; tc < nt; tc++) {
                uint32_t base = tc * 4;

                SEND_LO(OP_V_RD_DRAM, S_SCORE + base);
                SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_V_RD, MEM_SPU_BROADCAST, 0);
                SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_VV_B_SUB_A, 0, 0);
                SEND_SI(OP_V_EXP, 0, 0);
                SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_V_RD, MEM_SPU_BROADCAST, 2);
                SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_VV_MUL, 0, 0);
                SEND_LO(OP_V_WR_DRAM, S_PROB + base);

                for (d = 0; d < MODEL_DIM; d++) {
                    SEND_LO(OP_V_RD_DRAM, svt + tc * MODEL_DIM * MODEL_DIM + d * MODEL_DIM);
                    SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
                }
                SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

                SEND_LO(OP_V_RD_DRAM, S_PROB + base);
                SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
                SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
                SEND_SI(OP_MV_MUL, 0, 0);

                mask_head(ho);

                SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
                SEND_LO(OP_V_RD_DRAM, S_CTX + q * MODEL_DIM);
                SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_VV_ADD, 0, 0);
                SEND_LO(OP_V_WR_DRAM, S_CTX + q * MODEL_DIM);
            }
        }
    }

    // ── Phase 4: c_proj ──
    for (i = 0; i < sl; i++) {
        mvm(ADDR_C_PROJ, S_CTX + i * MODEL_DIM, MEM_MULTIPLY_VRF);
        SEND_LO(OP_V_WR_DRAM, S_ATTN_OUT + i * MODEL_DIM);
        SEND_SI(OP_V_RD, MEM_FILL, 0);
    }

    // ── Phase 5: Residual ──
    for (i = 0; i < sl; i++) {
        SEND_LO(OP_V_RD_DRAM, S_X + i * MODEL_DIM);
        SEND_LO(OP_V_RD_DRAM, S_ATTN_OUT + i * MODEL_DIM);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_LO(OP_V_WR_DRAM, S_AFTER_ATTN + i * MODEL_DIM);
    }

    // ── Phase 6: MLP c_fc + ReLU ──
    for (i = 0; i < sl; i++) {
        mvm(ADDR_C_FC, S_AFTER_ATTN + i * MODEL_DIM, MEM_MULTIPLY_VRF);
        SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
        SEND_LO(OP_V_RD_DRAM, ADDR_C_FC_BIAS);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_SI(OP_V_RELU, 0, 0);
        SEND_LO(OP_V_WR_DRAM, S_FC + i * MODEL_DIM);
    }

    // ── Phase 7: MLP rank-1 + residual (CPU-assisted via DRAM window) ──
    for (i = 0; i < sl; i++) {
        // NPU: dot = h · v → store to S_TEMP+32
        SEND_LO(OP_M_RD_DRAM, ADDR_C_PROJ_V);
        SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);
        SEND_LO(OP_V_RD_DRAM, S_FC + i * MODEL_DIM);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_MV_MUL, 0, 0);
        SEND_LO(OP_V_WR_DRAM, S_TEMP + 32);
        npu_wait_done();

        // CPU: read dot via DRAM window (raw uint32, avoid soft-float)
        uint32_t dot_raw = npu_read_reg(NPU_DRAM_BASE + (S_TEMP + 32) * 4);

        // mlp_out = u * dot. c_proj_u = [0, 0, 0, 1], so mlp_out = [0, 0, 0, dot]
        uint32_t mo = S_MLP_OUT + i * MODEL_DIM;
        npu_write_reg(NPU_DRAM_BASE + mo * 4, 0);
        npu_write_reg(NPU_DRAM_BASE + (mo + 1) * 4, 0);
        npu_write_reg(NPU_DRAM_BASE + (mo + 2) * 4, 0);
        npu_write_reg(NPU_DRAM_BASE + (mo + 3) * 4, dot_raw);

        // NPU: residual add (after_attn + mlp_out)
        SEND_LO(OP_V_RD_DRAM, S_AFTER_ATTN + i * MODEL_DIM);
        SEND_LO(OP_V_RD_DRAM, S_MLP_OUT + i * MODEL_DIM);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_LO(OP_V_WR_DRAM, S_AFTER_MLP + i * MODEL_DIM);
    }

    // ── Phase 8: last_h ──
    uint32_t last = S_AFTER_MLP + (sl - 1) * MODEL_DIM;
    SEND_LO(OP_V_RD_DRAM, last);
    SEND_LO(OP_V_WR_DRAM, FW_LAST_H);
    npu_wait_done();
}

void main(void)
{
    while (npu_read_reg(NPU_STATUS) & NPU_STATUS_BUSY);
    adder_forward();
    npu_set_done();
    while (1);
}
