/* NPU - BERT Encoder Layer Firmware (Single-Tile)
 *
 * Single-tile BERT encoder layer for NATIVE_DIM == hidden_size.
 * All tile loops removed - each projection uses one M_RD_DRAM + MV_MUL.
 * Heads_per_tile=2 attention with per-head masking.
 * K/V cached in VRF bank 6 to eliminate DRAM roundtrip.
 */
#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"
#include "npu_driver.h"

#define MAT_SIZE (NATIVE_DIM * NATIVE_DIM)
#define SEND_SI(op, opd0, opd1) npu_send_inst(SI(op, opd0, opd1))
#define SEND_LO(op, adr)        npu_send_inst(LO(op, adr))

#define SCRATCH_ADDR     0x500
#define SAVE_RES_BASE    0x700
#define SAVE_OUT_BASE    0x800
#define UNIT_VEC_BASE    0x900

/* VRF cache layout in MFU_INITIAL_VRF (bank 6):
 *   K[pos]:  offset = pos * NATIVE_DIM
 *   V[pos]:  offset = _SEQ_LEN * NATIVE_DIM + pos * NATIVE_DIM
 *   Z/cache: offset = 2 * _SEQ_LEN * NATIVE_DIM (VRF_CACHE_OFF)
 */
#define VRF_CACHE_OFF    (2 * _SEQ_LEN * NATIVE_DIM)

static void m_init_bias_accumulators(void)
{
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
}

static void mvm_tiled_q(uint32_t mat_dram_base, uint32_t vec_dram_base,
                         uint32_t num_tiles, uint32_t bias_dram_base)
{
    (void)num_tiles;
    SEND_SI(OP_S_WR, REG_TILE_ROWS, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS, 1);

    SEND_LO(OP_V_RD_DRAM, vec_dram_base);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

    SEND_LO(OP_M_RD_DRAM, mat_dram_base);
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

    SEND_SI(OP_MV_MUL, 0, 0);

    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
    SEND_LO(OP_V_RD_DRAM, bias_dram_base);
    SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, 0);
    SEND_SI(OP_VV_ADD, 0, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
}

static void mvm_tiled_vrf(uint32_t mat_dram_base, uint32_t vec_vrf_offset,
                           uint32_t num_tiles, uint32_t bias_dram_base)
{
    (void)num_tiles;
    SEND_SI(OP_S_WR, REG_TILE_ROWS, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS, 1);

    SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, vec_vrf_offset);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

    SEND_LO(OP_M_RD_DRAM, mat_dram_base);
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

    SEND_SI(OP_MV_MUL, 0, 0);

    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
    SEND_LO(OP_V_RD_DRAM, bias_dram_base);
    SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, 0);
    SEND_SI(OP_VV_ADD, 0, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
}

static void save_row_tiles(uint32_t dram_base, uint32_t vrf_src)
{
    SEND_SI(OP_V_RD, vrf_src, 0);
    SEND_LO(OP_V_WR_DRAM, dram_base);
}

static void load_and_add_row_tiles(uint32_t dram_base, uint32_t vrf_dst)
{
    SEND_SI(OP_V_RD, vrf_dst, 0);
    SEND_LO(OP_V_RD_DRAM, dram_base);
    SEND_SI(OP_VV_ADD, 0, 0);
    SEND_SI(OP_V_WR, vrf_dst, 0);
}

static void apply_layernorm(uint32_t ln_gamma_addr, uint32_t ln_beta_addr,
                             uint32_t scratch_addr)
{
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_LO(OP_V_WR_DRAM, scratch_addr);

    SEND_LO(OP_V_RD_DRAM, ln_gamma_addr);
    SEND_SI(OP_V_WR, 5, 0);

    SEND_LO(OP_V_RD_DRAM, ln_beta_addr);
    SEND_SI(OP_V_WR, 7, 0);

    SEND_LO(OP_V_RD_DRAM, scratch_addr);
    SEND_SI(OP_V_FUNC, SUB_LAYERNORM, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
}

static void precompute_k_all(
    uint32_t seq_len, uint32_t num_tiles,
    uint32_t k_base, uint32_t k_bias)
{
    uint32_t pos;
    for (pos = 0; pos < seq_len; pos++) {
        uint32_t x_base = pos * NATIVE_DIM;
        mvm_tiled_q(k_base, x_base, num_tiles, k_bias);
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, pos * NATIVE_DIM);
    }
}

static void precompute_v_all(
    uint32_t seq_len, uint32_t num_tiles,
    uint32_t v_base, uint32_t v_bias)
{
    uint32_t pos;
    for (pos = 0; pos < seq_len; pos++) {
        uint32_t x_base = pos * NATIVE_DIM;
        mvm_tiled_q(v_base, x_base, num_tiles, v_bias);
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, _SEQ_LEN * NATIVE_DIM + pos * NATIVE_DIM);
    }
}

static void dot_product_attention(
    uint32_t pos, uint32_t num_tiles,
    uint32_t q_base, uint32_t q_bias,
    uint32_t num_head)
{
    uint32_t x_base = pos * NATIVE_DIM;
    uint32_t head_size = NATIVE_DIM / num_head;
    uint32_t heads_per_tile = NATIVE_DIM / head_size;
    uint32_t elem_mask = 0xFF;
    uint32_t h;

    mvm_tiled_q(q_base, x_base, num_tiles, q_bias);

    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_SI(OP_V_WR, 1, 0);

    /* Zero accumulator once before head loop (each head writes to its own offset) */
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);

    for (h = 0; h < heads_per_tile; h++) {
        uint32_t p, pad, j;
        uint32_t head_shift = h * head_size;
        uint32_t read_mask = ((1 << head_size) - 1) << head_shift;
        uint32_t write_mask = (1 << head_size) - 1;
        uint32_t write_off  = head_shift;

        for (p = 0; p < _SEQ_LEN; p++) {
            SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, read_mask & elem_mask);
            SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, p * NATIVE_DIM);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        for (pad = _SEQ_LEN; pad < NATIVE_DIM; pad++) {
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

        SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, read_mask & elem_mask);
        SEND_SI(OP_V_RD, 1, 0);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_MV_MUL, 0, 0);
        SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, elem_mask);

        SEND_SI(OP_V_FUNC, SUB_SOFTMAX, 0);
        SEND_LO(OP_V_WR_DRAM, SCRATCH_ADDR);

        for (p = 0; p < _SEQ_LEN; p++) {
            SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, read_mask & elem_mask);
            SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, _SEQ_LEN * NATIVE_DIM + p * NATIVE_DIM);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        for (pad = _SEQ_LEN; pad < NATIVE_DIM; pad++) {
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, elem_mask);
        SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

        for (j = 0; j < head_size; j++) {
            uint32_t unit_off = (head_shift + j) * NATIVE_DIM;
            SEND_LO(OP_V_RD_DRAM, UNIT_VEC_BASE + unit_off);
            SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_MV_MUL, 0, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        for (pad = head_size; pad < NATIVE_DIM; pad++) {
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

        SEND_LO(OP_V_RD_DRAM, SCRATCH_ADDR);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_MV_MUL, 0, 0);

        SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, write_mask & elem_mask);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, write_off);
    }
    SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, elem_mask);
}

void bert_encoder_layer(
    uint32_t seq_len,
    uint32_t hidden_size,
    uint32_t num_head,
    uint32_t num_layers
)
{
    uint32_t num_tiles = hidden_size / NATIVE_DIM;

    SEND_SI(OP_S_WR, REG_PRECISION_MODE, 1);
    SEND_SI(OP_S_WR, REG_TILE_ROWS, num_tiles);
    SEND_SI(OP_S_WR, REG_TILE_COLS, num_tiles);
    SEND_SI(OP_S_WR, REG_ITERATIONS, seq_len);
    SEND_SI(OP_S_WR, REG_READ_MATRIX_MASK, 0xFF);

    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);

    precompute_k_all(seq_len, num_tiles,
        _PROJ_BASE + _STRIDE, _PROJ_BASE + _STRIDE + _MAT_SIZE);
    precompute_v_all(seq_len, num_tiles,
        _PROJ_BASE + 2 * _STRIDE, _PROJ_BASE + 2 * _STRIDE + _MAT_SIZE);

    uint32_t _pos;
    for (_pos = 0; _pos < seq_len; _pos++) {
        uint32_t x_base = _pos * hidden_size;

        dot_product_attention(_pos, num_tiles,
            _PROJ_BASE, _PROJ_BASE + _MAT_SIZE, num_head);

        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);

        mvm_tiled_vrf(_PROJ_BASE + 3 * _STRIDE, VRF_CACHE_OFF, num_tiles,
                      _PROJ_BASE + 3 * _STRIDE + _MAT_SIZE);

        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);

        SEND_LO(OP_V_RD_DRAM, x_base);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
        save_row_tiles(SAVE_RES_BASE + _pos * 8, MEM_ADDSUB_VRF_0);

        SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);

        apply_layernorm(_LN1_GAMMA, _LN1_BETA, _SCRATCH);

        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);

        mvm_tiled_vrf(_PROJ_BASE + 4 * _STRIDE, VRF_CACHE_OFF, num_tiles,
                      _PROJ_BASE + 4 * _STRIDE + _MAT_SIZE);

        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_GELU, 0, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);

        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);

        mvm_tiled_vrf(_PROJ_BASE + 5 * _STRIDE, VRF_CACHE_OFF, num_tiles,
                      _PROJ_BASE + 5 * _STRIDE + _MAT_SIZE);

        load_and_add_row_tiles(SAVE_RES_BASE + _pos * 8, MEM_ADDSUB_VRF_0);

        apply_layernorm(_LN2_GAMMA, _LN2_BETA, _SCRATCH);

        save_row_tiles(SAVE_OUT_BASE + _pos * 8, MEM_ADDSUB_VRF_0);
    }

    npu_wait_done();
    SEND_SI(OP_S_WR, REG_PRECISION_MODE, 0);
}

void main(void)
{
    uint32_t hidden_size = npu_read_reg(0x20);
    if (hidden_size == 0) {
        hidden_size = NATIVE_DIM;
    }

    uint32_t seq_len = npu_read_reg(0x24);
    if (seq_len == 0) {
        seq_len = 1;
    }

    uint32_t num_head = 4;
    #ifdef NUM_HEAD
    num_head = NUM_HEAD;
    #endif

    while (npu_read_reg(4) & 1);

    m_init_bias_accumulators();
    bert_encoder_layer(seq_len, hidden_size, num_head, 1);

    npu_set_done();
    while (1);
}
