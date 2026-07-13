/* NPU — BERT Encoder Layer Firmware (Single-Tile Optimized)
 *
 * Implements one BERT encoder layer for dim=4, hidden=4, num_head=2
 * (single-tile: num_tiles=1). Every NPU hardware feature is exercised
 * naturally by the inference computation.
 *
 * Features exercised during BERT inference:
 *   BFP:        Set precision_mode = 1 at layer start
 *   Masks:      Set read/write vector mask for attention heads
 *   MV_MUL:     Matrix-vector multiply (single tile, no accumulation)
 *   FILL:       Zero-initialize accumulator VRF before use
 *   LO-format:  V_RD_DRAM, V_WR_DRAM for weight/bias/X loads
 *   Chain:      INST_ISSUE groups independent load+compute ops
 *   SLU:        Softmax and layernorm via V_FUNC
 *   MultiMFU:   GELU via MFU0, AddSub via MFU1
 *   VRF Cache:  K/V cached in MFU_INITIAL_VRF, softmax in ADDSUB_VRF_2
 *   Scoreboard: RAW and WAR hazards in parallel chain groups
 */

#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"

#include "npu_driver.h"

#define MAT_SIZE (NATIVE_DIM * NATIVE_DIM)

#define SEND_SI(op, opd0, opd1) npu_send_inst(SI(op, opd0, opd1))
#define SEND_LO(op, adr)        npu_send_inst(LO(op, adr))

/* ── Configuration ──────────────────────────────────────────────── */
#define REG_TILE_ROWS_ADDR  1
#define REG_TILE_COLS_ADDR  2
#define REG_ITERATIONS_ADDR 3

/* ── DRAM Layout ────────────────────────────────────────────────── */
#define SAVE_Q_BASE      0x200
#define SAVE_K_BASE      0x300
#define SAVE_V_BASE      0x400
#define SCRATCH_ADDR     0x500
#define SAVE_RES_BASE    0x700
#define SAVE_OUT_BASE    0x800
#define UNIT_VEC_BASE    0x900

/* ── VRF Cache Layout (MFU_INITIAL_VRF, 4096 elements) ────────────
 * K[pos] stored at offset pos * NATIVE_DIM  (positions 0..seq_len-1)
 * V[pos] stored at offset seq_len * NATIVE_DIM + pos * NATIVE_DIM
 * Working cache (SO/LN/GELU) at VRF_CACHE_OFF = 2 * seq_len * NATIVE_DIM
 */
#define VRF_CACHE_OFF    (2 * _SEQ_LEN * _NUM_TILES * NATIVE_DIM)

/* ── Softmax VRF Cache (ADDSUB_VRF_2, 64 elements) ───────────────
 * After V_FUNC/SOFTMAX, result is cached here instead of DRAM scratch.
 * This eliminates the DRAM round-trip (V_WR_DRAM + V_RD_DRAM) for
 * softmax probabilities, saving 16 bytes per head per position.
 * ADDSUB_VRF_2 is unused in single-tile mode (num_tiles=1).
 */
#define SOFTMAX_VRF_CACHE  0


/* ── m_init_bias_accumulators ───────────────────────────────────── */
static void m_init_bias_accumulators(void)
{
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
}


/* ── mvm_tiled_q ──────────────────────────────────────────────────
 * Single-tile Q projection: Wq × X + bq.
 * Loads full input vector from DRAM, full weight matrix into MRF,
 * computes MV_MUL, adds bias, stores result to ADDSUB_VRF_0.
 *
 * Parameters:
 *   mat_dram_base  — DRAM offset of the weight matrix
 *   vec_dram_base  — DRAM offset of the input vector
 *   bias_dram_base — DRAM offset of the bias vector
 */
static void mvm_tiled_q(uint32_t mat_dram_base, uint32_t vec_dram_base,
                         uint32_t bias_dram_base)
{
    /* Configure single-tile: hardware loads NATIVE_DIM × NATIVE_DIM matrix */
    SEND_SI(OP_S_WR, REG_TILE_ROWS_ADDR, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS_ADDR, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS_ADDR, 1);

    /* Zero accumulator */
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);

    /* Load full input vector from DRAM */
    SEND_LO(OP_V_RD_DRAM, vec_dram_base);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

    /* Load full weight matrix into MRF */
    SEND_LO(OP_M_RD_DRAM, mat_dram_base);
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

    /* MVM: W × X → pipeline */
    SEND_SI(OP_MV_MUL, 0, 0);

    /* Add bias: load accumulated, load bias, add, store back */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_LO(OP_V_RD_DRAM, bias_dram_base);
    SEND_SI(OP_VV_ADD, 0, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
}


/* ── mvm_tiled_vrf ────────────────────────────────────────────────
 * Single-tile projection from VRF source: W × X_vrf + b.
 * Input vector is read from MFU_INITIAL_VRF at vec_vrf_base.
 */
static void mvm_tiled_vrf(uint32_t mat_dram_base, uint32_t vec_vrf_base,
                           uint32_t bias_dram_base)
{
    SEND_SI(OP_S_WR, REG_TILE_ROWS_ADDR, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS_ADDR, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS_ADDR, 1);

    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);

    /* Load input vector from VRF cache */
    SEND_SI(OP_V_RD, vec_vrf_base, 0);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

    /* Load weight matrix into MRF */
    SEND_LO(OP_M_RD_DRAM, mat_dram_base);
    SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

    /* MVM */
    SEND_SI(OP_MV_MUL, 0, 0);

    /* Add bias */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_LO(OP_V_RD_DRAM, bias_dram_base);
    SEND_SI(OP_VV_ADD, 0, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
}


/* ── save_row_tiles ───────────────────────────────────────────────
 * Single-tile: save ADDSUB_VRF_0 to DRAM at dram_base. */
static void save_row_tiles(uint32_t dram_base)
{
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_LO(OP_V_WR_DRAM, dram_base);
}

/* ── load_and_add_row_tiles ───────────────────────────────────────
 * Single-tile: load DRAM at dram_base, add to ADDSUB_VRF_0, store back. */
static void load_and_add_row_tiles(uint32_t dram_base)
{
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_LO(OP_V_RD_DRAM, dram_base);
    SEND_SI(OP_VV_ADD, 0, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
}


/* ── apply_layernorm ──────────────────────────────────────────────
 * Single-tile LayerNorm: gamma and beta each occupy one tile row.
 * Input is in ADDSUB_VRF_0, output stored back to ADDSUB_VRF_0.
 *
 * The 2-read-port constraint requires staging the input through
 * DRAM scratch: VRF → DRAM → pipe, while gamma/beta load from DRAM.
 */
static void apply_layernorm(uint32_t ln_gamma_addr,
                             uint32_t ln_beta_addr,
                             uint32_t scratch_addr)
{
    /* Save tile row from VRF to scratch DRAM (free read port for gamma/beta) */
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
    SEND_LO(OP_V_WR_DRAM, scratch_addr);

    /* Load LN gamma → MVM_INITIAL_VRF (mem 5) */
    SEND_LO(OP_V_RD_DRAM, ln_gamma_addr);
    SEND_SI(OP_V_WR, 5, 0);

    /* Load LN beta → ADDSUB_VRF_0 (mem 7) */
    SEND_LO(OP_V_RD_DRAM, ln_beta_addr);
    SEND_SI(OP_V_WR, 7, 0);

    /* Load tile row from scratch → pipe (3rd input for V_FUNC) */
    SEND_LO(OP_V_RD_DRAM, scratch_addr);

    /* Apply LayerNorm: pipe = layernorm(pipe, gamma, beta) */
    SEND_SI(OP_V_FUNC, SUB_LAYERNORM, 0);

    /* Save normalized result back to VRF */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
}


/* ── compute_k_all_positions ──────────────────────────────────────
 * Compute K = Wk × X[pos] + bk for ALL positions and cache in VRF.
 * Each position's K vector is stored at MFU_INITIAL_VRF[pos * NATIVE_DIM].
 */
static void compute_k_all_positions(
    uint32_t seq_len, uint32_t hidden_size,
    uint32_t k_base, uint32_t k_bias)
{
    uint32_t pos;
    for (pos = 0; pos < seq_len; pos++) {
        uint32_t x_base = pos * hidden_size;
        mvm_tiled_q(k_base, x_base, k_bias);
        /* Cache K[pos] in MFU_INITIAL_VRF */
        uint32_t cache_off = pos * NATIVE_DIM;
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, cache_off);
    }
}

/* ── compute_v_all_positions ──────────────────────────────────────
 * Compute V = Wv × X[pos] + bv for ALL positions and cache in VRF.
 * V[pos] stored at MFU_INITIAL_VRF[seq_len * NATIVE_DIM + pos * NATIVE_DIM].
 */
static void compute_v_all_positions(
    uint32_t seq_len, uint32_t hidden_size,
    uint32_t v_base, uint32_t v_bias)
{
    uint32_t pos;
    uint32_t v_base_off = seq_len * NATIVE_DIM;
    for (pos = 0; pos < seq_len; pos++) {
        uint32_t x_base = pos * hidden_size;
        mvm_tiled_q(v_base, x_base, v_bias);
        uint32_t cache_off = v_base_off + pos * NATIVE_DIM;
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, cache_off);
    }
}


/* ── dot_product_attention ────────────────────────────────────────
 *
 * For a single query position, compute dot-product attention:
 *   1. Compute Q = Wq × X[pos] + bq
 *   2. For each head h, build K.T MRF tile from VRF-cached K
 *   3. Score = MV_MUL(K.T, Q_h) → [seq_len] score vector
 *   4. Softmax → prob vector (cached in VRF, not DRAM)
 *   5. Build V.T MRF tile from VRF-cached V
 *   6. Context = MV_MUL(V.T, prob) → [head_size] context vector
 *   7. Accumulate context into Z with per-head write mask
 *
 * K and V are pre-cached in MFU_INITIAL_VRF by Phase 1.
 * Softmax probabilities are cached in ADDSUB_VRF_2 to eliminate
 * the DRAM round-trip (V_WR_DRAM + V_RD_DRAM).
 */
static void dot_product_attention(
    uint32_t pos, uint32_t hidden_size,
    uint32_t q_base, uint32_t q_bias,
    uint32_t num_head)
{
    uint32_t x_base = pos * hidden_size;
    uint32_t head_size = hidden_size / num_head;
    uint32_t heads_per_tile = NATIVE_DIM / head_size;
    uint32_t h;

    /* Compute Q for this position — kept in ADDSUB_VRF_0 */
    mvm_tiled_q(q_base, x_base, q_bias);

    /* acc_vrf = ADDSUB_VRF_0 for single tile (tr=0) */
    uint32_t acc_vrf = MEM_ADDSUB_VRF_0;

    for (h = 0; h < heads_per_tile; h++) {
        uint32_t p, pad, j;

        /* ── Build K.T MRF tile for head h ── */
        for (p = 0; p < _SEQ_LEN; p++) {
            SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);
            uint32_t k_off = p * NATIVE_DIM;
            SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, k_off);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        /* Zero-pad remaining rows to NATIVE_DIM */
        for (pad = _SEQ_LEN; pad < NATIVE_DIM; pad++) {
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        /* Transfer row buffer to MRF */
        SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

        /* ── Score = K.T @ Q_h → [seq_len] ── */
        SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);
        SEND_SI(OP_V_RD, acc_vrf, 0);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_MV_MUL, 0, 0);
        /* Pipeline: score[i] = K[pos_i, head_h] · Q[head_h] */
        SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);

        /* ── Softmax across key positions ── */
        SEND_SI(OP_V_FUNC, SUB_SOFTMAX, 0);
        /* Cache prob in VRF instead of DRAM scratch.
         * V_WR stores pipeline to VRF without consuming it.
         * This eliminates the DRAM round-trip (V_WR_DRAM + V_RD_DRAM). */
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_2, SOFTMAX_VRF_CACHE);

        /* Q consumed — zero accumulator VRF for context accumulation */
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        SEND_SI(OP_V_WR, acc_vrf, 0);

        /* ── Build V.T MRF tile for head h (element-major) ──
         * Step 1: Build V position-major into MRF */
        for (p = 0; p < _SEQ_LEN; p++) {
            SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);
            uint32_t v_off = _SEQ_LEN * _NUM_TILES * NATIVE_DIM
                           + p * NATIVE_DIM;
            SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, v_off);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        for (pad = _SEQ_LEN; pad < NATIVE_DIM; pad++) {
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        /* Restore read mask before M_RD */
        SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);
        SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

        /* Step 2: Re-transpose by extracting columns via unit vectors.
         * For each element j (0..head_size-1), MV_MUL(V, e_j)
         * extracts column j of V → pipeline gets [V[0,j], V[1,j], ...].
         * Write this column as row j of V.T via VecToMatRow. */
        for (j = 0; j < head_size; j++) {
            SEND_LO(OP_V_RD_DRAM, UNIT_VEC_BASE + j * NATIVE_DIM);
            SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_MV_MUL, 0, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        /* Zero-pad remaining rows of V.T */
        for (pad = head_size; pad < NATIVE_DIM; pad++) {
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        /* Transfer V.T row buffer to MRF */
        SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

        /* ── Context = V.T @ prob → [head_size] ── */
        /* Read prob from VRF cache instead of DRAM scratch */
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_2, SOFTMAX_VRF_CACHE);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_MV_MUL, 0, 0);
        /* Pipeline: context[j] = sum_pos V[pos, head_h, j] * prob[pos] */

        /* ── Accumulate into Z with write mask ── */
        SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, 0xFF);
        SEND_SI(OP_V_WR, acc_vrf, 0);
    }
}


/* ── BERT Encoder Layer ───────────────────────────────────────────
 *
 * Full BERT encoder layer: attention + self-output + FFN.
 * Single-tile operation: hidden_size == NATIVE_DIM, num_tiles == 1.
 *
 * Structure:
 *   Phase 1: Compute K and V for all positions (cached in VRF)
 *   Phase 2: For each query position, compute Q and dot-product attention
 *   Phase 3: Self-output, residual, LayerNorm, FFN (per position)
 */
void bert_encoder_layer(
    uint32_t seq_len,
    uint32_t hidden_size,
    uint32_t num_head,
    uint32_t num_layers
)
{
    uint32_t num_tiles = hidden_size / NATIVE_DIM;

    /* ════════════════════════════════════════════════════════════
     * 1.  CONFIGURATION
     * ════════════════════════════════════════════════════════════ */
    SEND_SI(OP_S_WR, 20, 1);                         /* BFP: precision_mode */
    SEND_SI(OP_S_WR, REG_TILE_ROWS, num_tiles);
    SEND_SI(OP_S_WR, REG_TILE_COLS, num_tiles);
    SEND_SI(OP_S_WR, REG_ITERATIONS, seq_len);
    SEND_SI(OP_S_WR, REG_READ_MATRIX_MASK, 0xFF);

    /* ════════════════════════════════════════════════════════════
     * 2.  FILL + BIAS INIT
     * ════════════════════════════════════════════════════════════ */
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);

    /* ── Phase 1: Compute K and V for all positions ── */
    compute_k_all_positions(seq_len, hidden_size,
        _PROJ_BASE + _STRIDE, _PROJ_BASE + _STRIDE + _MAT_SIZE);
    compute_v_all_positions(seq_len, hidden_size,
        _PROJ_BASE + 2 * _STRIDE, _PROJ_BASE + 2 * _STRIDE + _MAT_SIZE);

    /* ── Phase 2+3: Per-query-position attention + rest ── */
    uint32_t _pos;
    for (_pos = 0; _pos < seq_len; _pos++) {
        uint32_t x_base = _pos * hidden_size;

        /* Compute Q and dot-product attention, result in ADDSUB_VRF_0 */
        dot_product_attention(_pos, hidden_size,
            _PROJ_BASE, _PROJ_BASE + _MAT_SIZE, num_head);

        /* Save attention context Z to VRF cache for self-output */
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);

        /* ── Self-output + first residual + LayerNorm ──────── */
        mvm_tiled_vrf(_PROJ_BASE + 3 * _STRIDE, VRF_CACHE_OFF,
                      _PROJ_BASE + 3 * _STRIDE + _MAT_SIZE);
        /* Residual 1: SO + X.  Cache SO in VRF, then load X and add. */
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);
        /* Load original X for residual 2 skip connection */
        SEND_LO(OP_V_RD_DRAM, x_base);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
        save_row_tiles(SAVE_RES_BASE + _pos * num_tiles * 8);
        /* Residual 1: reload SO from VRF cache, add X from VRF */
        SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
        apply_layernorm(_LN1_GAMMA, _LN1_BETA, _SCRATCH);
        /* Cache LN1 output in VRF for FFN inter */
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);

        /* ── FFN: intermediate + GELU ──────────────────────── */
        mvm_tiled_vrf(_PROJ_BASE + 4 * _STRIDE, VRF_CACHE_OFF,
                      _PROJ_BASE + 4 * _STRIDE + _MAT_SIZE);
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_GELU, 0, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
        /* Cache GELU output in VRF for FFN output */
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF);

        /* ── FFN output + second residual + LayerNorm ──────── */
        mvm_tiled_vrf(_PROJ_BASE + 5 * _STRIDE, VRF_CACHE_OFF,
                      _PROJ_BASE + 5 * _STRIDE + _MAT_SIZE);
        load_and_add_row_tiles(SAVE_RES_BASE + _pos * num_tiles * 8);
        apply_layernorm(_LN2_GAMMA, _LN2_BETA, _SCRATCH);
        save_row_tiles(SAVE_OUT_BASE + _pos * num_tiles * 8);
    }

    npu_wait_done();

    /* Restore FP16 mode */
    SEND_SI(OP_S_WR, 20, 0);
}


/* ── main ───────────────────────────────────────────────────────── */

void main(void)
{
    /* Read hidden_size from MMIO register NPU_REG_HIDDEN_SIZE (0x20).
     * Test harness writes this before the firmware runs.
     * Default (when not set): NATIVE_DIM → single-tile mode.
     */
    uint32_t hidden_size = npu_read_reg(NPU_REG_HIDDEN_SIZE);
    if (hidden_size == 0) {
        hidden_size = NATIVE_DIM;
    }

    uint32_t seq_len = npu_read_reg(NPU_REG_SEQ_LEN);
    if (seq_len == 0) {
        seq_len = 1;
    }

    uint32_t num_head = 4;
    #ifdef NUM_HEAD
    num_head = NUM_HEAD;
    #endif

    /* Wait until NPU is idle */
    while (npu_read_reg(NPU_STATUS) & NPU_STATUS_BUSY);

    m_init_bias_accumulators();
    bert_encoder_layer(seq_len, hidden_size, num_head, 1);

    npu_set_done();

    while (1);
}
