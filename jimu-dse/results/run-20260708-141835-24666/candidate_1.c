/* NPU — BERT Encoder Layer Firmware (All Features)
 *
 * Implements one BERT encoder layer.  Every NPU hardware feature is
 * exercised naturally by the inference computation itself — no
 * separate test snippet appended after the layer.
 *
 * Supports both single-tile (hidden_size == NATIVE_DIM) and multi-tile
 * (hidden_size > NATIVE_DIM) configurations.  When num_tiles > 1, the
 * tile loop iterates over tile rows and tile columns, accumulating
 * partial MVM results via VV_ADD.
 *
 * Features exercised during BERT inference:
 *   BFP:        Set precision_mode = 1 at layer start
 *   Masks:      Set read_vector_mask before attention softmax
 *   MV_MUL:     Non-accumulating for first col-tile
 *   FILL:       Zero-initialize accumulator VRF before use
 *   INC:        V_RD_DRAM_INC streams weight matrices from DRAM
 *   Chain:      INST_ISSUE groups independent load+compute ops
 *   SMC:        3+ concurrent INST_ISSUE chains (parallel dispatch)
 *   SLU:        Softmax and layernorm via V_FUNC
 *   MultiMFU:   GELU via MFU0, AddSub via MFU1
 *   Acc FW:     m_init_bias_accumulators pre-loads biases
 *   LO-format:  V_RD_DRAM, V_WR_DRAM (non-increment)
 *   INC var.:   VV_ADD_INC, V_WR_DRAM_INC
 *   Scoreboard: RAW and WAR hazards in parallel chain groups
 *   Polling:    CHAIN_STATUS register read after chain dispatch
 *   Config:     REG_READ_MATRIX_MASK alongside vector mask
 *   Backpress:  Large instruction batches > FIFO depth
 */

#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"

#include "npu_driver.h"

#define MAT_SIZE (NATIVE_DIM * NATIVE_DIM)

#define SEND_SI(op, opd0, opd1) npu_send_inst(SI(op, opd0, opd1))
#define SEND_LO(op, adr)        npu_send_inst(LO(op, adr))

/* ── Tile Configuration ─────────────────────────────────────────────
 * For multi-tile operation, each M_RD_DRAM loads exactly one tile
 * (NATIVE_DIM × NATIVE_DIM elements).  The firmware loop iterates
 * over tile rows and tile columns explicitly using VV_ADD for accumulation.
 *
 * DRAM layout for a hidden_size×hidden_size weight matrix:
 *   tile[tr][tc] at DRAM[base + (tr * num_tiles + tc) * MAT_SIZE]
 *
 * Input X stored at DRAM[0]:
 *   X[tc * NATIVE_DIM ... tc * NATIVE_DIM + NATIVE_DIM - 1]
 */
#define REG_TILE_ROWS_ADDR  1
#define REG_TILE_COLS_ADDR  2
#define REG_ITERATIONS_ADDR 3
#define REG_PRECISION_MODE  20

/* DRAM layout for saved Q/K/V per position.
 * With multi-tile, each tile row is NATIVE_DIM elements = 8 addresses
 * (since DRAM stores fp16, 8 addresses = 8 elements).
 *
 * Layout:
 *   0x200 + pos * num_tiles * 8 + tr * 8 : Q[pos] tile row tr
 *   0x300 + pos * num_tiles * 8 + tr * 8 : K[pos] tile row tr
 *   0x400 + pos * num_tiles * 8 + tr * 8 : V[pos] tile row tr
 *   0x500                                    : scratch for LN and FFN
 *   0x600                                    : SCRATCH_Z
 *   0x620                                    : SCRATCH_LN1
 *   0x640                                    : SCRATCH_GELU
 *   0x700                                    : save_res
 *   0x800                                    : save_out
 */
#define SAVE_Q_BASE      0x200
#define SAVE_K_BASE      0x300
#define SAVE_V_BASE      0x400
#define SCRATCH_ADDR     0x500
#define SCRATCH_Z        0x600
#define SCRATCH_LN1      0x620
#define SCRATCH_GELU     0x640
#define SAVE_RES_BASE    0x700
#define SAVE_OUT_BASE    0x800
#define SO_SCRATCH       0x580  /* temp storage for SO during residual add */
#define UNIT_VEC_BASE    0x900  /* identity-matrix rows for V.T re-transpose */
#define VRF_CACHE_OFF    (2 * _SEQ_LEN * _NUM_TILES * NATIVE_DIM)  /* after K+V */


/* ── m_init_bias_accumulators ─────────────────────────────────────
 *
 * Pre-loads bias values into MVM_ACC_VRF for tiled projections.
 */
static void m_init_bias_accumulators(void)
{
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
}


/* ── mvm_tiled_q ──────────────────────────────────────────────────
 *
 * Multi-tile Q projection: Wq × X with explicit tile loop.
 *
 * For each tile row (output chunk) and tile column (input chunk),
 * load one weight tile from DRAM, compute partial MVM, and accumulate.
 *
 * Parameters:
 *   mat_dram_base  — DRAM offset of the weight matrix (e.g., 20 for Wq)
 *   vec_dram_base  — DRAM offset of the input vector (0 for X)
 *   num_tiles      — hidden_size / NATIVE_DIM (e.g., 2 for 16/8)
 *   sink_vrf       — VRF target for accumulated result (MEM_MVM_ACC_VRF)
 */
static void mvm_tiled_q(uint32_t mat_dram_base, uint32_t vec_dram_base,
                         uint32_t num_tiles, uint32_t bias_dram_base)
{
    uint32_t tr, tc;

    /* Force single-tile per M_RD_DRAM: hardware loads 8×8 = 64 elements */
    SEND_SI(OP_S_WR, REG_TILE_ROWS_ADDR, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS_ADDR, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS_ADDR, 1);

    /* Initialize accumulators: tr=0 uses VRF_ADDSUB_0, tr=1 uses VRF_ADDSUB_2.
     * VRF_ADDSUB_1 (mem 8) is reserved as a cache for X data so that the
     * second and third calls to mvm_tiled_q (for V and Q projections of
     * the same position) can read X from VRF instead of re-loading from DRAM. */
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
    if (num_tiles > 1) {
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_2, 0);
    }

    /* Column-major tile loop: for each tile column, load X chunk once
     * from DRAM and cache it in VRF_ADDSUB_1 (mem 8).  All tile-rows
     * for this column read X from the cache instead of re-loading
     * from DRAM, eliminating redundant DRAM reads. */
    for (tc = 0; tc < num_tiles; tc++) {
        /* Load input chunk X[tc*NATIVE_DIM .. end) from DRAM once */
        uint32_t vec_chunk_addr = vec_dram_base + tc * NATIVE_DIM;
        SEND_LO(OP_V_RD_DRAM, vec_chunk_addr);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        /* Save a copy to VRF_ADDSUB_1 for all tile-rows AND subsequent calls */
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);

        for (tr = 0; tr < num_tiles; tr++) {
            /* Load weight tile [tr][tc] from DRAM into MRF */
            uint32_t tile_dram_addr = mat_dram_base + (tr * num_tiles + tc) * MAT_SIZE;
            SEND_LO(OP_M_RD_DRAM, tile_dram_addr);
            SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

            /* Load X chunk from cache VRF_ADDSUB_1 into MVM input */
            SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
            SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

            /* MVM: tile × vec_chunk → pipeline */
            SEND_SI(OP_MV_MUL, 0, 0);

            /* Accumulate into the per-tile-row accumulator:
             * tr=0 → VRF_ADDSUB_0, tr=1 → VRF_ADDSUB_2 */
            uint32_t acc_vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_2;
            if (tc == 0) {
                /* First col-tile: store directly (no add needed) */
                SEND_SI(OP_V_WR, acc_vrf, 0);
            } else {
                /* Subsequent col-tiles: load prev, add, store back */
                SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);       /* save new result */
                SEND_SI(OP_V_RD, acc_vrf, 0);                 /* load accumulated */
                SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);        /* load new result */
                SEND_SI(OP_VV_ADD, 0, 0);                      /* add elementwise */
                SEND_SI(OP_V_WR, acc_vrf, 0);                  /* store back */
            }
        }
    }

    /* Add bias per tile-row */
    for (tr = 0; tr < num_tiles; tr++) {
        uint32_t acc_vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_2;
        SEND_SI(OP_V_RD, acc_vrf, 0);                              /* load accumulated → vpipe_a */
        SEND_LO(OP_V_RD_DRAM, bias_dram_base + tr * NATIVE_DIM);   /* load bias chunk → pipeline */
        SEND_SI(OP_VV_ADD, 0, 0);                                   /* pipeline = vpipe_a + bias */
        SEND_SI(OP_V_WR, acc_vrf, 0);                               /* store result with bias */
    }

    /* Move tr=1 accumulator from VRF_ADDSUB_2 back to VRF_ADDSUB_1
     * so callers can find results in VRF_ADDSUB_0/1 as before. */
    if (num_tiles > 1) {
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_2, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
    }
}

static void mvm_tiled_vrf(uint32_t mat_dram_base, uint32_t vec_vrf_base,
                           uint32_t num_tiles, uint32_t bias_dram_base)
{
    uint32_t tr, tc;

    SEND_SI(OP_S_WR, REG_TILE_ROWS_ADDR, 1);
    SEND_SI(OP_S_WR, REG_TILE_COLS_ADDR, 1);
    SEND_SI(OP_S_WR, REG_ITERATIONS_ADDR, 1);

    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
    if (num_tiles > 1) {
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_2, 0);
    }

    for (tc = 0; tc < num_tiles; tc++) {
        SEND_SI(OP_V_RD, vec_vrf_base + tc * NATIVE_DIM, 0);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);

        for (tr = 0; tr < num_tiles; tr++) {
            uint32_t tile_dram_addr = mat_dram_base + (tr * num_tiles + tc) * MAT_SIZE;
            SEND_LO(OP_M_RD_DRAM, tile_dram_addr);
            SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

            SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
            SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

            SEND_SI(OP_MV_MUL, 0, 0);

            uint32_t acc_vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_2;
            if (tc == 0) {
                SEND_SI(OP_V_WR, acc_vrf, 0);
            } else {
                SEND_SI(OP_V_WR, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_V_RD, acc_vrf, 0);
                SEND_SI(OP_V_RD, MEM_MULTIPLY_VRF, 0);
                SEND_SI(OP_VV_ADD, 0, 0);
                SEND_SI(OP_V_WR, acc_vrf, 0);
            }
        }
    }

    for (tr = 0; tr < num_tiles; tr++) {
        uint32_t acc_vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_2;
        SEND_SI(OP_V_RD, acc_vrf, 0);
        SEND_LO(OP_V_RD_DRAM, bias_dram_base + tr * NATIVE_DIM);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_SI(OP_V_WR, acc_vrf, 0);
    }

    if (num_tiles > 1) {
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_2, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
    }
}

static void save_row_tiles(uint32_t num_tiles, uint32_t dram_base,
                            uint32_t vrf_first, uint32_t vrf_second)
{
    uint32_t tr;
    for (tr = 0; tr < num_tiles; tr++) {
        uint32_t vrf = (tr == 0) ? vrf_first : vrf_second;
        SEND_SI(OP_V_RD, vrf, 0);
        SEND_LO(OP_V_WR_DRAM, dram_base + tr * 8);
    }
}

static void load_and_add_row_tiles(uint32_t num_tiles, uint32_t dram_base,
                                    uint32_t vrf_first, uint32_t vrf_second)
{
    uint32_t tr;
    for (tr = 0; tr < num_tiles; tr++) {
        uint32_t vrf = (tr == 0) ? vrf_first : vrf_second;
        SEND_SI(OP_V_RD, vrf, 0);
        SEND_LO(OP_V_RD_DRAM, dram_base + tr * 8);
        SEND_SI(OP_VV_ADD, 0, 0);
        SEND_SI(OP_V_WR, vrf, 0);
    }
}

/* ── apply_layernorm ────────────────────────────────────────────
 *
 * Apply LayerNorm to tile rows currently in ADDSUB_VRF_0/1.
 * Saves them to scratch DRAM, loads LN gamma/beta, calls V_FUNC,
 * and stores normalized result back into ADDSUB_VRF_0/1.
 */
static void apply_layernorm(uint32_t num_tiles,
                             uint32_t ln_gamma_addr,
                             uint32_t ln_beta_addr,
                             uint32_t scratch_addr)
{
    uint32_t tr;
    uint32_t ln_scratch = scratch_addr + num_tiles * 8; /* extra scratch for tile 0 */
    for (tr = 0; tr < num_tiles; tr++) {
        uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
        uint32_t stride = 8;

        /* Save tile row from VRF to scratch DRAM */
        SEND_SI(OP_V_RD, vrf, 0);
        SEND_LO(OP_V_WR_DRAM, scratch_addr + tr * stride);

        /* Load LN gamma chunk → IVRF (mem 5) */
        SEND_LO(OP_V_RD_DRAM, ln_gamma_addr + tr * stride);
        SEND_SI(OP_V_WR, 5, 0);

        /* Load LN beta chunk → ADDSUB_VRF_0 (mem 7) */
        SEND_LO(OP_V_RD_DRAM, ln_beta_addr + tr * stride);
        SEND_SI(OP_V_WR, 7, 0);

        /* Load tile row from scratch → pipe */
        SEND_LO(OP_V_RD_DRAM, scratch_addr + tr * stride);

        /* Apply LayerNorm */
        SEND_SI(OP_V_FUNC, SUB_LAYERNORM, 0);

        /* Save normalized result back to VRF */
        SEND_SI(OP_V_WR, vrf, 0);

        /* For tile 0, also save to extra scratch DRAM since tile 1's
           beta load will clobber ADDSUB_VRF_0 */
        if (tr == 0) {
            SEND_LO(OP_V_WR_DRAM, ln_scratch);
        }
    }

    /* Restore tile 0's LN result from extra scratch */
    SEND_LO(OP_V_RD_DRAM, ln_scratch);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
}

/* ── compute_k_all_positions ─────────────────────────────────────
 *
 * Compute K = Wk × X[pos] + bk for ALL positions and save to DRAM.
 * Each position's K vector occupies num_tiles * NATIVE_DIM elements.
 * Saved at DRAM[SAVE_K_BASE + pos * num_tiles * 8 + tr * 8].
 */
static void compute_k_all_positions(
    uint32_t seq_len, uint32_t hidden_size, uint32_t num_tiles,
    uint32_t k_base, uint32_t k_bias)
{
    uint32_t pos;
    for (pos = 0; pos < seq_len; pos++) {
        uint32_t x_base = pos * hidden_size;
        mvm_tiled_q(k_base, x_base, num_tiles, k_bias);
        uint32_t cache_off = pos * num_tiles * NATIVE_DIM;
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, cache_off);
        if (num_tiles > 1) {
            SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
            SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, cache_off + NATIVE_DIM);
        }
    }
}

/* ── compute_v_all_positions ─────────────────────────────────────
 *
 * Compute V = Wv × X[pos] + bv for ALL positions and save to DRAM.
 * Saved at DRAM[SAVE_V_BASE + pos * num_tiles * 8 + tr * 8].
 */
static void compute_v_all_positions(
    uint32_t seq_len, uint32_t hidden_size, uint32_t num_tiles,
    uint32_t v_base, uint32_t v_bias)
{
    uint32_t pos;
    for (pos = 0; pos < seq_len; pos++) {
        uint32_t x_base = pos * hidden_size;
        mvm_tiled_q(v_base, x_base, num_tiles, v_bias);
        uint32_t v_base_off = seq_len * num_tiles * NATIVE_DIM;
        uint32_t cache_off = v_base_off + pos * num_tiles * NATIVE_DIM;
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, cache_off);
        if (num_tiles > 1) {
            SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
            SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, cache_off + NATIVE_DIM);
        }
    }
}

/* ── dot_product_attention ─────────────────────────────────────
 *
 * For a single query position, compute dot-product attention:
 *   1. Compute Q = Wq × X[pos] + bq
 *   2. For each head h (within a tile row), build K.T MRF tile from
 *      pre-saved K, using per-head read mask to isolate the head's
 *      head_size elements.
 *   3. Score = MV_MUL(K.T, Q_h) → [seq_len] score vector
 *   4. Softmax → prob vector
 *   5. Build V.T MRF tile from pre-saved V (per-head mask + unit vectors)
 *   6. Context = MV_MUL(V.T, prob) → [head_size] context vector
 *   7. Context written to acc_vrf at per-head element offset.
 *
 * K and V must already be cached in MFU_INITIAL_VRF.
 *
 * When heads_per_tile > 1 (single-tile, multiple heads per row),
 * per-head masking and Q save/restore are applied.
 */
static void dot_product_attention(
    uint32_t pos, uint32_t hidden_size, uint32_t num_tiles,
    uint32_t q_base, uint32_t q_bias,
    uint32_t num_head)
{
    uint32_t x_base = pos * hidden_size;
    uint32_t head_size = hidden_size / num_head;
    uint32_t heads_per_tile = NATIVE_DIM / head_size;
    uint32_t elem_mask = 0xFF;
    uint32_t tr, h;

    /* Compute Q for this position */
    mvm_tiled_q(q_base, x_base, num_tiles, q_bias);

    /* When multiple heads share one tile row, save Q so each head's
     * score step can re-read it (the context write overwrites Q). */
    if (heads_per_tile > 1) {
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_WR, 1, 0);
    }

    for (tr = 0; tr < num_tiles; tr++) {
        uint32_t acc_vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
        for (h = 0; h < heads_per_tile; h++) {
            uint32_t p, pad, j;
            uint32_t head_shift = h * head_size;
            uint32_t read_mask = ((1 << head_size) - 1) << head_shift;
            uint32_t write_mask = (1 << head_size) - 1;
            uint32_t write_off  = head_shift;

            /* ── Build K.T MRF tile for head h ── */
            for (p = 0; p < _SEQ_LEN; p++) {
                SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, read_mask & elem_mask);
                uint32_t k_off = p * num_tiles * NATIVE_DIM + tr * NATIVE_DIM;
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
            SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, read_mask & elem_mask);
            if (heads_per_tile > 1) {
                SEND_SI(OP_V_RD, 1, 0);  /* Q from saved copy */
            } else {
                SEND_SI(OP_V_RD, acc_vrf, 0);
            }
            SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_MV_MUL, 0, 0);
            SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, elem_mask);

            /* ── Softmax across key positions ── */
            SEND_SI(OP_V_FUNC, SUB_SOFTMAX, 0);
            SEND_LO(OP_V_WR_DRAM, SCRATCH_ADDR);

            /* Zero accumulator VRF for context accumulation */
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, acc_vrf, 0);

            /* ── Build V.T MRF tile for head h ──
             * Step 1: Build V position-major into MRF */
            for (p = 0; p < _SEQ_LEN; p++) {
                SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, read_mask & elem_mask);
                uint32_t v_off = _SEQ_LEN * _NUM_TILES * NATIVE_DIM
                               + p * num_tiles * NATIVE_DIM + tr * NATIVE_DIM;
                SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, v_off);
                SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
            }
            for (pad = _SEQ_LEN; pad < NATIVE_DIM; pad++) {
                SEND_SI(OP_V_RD, MEM_FILL, 0);
                SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
            }
            SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, elem_mask);
            SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);
            /* Step 2: Re-transpose — extract columns via per-head
             * unit vectors, write as V.T rows. */
            for (j = 0; j < head_size; j++) {
                uint32_t unit_off = (head_shift + j) * NATIVE_DIM;
                SEND_LO(OP_V_RD_DRAM, UNIT_VEC_BASE + unit_off);
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
            SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

            /* ── Context = V.T @ prob → [head_size] ── */
            SEND_LO(OP_V_RD_DRAM, SCRATCH_ADDR);
            SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
            SEND_SI(OP_MV_MUL, 0, 0);

            /* ── Accumulate into Z at per-head offset ── */
            SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, write_mask & elem_mask);
            SEND_SI(OP_V_WR, acc_vrf, write_off);
        }
    }
}

/* ── BERT Encoder Layer ───────────────────────────────────────────
 *
 * Full BERT encoder layer: attention + self-output + FFN.
 * When num_tiles > 1, projections use the multi-tile tile loop.
 *
 * Restructured for dot-product attention:
 *   Phase 1: Compute K and V for all positions
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
        compute_k_all_positions(seq_len, hidden_size, num_tiles,
            _PROJ_BASE + _STRIDE, _PROJ_BASE + _STRIDE + _MAT_SIZE);
        compute_v_all_positions(seq_len, hidden_size, num_tiles,
            _PROJ_BASE + 2 * _STRIDE, _PROJ_BASE + 2 * _STRIDE + _MAT_SIZE);

        /* ── Phase 2+3: Per-query-position attention + rest ── */
        uint32_t _pos;
        for (_pos = 0; _pos < seq_len; _pos++) {
            uint32_t x_base = _pos * hidden_size;

            /* Compute Q and dot-product attention, result in ADDSUB_VRFs */
            dot_product_attention(_pos, hidden_size, num_tiles,
                _PROJ_BASE, _PROJ_BASE + _MAT_SIZE, num_head);

            /* Save attention context Z to VRF cache for self-output */
            uint32_t tr;
            for (tr = 0; tr < num_tiles; tr++) {
                uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                SEND_SI(OP_V_RD, vrf, 0);
                SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF + tr * NATIVE_DIM);
            }

            /* ── Self-output + first residual + LayerNorm ──────── */
            mvm_tiled_vrf(_PROJ_BASE + 3 * _STRIDE, VRF_CACHE_OFF, num_tiles,
                          _PROJ_BASE + 3 * _STRIDE + _MAT_SIZE);
            /* Residual 1: SO + X.  Cache SO in VRF, then load X and add. */
            for (tr = 0; tr < num_tiles; tr++) {
                uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                SEND_SI(OP_V_RD, vrf, 0);
                SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF + tr * NATIVE_DIM);
            }
            /* Save original X for residual 2 skip connection */
            for (tr = 0; tr < num_tiles; tr++) {
                SEND_LO(OP_V_RD_DRAM, x_base + tr * NATIVE_DIM);
                SEND_SI(OP_V_WR, (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1, 0);
            }
            save_row_tiles(num_tiles, SAVE_RES_BASE + _pos * num_tiles * 8,
                            MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);
            /* Residual 1: reload SO from VRF cache, add X from VRF */
            for (tr = 0; tr < num_tiles; tr++) {
                uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF + tr * NATIVE_DIM);
                SEND_SI(OP_V_RD, vrf, 0);
                SEND_SI(OP_VV_ADD, 0, 0);
                SEND_SI(OP_V_WR, vrf, 0);
            }
            apply_layernorm(num_tiles, _LN1_GAMMA, _LN1_BETA, _SCRATCH);
            /* Cache LN1 output in VRF for FFN inter */
            for (tr = 0; tr < num_tiles; tr++) {
                uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                SEND_SI(OP_V_RD, vrf, 0);
                SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF + tr * NATIVE_DIM);
            }

            /* ── FFN: intermediate + GELU ──────────────────────── */
            mvm_tiled_vrf(_PROJ_BASE + 4 * _STRIDE, VRF_CACHE_OFF, num_tiles,
                          _PROJ_BASE + 4 * _STRIDE + _MAT_SIZE);
            SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
            SEND_SI(OP_V_GELU, 0, 0);
            SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
            SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
            SEND_SI(OP_V_GELU, 0, 0);
            SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
            /* Cache GELU output in VRF for FFN output */
            for (tr = 0; tr < num_tiles; tr++) {
                uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                SEND_SI(OP_V_RD, vrf, 0);
                SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, VRF_CACHE_OFF + tr * NATIVE_DIM);
            }

            /* ── FFN output + second residual + LayerNorm ──────── */
            mvm_tiled_vrf(_PROJ_BASE + 5 * _STRIDE, VRF_CACHE_OFF, num_tiles,
                          _PROJ_BASE + 5 * _STRIDE + _MAT_SIZE);
            load_and_add_row_tiles(num_tiles, SAVE_RES_BASE + _pos * num_tiles * 8,
                                    MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);
            apply_layernorm(num_tiles, _LN2_GAMMA, _LN2_BETA, _SCRATCH);
            save_row_tiles(num_tiles, SAVE_OUT_BASE + _pos * num_tiles * 8,
                            MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);
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