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

/* ── Single-tile VRF parameter cache ────────────────────────────────
 * All projection biases and LayerNorm gamma/beta vectors are streamed
 * into MEM_MVM_ACC_VRF (bank 13, 256 elements, otherwise unused in
 * single-tile mode) once before the K/V/Q phases.  Every in-loop bias
 * load then reads the 1-cycle VRF cache instead of a 3-cycle DRAM
 * transfer, and the phase-5 gamma/beta loads are hoisted into the
 * pre-compute slot so no LN input staging round trip is needed.
 * Only valid when num_tiles == 1; multi-tile keeps the DRAM path. */
#define CACHE_K_BIAS     (0 * NATIVE_DIM)
#define CACHE_V_BIAS     (1 * NATIVE_DIM)
#define CACHE_Q_BIAS     (2 * NATIVE_DIM)
#define CACHE_BO         (3 * NATIVE_DIM)
#define CACHE_BI         (4 * NATIVE_DIM)
#define CACHE_BO2        (5 * NATIVE_DIM)
#define CACHE_LN1_GAMMA  (6 * NATIVE_DIM)
#define CACHE_LN1_BETA   (7 * NATIVE_DIM)
#define CACHE_LN2_GAMMA  (8 * NATIVE_DIM)
#define CACHE_LN2_BETA   (9 * NATIVE_DIM)

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
 *   0x800                                    : save_out
 */
#define SAVE_Q_BASE      0x200
#define SAVE_K_BASE      0x300
#define SAVE_V_BASE      0x400
#define SCRATCH_ADDR     0x500
#define SCRATCH_Z        0x600
#define SCRATCH_LN1      0x620
#define SCRATCH_GELU     0x640
#define SAVE_OUT_BASE    0x800
#define UNIT_VEC_BASE    0x900  /* identity-matrix rows for V.T re-transpose */


/* ── m_init_bias_accumulators ─────────────────────────────────────
 *
 * Pre-loads bias values into MVM_ACC_VRF for tiled projections.
 */
static void m_init_bias_accumulators(void)
{
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
}


/* ── cache_bias_layernorm_params ─────────────────────────────────
 *
 * Single-tile only: preload every projection bias and LN gamma/beta
 * vector from DRAM into the MVM_ACC_VRF cache once.  Each vector is
 * NATIVE_DIM elements; the pipeline semantics (V_RD_DRAM loads pipe,
 * V_WR writes pipe and keeps it) preserve bit-identical fp16 values.
 */
static void cache_bias_layernorm_params(void)
{
    SEND_LO(OP_V_RD_DRAM, _PROJ_BASE + _STRIDE + _MAT_SIZE);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_K_BIAS);
    SEND_LO(OP_V_RD_DRAM, _PROJ_BASE + 2 * _STRIDE + _MAT_SIZE);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_V_BIAS);
    SEND_LO(OP_V_RD_DRAM, _PROJ_BASE + _MAT_SIZE);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_Q_BIAS);
    SEND_LO(OP_V_RD_DRAM, _PROJ_BASE + 3 * _STRIDE + _MAT_SIZE);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_BO);
    SEND_LO(OP_V_RD_DRAM, _PROJ_BASE + 4 * _STRIDE + _MAT_SIZE);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_BI);
    SEND_LO(OP_V_RD_DRAM, _PROJ_BASE + 5 * _STRIDE + _MAT_SIZE);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_BO2);
    SEND_LO(OP_V_RD_DRAM, _LN1_GAMMA);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_LN1_GAMMA);
    SEND_LO(OP_V_RD_DRAM, _LN1_BETA);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_LN1_BETA);
    SEND_LO(OP_V_RD_DRAM, _LN2_GAMMA);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_LN2_GAMMA);
    SEND_LO(OP_V_RD_DRAM, _LN2_BETA);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, CACHE_LN2_BETA);
}


/* ── proj_single_tile ────────────────────────────────────────────
 *
 * Single-tile projection: W × x + b written straight to DRAM.
 * The input vector streams from DRAM through MV_MUL (no IVRF round
 * trip), the bias is read from the VRF cache, and the biased result
 * is written directly from the pipeline — dropping the per-position
 * V_WR IVRF, accumulator V_WR/V_RD, and save_row_tiles V_RD.
 */
static void proj_single_tile(uint32_t mat_base, uint32_t vec_base,
                             uint32_t bias_cache_off, uint32_t save_addr,
                             uint32_t load_weight)
{
    SEND_LO(OP_V_RD_DRAM, vec_base);
    if (load_weight) {
        SEND_LO(OP_M_RD_DRAM, mat_base);
    }
    SEND_SI(OP_MV_MUL, 0, 0);
    SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, bias_cache_off);
    SEND_SI(OP_VV_ADD, 0, 0);
    SEND_LO(OP_V_WR_DRAM, save_addr);
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
                         uint32_t num_tiles, uint32_t bias_dram_base,
                         uint32_t load_weight)
{
    uint32_t tr, tc;

    /* Force single-tile per M_RD_DRAM: hardware loads 8×8 = 64 elements.
     * In single-tile mode (num_tiles == 1) the layer-start configuration
     * already sets REG_TILE_ROWS/COLS to 1, so re-writing these scalar
     * registers on every call is redundant.  Only the multi-tile path
     * needs to reset them to 1 for each tile load. */
    if (num_tiles > 1) {
        SEND_SI(OP_S_WR, REG_TILE_ROWS_ADDR, 1);
        SEND_SI(OP_S_WR, REG_TILE_COLS_ADDR, 1);
        SEND_SI(OP_S_WR, REG_ITERATIONS_ADDR, 1);
    }

    /* Initialize accumulators: tr=0 uses VRF_ADDSUB_0, tr=1 uses VRF_ADDSUB_2.
     * Only needed in multi-tile mode where the second tile column
     * accumulates on top of the first.  In single-tile mode the result
     * simply overwrites VRF_ADDSUB_0.
     * VRF_ADDSUB_1 (mem 8) is reserved as a cache for X data so that the
     * tile-row loop (and, for K/V/Q projections, subsequent calls for the
     * same position) can read X from VRF instead of DRAM. */
    if (num_tiles > 1) {
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
        SEND_SI(OP_V_RD, MEM_FILL, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_2, 0);
    }

    /* Column-major tile loop: for each tile column, load X chunk once
     * from DRAM.  In single-tile mode the pipeline carries X straight
     * through M_RD_DRAM into MV_MUL, so the ADDSUB_1 cache round-trip
     * is dropped entirely.  In multi-tile mode X is cached in ADDSUB_1
     * and reloaded for tile rows after the first, whose MV_MUL consumes
     * the pipeline. */
    for (tc = 0; tc < num_tiles; tc++) {
        uint32_t vec_chunk_addr = vec_dram_base + tc * NATIVE_DIM;
        SEND_LO(OP_V_RD_DRAM, vec_chunk_addr);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        if (num_tiles > 1) {
            SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
        }

        for (tr = 0; tr < num_tiles; tr++) {
            /* Load weight tile [tr][tc] from DRAM into MRF.
             * M_RD_DRAM loads the MRF directly and preserves the
             * pipeline (X), so no M_WR or X reload is needed.
             * When load_weight == 0 the tile is already resident in
             * MRF (loaded once for position 0 and unchanged since, as
             * nothing else writes MRF between positions), so the load
             * is skipped entirely. */
            uint32_t tile_dram_addr = mat_dram_base + (tr * num_tiles + tc) * MAT_SIZE;
            if (load_weight) {
                SEND_LO(OP_M_RD_DRAM, tile_dram_addr);
            }
            if (tr > 0) {
                /* Reload X from the cache; IVRF must be refreshed for
                 * the MVM-input semantics after the V_RD. */
                SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
                SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
            }

            /* MVM: tile × vec_chunk → pipeline */
            SEND_SI(OP_MV_MUL, 0, 0);

            /* Accumulate into the per-tile-row accumulator:
             * tr=0 → VRF_ADDSUB_0, tr=1 → VRF_ADDSUB_2 */
            uint32_t acc_vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_2;
            if (tc == 0) {
                /* First col-tile: in single-tile mode the result is left
                 * in the pipeline (V_WR keeps it) and the bias add below
                 * reads it straight from there — no store needed.  The
                 * multi-tile path stores to seed the accumulation. */
                if (num_tiles > 1) {
                    SEND_SI(OP_V_WR, acc_vrf, 0);
                }
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
        if (num_tiles == 1) {
            /* Single-tile: the MV_MUL result is still in the pipeline
             * (V_WR keeps it), so the pre-bias V_RD is redundant and
             * dropped.  VV_ADD reads pipe (bias) + vpipe_a (result). */
            SEND_LO(OP_V_RD_DRAM, bias_dram_base + tr * NATIVE_DIM);
            SEND_SI(OP_VV_ADD, 0, 0);
            SEND_SI(OP_V_WR, acc_vrf, 0);
        } else {
            SEND_SI(OP_V_RD, acc_vrf, 0);                              /* load accumulated → vpipe_a */
            SEND_LO(OP_V_RD_DRAM, bias_dram_base + tr * NATIVE_DIM);   /* load bias chunk → pipeline */
            SEND_SI(OP_VV_ADD, 0, 0);                                   /* pipeline = vpipe_a + bias */
            SEND_SI(OP_V_WR, acc_vrf, 0);                               /* store result with bias */
        }
    }

    /* Move tr=1 accumulator from VRF_ADDSUB_2 back to VRF_ADDSUB_1
     * so callers can find results in VRF_ADDSUB_0/1 as before. */
    if (num_tiles > 1) {
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_2, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
    }
}

/* ── mvm_vrf_q ──────────────────────────────────────────────────
 *
 * Single-tile MVM whose input vector already lives in ADDSUB_VRF_0.
 * Equivalent to mvm_tiled_q for num_tiles == 1 but sources the input
 * from VRF instead of DRAM, dropping the SCRATCH_LN1/SCRATCH_GELU
 * store+reload round trips in phase 5.  Only valid for single-tile:
 * the ADDSUB_VRF_1 X-cache slot is not free in multi-tile mode.
 */
static void mvm_vrf_q(uint32_t mat_dram_base, uint32_t bias_dram_base)
{
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);   /* input → pipe */
    SEND_LO(OP_M_RD_DRAM, mat_dram_base);    /* weight tile → MRF */
    SEND_SI(OP_MV_MUL, 0, 0);                /* MRF × input → pipe */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);   /* result → ADDSUB_VRF_0 */
    SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);   /* result → vpipe_a */
    SEND_LO(OP_V_RD_DRAM, bias_dram_base);   /* bias → pipe */
    SEND_SI(OP_VV_ADD, 0, 0);                /* result + bias → pipe */
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);   /* biased result → ADDSUB_VRF_0 */
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

    /* Single-tile fast path: ADDSUB_VRF_1 is spare (only used as the
     * X cache in multi-tile mode), so stage the input tile there while
     * gamma/beta load into IVRF/ADDSUB_VRF_0.  This drops two DRAM
     * round trips per call: the scratch save+reload of the input tile
     * and the tile-0 ln_scratch save/restore that only exists because a
     * second tile's beta load would clobber ADDSUB_VRF_0. */
    if (num_tiles == 1) {
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);   /* input tile → pipe */
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);   /* stage copy in ADDSUB_VRF_1 */
        SEND_LO(OP_V_RD_DRAM, ln_gamma_addr);    /* gamma → pipe */
        SEND_SI(OP_V_WR, 5, 0);                  /* gamma → IVRF */
        SEND_LO(OP_V_RD_DRAM, ln_beta_addr);     /* beta → pipe */
        SEND_SI(OP_V_WR, 7, 0);                  /* beta → ADDSUB_VRF_0 */
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);   /* reload input tile → pipe */
        SEND_SI(OP_V_FUNC, SUB_LAYERNORM, 0);
        SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);   /* normalized → ADDSUB_VRF_0 */
        return;
    }

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
        if (num_tiles == 1) {
            proj_single_tile(k_base, x_base, CACHE_K_BIAS,
                             SAVE_K_BASE + pos * num_tiles * 8,
                             (pos == 0) ? 1 : 0);
        } else {
            /* The K weight tile is loaded into MRF once (position 0) and
             * reused for every subsequent position — nothing else writes
             * MRF in this loop, so the reload is skipped. */
            mvm_tiled_q(k_base, x_base, num_tiles, k_bias, (pos == 0) ? 1 : 0);
            save_row_tiles(num_tiles, SAVE_K_BASE + pos * num_tiles * 8,
                            MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);
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
        if (num_tiles == 1) {
            proj_single_tile(v_base, x_base, CACHE_V_BIAS,
                             SAVE_V_BASE + pos * num_tiles * 8,
                             (pos == 0) ? 1 : 0);
        } else {
            mvm_tiled_q(v_base, x_base, num_tiles, v_bias, (pos == 0) ? 1 : 0);
            save_row_tiles(num_tiles, SAVE_V_BASE + pos * num_tiles * 8,
                            MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);
        }
    }
}

/* ── compute_q_all_positions ───────────────────────────────────
 *
 * Compute Q = Wq × X[pos] + bq for ALL positions and save to DRAM.
 * Q[pos] depends only on X[pos], so all positions can be projected
 * up front (before attention), matching the K/V phases.
 */
static void compute_q_all_positions(
    uint32_t seq_len, uint32_t hidden_size, uint32_t num_tiles,
    uint32_t q_base, uint32_t q_bias)
{
    uint32_t pos;
    for (pos = 0; pos < seq_len; pos++) {
        uint32_t x_base = pos * hidden_size;
        if (num_tiles == 1) {
            proj_single_tile(q_base, x_base, CACHE_Q_BIAS,
                             SAVE_Q_BASE + pos * num_tiles * 8,
                             (pos == 0) ? 1 : 0);
        } else {
            mvm_tiled_q(q_base, x_base, num_tiles, q_bias, (pos == 0) ? 1 : 0);
            save_row_tiles(num_tiles, SAVE_Q_BASE + pos * num_tiles * 8,
                            MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);
        }
    }
}

/* ── attention_scores_all_positions ─────────────────────────────
 *
 * For every query position, compute score = K.T @ Q[pos] and softmax,
 * saving the probability vector to DRAM scratch.
 *
 * The K.T tile for a tile row depends only on the key positions and the
 * tile row (not the query position), so it is built ONCE per tile row and
 * shared across all query positions.  All heads within a tile row share
 * the identical K.T tile and query vector (no instruction below depends
 * on the head), so the score/prob is bit-identical per head and computed
 * once.
 */
static void attention_scores_all_positions(
    uint32_t seq_len, uint32_t hidden_size, uint32_t num_tiles)
{
    uint32_t tr, pos, p, pad;
    for (tr = 0; tr < num_tiles; tr++) {
        /* ── Build K.T MRF tile ── */
        for (p = 0; p < _SEQ_LEN; p++) {
            SEND_LO(OP_V_RD_DRAM, SAVE_K_BASE + p * num_tiles * 8 + tr * 8);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        /* Zero-pad remaining rows to NATIVE_DIM */
        for (pad = _SEQ_LEN; pad < NATIVE_DIM; pad++) {
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        /* Transfer row buffer to MRF */
        SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);

        /* ── Score + softmax for every query position ── */
        for (pos = 0; pos < seq_len; pos++) {
            SEND_LO(OP_V_RD_DRAM, SAVE_Q_BASE + pos * num_tiles * 8 + tr * 8);
            SEND_SI(OP_MV_MUL, 0, 0);
            /* Pipeline: score[i] = K[pos_i, head_h] · Q[head_h] */
            SEND_SI(OP_V_FUNC, SUB_SOFTMAX, 0);
            /* Save prob to DRAM scratch — V.T build overwrites both
             * pipeline and IVRF (via V_RD_DRAM auto-store). */
            SEND_LO(OP_V_WR_DRAM, SCRATCH_ADDR + pos * num_tiles * 8 + tr * 8);
        }
    }
}

/* ── attention_contexts_all_positions ────────────────────────────
 *
 * For every query position, compute context = V.T @ prob[pos] and save
 * it directly to Z scratch (contiguous per position).
 *
 * The V.T tile for a tile row depends only on the tile row, so it is
 * built ONCE per tile row and shared across all query positions.
 */
static void attention_contexts_all_positions(
    uint32_t seq_len, uint32_t hidden_size, uint32_t num_tiles,
    uint32_t num_head)
{
    uint32_t head_size = hidden_size / num_head;
    uint32_t tr, pos, p, pad, j;
    for (tr = 0; tr < num_tiles; tr++) {
        /* ── Build V position-major into MRF ── */
        for (p = 0; p < _SEQ_LEN; p++) {
            SEND_LO(OP_V_RD_DRAM, SAVE_V_BASE + p * num_tiles * 8 + tr * 8);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        for (pad = _SEQ_LEN; pad < NATIVE_DIM; pad++) {
            SEND_SI(OP_V_RD, MEM_FILL, 0);
            SEND_SI(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0);
        }
        SEND_SI(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0);
        /* MRF now has V in position-major order.
         * Re-transpose by extracting columns via unit vectors:
         * MV_MUL(V, e_j) extracts column j of V → pipeline gets
         * [V[0,j], V[1,j], ...], written as row j of V.T. */
        for (j = 0; j < head_size; j++) {
            SEND_LO(OP_V_RD_DRAM, UNIT_VEC_BASE + j * NATIVE_DIM);
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

        /* ── Context for every query position, saved straight to Z ── */
        for (pos = 0; pos < seq_len; pos++) {
            SEND_LO(OP_V_RD_DRAM, SCRATCH_ADDR + pos * num_tiles * 8 + tr * 8);
            SEND_SI(OP_MV_MUL, 0, 0);
            /* Pipeline: context[j] = sum_pos V[pos, head_h, j] * prob[pos].
             * V_WR_DRAM keeps the pipeline; the context is identical for
             * every head in this tile row. */
            SEND_LO(OP_V_WR_DRAM, SCRATCH_Z + pos * hidden_size + tr * NATIVE_DIM);
        }
    }
}

/* ── BERT Encoder Layer ───────────────────────────────────────────
 *
 * Full BERT encoder layer: attention + self-output + FFN.
 * When num_tiles > 1, projections use the multi-tile tile loop.
 *
 * Phase structure:
 *   Phase 1: Compute K and V for all positions
 *   Phase 2: Compute Q for all positions
 *   Phase 3: Attention scores (K.T tile built once per tile row)
 *   Phase 4: Attention contexts (V.T tile built once per tile row)
 *   Phase 5: Per-position self-output, residual, LayerNorm, FFN
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
    /* Vector read/write masks are always 0xFF in this firmware; configure
     * them once here instead of re-issuing the identical S_WR before every
     * K/V/Q load in the attention tile builds. */
    SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);
    SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, 0xFF);

    /* ════════════════════════════════════════════════════════════
     * 2.  FILL + BIAS INIT
     * ════════════════════════════════════════════════════════════ */
    SEND_SI(OP_V_RD, MEM_FILL, 0);
    SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);

    /* Single-tile only: preload biases and LN params into VRF so the
     * per-position loops read the 1-cycle cache instead of DRAM. */
    if (num_tiles == 1) {
        cache_bias_layernorm_params();
    }

    /* ── Phase 1: Compute K and V for all positions ── */
        compute_k_all_positions(seq_len, hidden_size, num_tiles,
            _PROJ_BASE + _STRIDE, _PROJ_BASE + _STRIDE + _MAT_SIZE);
        compute_v_all_positions(seq_len, hidden_size, num_tiles,
            _PROJ_BASE + 2 * _STRIDE, _PROJ_BASE + 2 * _STRIDE + _MAT_SIZE);

        /* ── Phase 2: Compute Q for all positions ── */
        compute_q_all_positions(seq_len, hidden_size, num_tiles,
            _PROJ_BASE, _PROJ_BASE + _MAT_SIZE);

        /* ── Phase 3: Attention scores — K.T tile shared per tile row ── */
        attention_scores_all_positions(seq_len, hidden_size, num_tiles);

        /* ── Phase 4: Attention contexts — V.T tile shared per tile row.
         * Contexts are saved straight to Z scratch (contiguous per
         * position) for the self-output projection. ── */
        attention_contexts_all_positions(seq_len, hidden_size, num_tiles,
                                         num_head);

        /* ── Phase 5: Per-query-position self-output + FFN ── */
        uint32_t _pos;
        for (_pos = 0; _pos < seq_len; _pos++) {
            uint32_t x_base = _pos * hidden_size;
            uint32_t tr;

            if (num_tiles == 1) {
                /* ── Single-tile fast path ─────────────────────────
                 * Pipeline semantics used (V_WR keeps the pipeline;
                 * V_RD moves the previous pipeline into vpipe_a):
                 *   MV_MUL → pipe = W·x
                 *   V_RD bias → vpipe_a = W·x, pipe = bias
                 *   VV_ADD → pipe = W·x + bias
                 * so every pre-bias V_WR/V_RD round trip in mvm_tiled_q
                 * is dropped.  V_FUNC (LN/GELU) leaves its result in the
                 * pipe AND refreshes IVRF, so the following MV_MUL reads
                 * it straight from the pipe.  LN1 gamma/beta are hoisted
                 * to the top of each position (read from the VRF cache)
                 * so the LN1 input never needs a staging round trip;
                 * LN2 gamma/beta are loaded while the GELU result is
                 * staged in ADDSUB_VRF_1.  Each FFN weight M_RD_DRAM is
                 * issued right after the MVM that consumed the previous
                 * MRF so the matrix load overlaps the activation, and
                 * Wo for position _pos+1 is issued at the end of
                 * position _pos to overlap LN2. */

                /* LN1 params: gamma → IVRF, beta → ADDSUB_VRF_0 */
                SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, CACHE_LN1_GAMMA);
                SEND_SI(OP_V_WR, 5, 0);
                SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, CACHE_LN1_BETA);
                SEND_SI(OP_V_WR, 7, 0);

                /* Self-output + residual 1: Z → Wo → +b_o → +X.
                 * Wo is already in MRF (preloaded for position 0 and
                 * hoisted into the previous position's tail for the
                 * rest), so MV_MUL reads the pipe (Z) directly. */
                SEND_LO(OP_V_RD_DRAM, SCRATCH_Z + _pos * hidden_size);
                if (_pos == 0) {
                    SEND_LO(OP_M_RD_DRAM, _PROJ_BASE + 3 * _STRIDE);
                }
                SEND_SI(OP_MV_MUL, 0, 0);
                SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, CACHE_BO);
                SEND_SI(OP_VV_ADD, 0, 0);
                SEND_LO(OP_V_RD_DRAM, x_base);
                SEND_SI(OP_VV_ADD, 0, 0);

                /* LN1 — reads the residual straight from the pipeline */
                SEND_SI(OP_V_FUNC, SUB_LAYERNORM, 0);

                /* FFN1: LN1 result consumed straight from the pipeline */
                SEND_LO(OP_M_RD_DRAM, _PROJ_BASE + 4 * _STRIDE);
                SEND_SI(OP_MV_MUL, 0, 0);
                SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, CACHE_BI);
                SEND_SI(OP_VV_ADD, 0, 0);

                /* GELU reads the FFN1 output from the pipeline */
                SEND_SI(OP_V_GELU, 0, 0);

                /* LN2 params: stage GELU, load gamma → IVRF and
                 * beta → ADDSUB_VRF_0, restore the GELU result. */
                SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);   /* stage GELU input */
                SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, CACHE_LN2_GAMMA);
                SEND_SI(OP_V_WR, 5, 0);
                SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, CACHE_LN2_BETA);
                SEND_SI(OP_V_WR, 7, 0);
                SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);

                /* FFN2: GELU result consumed straight from the pipeline */
                SEND_LO(OP_M_RD_DRAM, _PROJ_BASE + 5 * _STRIDE);
                SEND_SI(OP_MV_MUL, 0, 0);
                SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, CACHE_BO2);
                SEND_SI(OP_VV_ADD, 0, 0);

                /* Residual 2: FFN2 + original X */
                SEND_LO(OP_V_RD_DRAM, x_base);
                SEND_SI(OP_VV_ADD, 0, 0);

                /* LN2 — reads the residual straight from the pipeline */
                SEND_SI(OP_V_FUNC, SUB_LAYERNORM, 0);

                /* Save the LN2 output straight from the pipeline, then
                 * hoist the next position's Wo load so it overlaps LN2. */
                SEND_LO(OP_V_WR_DRAM, SAVE_OUT_BASE + _pos * num_tiles * 8);
                if (_pos + 1 < seq_len) {
                    SEND_LO(OP_M_RD_DRAM, _PROJ_BASE + 3 * _STRIDE);
                }
            } else {
            /* ── Self-output + first residual + LayerNorm ──────── */
            mvm_tiled_q(_PROJ_BASE + 3 * _STRIDE,
                        SCRATCH_Z + _pos * hidden_size, num_tiles,
                        _PROJ_BASE + 3 * _STRIDE + _MAT_SIZE, 1);
            /* Residual 1: SO + X.  VV_ADD reads the pipeline (X) and
             * vpipe_a (SO), so X is streamed straight from DRAM and no
             * SO scratch round-trip is needed.  The original X input is
             * never overwritten, so the residual-2 skip connection reads
             * it from x_base later instead of staging a copy. */
            for (tr = 0; tr < num_tiles; tr++) {
                uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                SEND_SI(OP_V_RD, vrf, 0);
                SEND_LO(OP_V_RD_DRAM, x_base + tr * NATIVE_DIM);
                SEND_SI(OP_VV_ADD, 0, 0);
                SEND_SI(OP_V_WR, vrf, 0);
            }
            apply_layernorm(num_tiles, _LN1_GAMMA, _LN1_BETA, _SCRATCH);
            if (num_tiles == 1) {
                /* Single-tile: FFN1 reads the LN1 result straight from
                 * ADDSUB_VRF_0 — no SCRATCH_LN1 store/reload. */
                mvm_vrf_q(_PROJ_BASE + 4 * _STRIDE,
                          _PROJ_BASE + 4 * _STRIDE + _MAT_SIZE);
            } else {
                /* Save LN1 output to DRAM scratch for FFN Wi input (contiguous) */
                for (tr = 0; tr < num_tiles; tr++) {
                    uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                    SEND_SI(OP_V_RD, vrf, 0);
                    SEND_LO(OP_V_WR_DRAM, SCRATCH_LN1 + tr * NATIVE_DIM);
                }
                mvm_tiled_q(_PROJ_BASE + 4 * _STRIDE, SCRATCH_LN1, num_tiles,
                            _PROJ_BASE + 4 * _STRIDE + _MAT_SIZE, 1);
            }
            SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
            SEND_SI(OP_V_GELU, 0, 0);
            SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);
            if (num_tiles > 1) {
                SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
                SEND_SI(OP_V_GELU, 0, 0);
                SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
            }
            if (num_tiles == 1) {
                /* Single-tile: FFN2 reads the GELU output straight from
                 * ADDSUB_VRF_0 — no SCRATCH_GELU store/reload. */
                mvm_vrf_q(_PROJ_BASE + 5 * _STRIDE,
                          _PROJ_BASE + 5 * _STRIDE + _MAT_SIZE);
            } else {
                /* Save GELU output to DRAM scratch for FFN Wo input (contiguous) */
                for (tr = 0; tr < num_tiles; tr++) {
                    uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                    SEND_SI(OP_V_RD, vrf, 0);
                    SEND_LO(OP_V_WR_DRAM, SCRATCH_GELU + tr * NATIVE_DIM);
                }
                mvm_tiled_q(_PROJ_BASE + 5 * _STRIDE, SCRATCH_GELU, num_tiles,
                            _PROJ_BASE + 5 * _STRIDE + _MAT_SIZE, 1);
            }
            /* Residual 2: FFN2 + original X (skip connection read
             * directly from the untouched input buffer). */
            for (tr = 0; tr < num_tiles; tr++) {
                uint32_t vrf = (tr == 0) ? MEM_ADDSUB_VRF_0 : MEM_ADDSUB_VRF_1;
                SEND_SI(OP_V_RD, vrf, 0);
                SEND_LO(OP_V_RD_DRAM, x_base + tr * NATIVE_DIM);
                SEND_SI(OP_VV_ADD, 0, 0);
                SEND_SI(OP_V_WR, vrf, 0);
            }
            apply_layernorm(num_tiles, _LN2_GAMMA, _LN2_BETA, _SCRATCH);
            save_row_tiles(num_tiles, SAVE_OUT_BASE + _pos * num_tiles * 8,
                            MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);
            }
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
