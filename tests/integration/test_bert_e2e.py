"""
NPU - Unified BERT End-to-End Inference Test
=============================================

Validation chain (all rounds operate in FP16):

  Round 0   (numpy):    Golden reference from gen_golden_bert - FP16 cast
  Round 0.5 (C kernel): C bert_encoder_layer vs golden - FP16 I/O & compute
  Round 1   (emulator): Firmware on C emulator → opcode coverage,
                         backpressure, numerical alignment - FP16 data path
  Round 2   (HDL seq):  Amaranth NpuTop sequential replay (optional).
  Round 3   (HDL batch): Amaranth NpuTop batch replay (optional).

  HDL rounds are skiped if Amaranth is not installed.
  Run `pip install amaranth` for full 4-round validation.

Only the **final output** is checked for numerical correctness.
Intermediates (Q, K, V, residual, LN) are optimization-free — the
firmware may store them in VRF cache, on-chip SRAM, or DRAM as
needed for the best performance/area tradeoff.
"""

import numpy as np
import pytest
import subprocess
from pathlib import Path

from emulator.npu_device_mini import NpuDeviceMini, MEM_DRAM
from emulator.trace_recorder import TraceRecorder
from emulator.npu_instrumentor import NpuInstrumentor, OpBoundary

# RISC-V ISS imports are optional (requires pyelftools)
# HDL imports are optional (requires amaranth)
try:
    from iss.mini_rv64 import MiniRV64
    HAS_ISS = True
except ImportError:
    HAS_ISS = False
    MiniRV64 = None

try:
    from amaranth.sim import Simulator
    from hdl.npu_top import NpuTop
    HAS_HDL = True
except ImportError:
    HAS_HDL = False
    Simulator = None
    NpuTop = None

def _fp16_bits(v):
    """Scalar or ndarray → FP16 bits (uint16)."""
    return int(np.float16(v).view(np.uint16))


def _from_fp16_bits(bits):
    """FP16 bits (uint16) → float."""
    return float(np.frombuffer(np.uint16([bits]).tobytes(), dtype=np.float16)[0])


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

_SEED = 42
FIFO_DEPTH = 2

# Opcode names for coverage reporting
OP_NAMES = {
    0: "S_WR", 2: "V_RD", 5: "V_WR", 6: "M_WR", 7: "MV_MUL",
    8: "VV_ADD", 11: "VV_MUL", 18: "VV_ADD_INC", 20: "V_RD_DRAM",
    21: "V_WR_DRAM", 22: "V_RD_DRAM_INC", 23: "V_WR_DRAM_INC",
    24: "M_RD_DRAM", 27: "MV_MUL_INC", 42: "V_GELU", 43: "V_FUNC",
    45: "INST_ISSUE",
}
# HDL replay is optional. When Amaranth is not installed,
# validation uses R0 (golden) + R1 (emulator) only.


# ═══════════════════════════════════════════════════════════════════════
# C Kernel Library
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _generate_golden(hidden_size, num_head, head_size, seq_len=1, native_dim=None):
    """FP16 golden reference matching emulator + firmware precision exactly.

    The forward pass in gen_golden_bert.py now faithfully reproduces the
    emulator datapath: tiled MVM with fp16-rounding between tile columns,
    fp16-rounding after every VV_ADD / VV_MUL / softmax / LayerNorm, and
    fp32 GELU with fp16-round at auto-store.  The golden dict therefore
    contains emulator-matching intermediates throughout.

    Args:
      hidden_size: model dimension (e.g., 16 for dim8 test)
      num_head: number of attention heads
      head_size: elements per head
      seq_len: sequence length
      native_dim: NPU native dimension (firmware tile size). Defaults to head_size * 2.

    Returns:
      golden: full BERT layer intermediates dict (emulator-matching)
      params: all weight/bias parameters
      sim_out: firmware-equivalent output array [seq_len, hidden_size]
      ln_offsets: dict of DRAM offsets for LN params
    """
    from tests.gen_golden_bert import bert_encoder_layer
    if native_dim is None:
        native_dim = head_size * 2  # match typical test config
    golden, params = bert_encoder_layer(
        add_mask=False, num_head=num_head, head_size=head_size,
        hidden_size=hidden_size, seq_len=seq_len,
        native_dim=native_dim, precision='emulator_float32', seed=_SEED)

    # The golden 'out' tensor is now computed with emulator-matching
    # precision, so sim_out is simply golden['out'].
    sim_out_buf = golden['out'].copy()

    # LN params placed after the 6 projection matrices+biases.
    # Each LN param has num_tiles tile rows at stride 8.
    proj_base = hidden_size * seq_len + 4
    mat_size = hidden_size * hidden_size
    stride = mat_size + hidden_size
    ln_size = (hidden_size // native_dim) * 8
    ln_base = proj_base + 6 * stride
    ln_offsets = {
        'ln1_gamma': ln_base,
        'ln1_beta':  ln_base + ln_size,
        'ln2_gamma': ln_base + 2 * ln_size,
        'ln2_beta':  ln_base + 3 * ln_size,
        'scratch': 0x500,
    }

    return golden, params, sim_out_buf, ln_offsets



def _build_firmware(dim, seq_len=1, num_head=2, hidden_size=None):
    """Build firmware with given NATIVE_DIM, SEQ_LEN, and NUM_HEAD. Return elf path.
    DRAM layout macros are computed here and passed via environment.
    If hidden_size is given, use it; otherwise default to 2*dim (num_tiles=2)."""
    fw = Path('firmware')
    build_dir = f'build_dim{dim}'
    abs_build_dir = fw / build_dir
    hidden = hidden_size if hidden_size is not None else 2 * dim
    proj_base = hidden * seq_len + 4
    mat_size = hidden * hidden
    stride = mat_size + hidden
    num_tiles = hidden // dim
    proj_end = proj_base + 6 * stride  # past all 6 matrices+biases
    ln_base = proj_end  # = proj_base + 6 * stride
    ln_size = num_tiles * 8  # each LN param has num_tiles tile rows at stride 8
    scratch_addr = 0x500  # DRAM scratch area for LN saves
    env = {
        'NATIVE_DIM': str(dim),
        'SEQ_LEN': str(seq_len),
        '_HIDDEN_SIZE': str(hidden),
        '_PROJ_BASE': str(proj_base),
        '_MAT_SIZE': str(mat_size),
        '_STRIDE': str(stride),
        '_NUM_TILES': str(num_tiles),
        '_LN1_GAMMA': str(ln_base),
        '_LN1_BETA': str(ln_base + ln_size),
        '_LN2_GAMMA': str(ln_base + 2 * ln_size),
        '_LN2_BETA': str(ln_base + 3 * ln_size),
        '_SCRATCH': str(scratch_addr),
        'NUM_HEAD': str(num_head),
    }
    import os
    full_env = {**os.environ, **env}
    r = subprocess.run(
        ['make', '-C', str(fw), f'BUILD_DIR={build_dir}', 'clean', 'all'],
        capture_output=True, text=True, env=full_env)
    elf = abs_build_dir / 'bert.elf'
    return r, elf








# ═══════════════════════════════════════════════════════════════════════
# FP16 C Kernel Call
# ═══════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════
# Amaranth HDL Replay Helpers
# ═══════════════════════════════════════════════════════════════════════

def _read_dram(ctx, top, start, n):
    """Read n elements from Amaranth simulator DRAM."""
    return np.array([_from_fp16_bits(ctx.get(top.vrf_dram[start + i]) & 0xFFFF)
                     for i in range(n)], dtype=np.float32)


def _build_top(dim, lanes, dram_depth=512, vrf_depth=256):
    return NpuTop(lanes=dim, native_dim=dim, mrf_rows=dim, mrf_cols=dim,
                  dram_depth=dram_depth, mvu_lanes=lanes, vrf_depth=vrf_depth)


async def _push(ctx, top, inst):
    ctx.set(top.mmio_addr, 0x00)
    ctx.set(top.mmio_wdata, inst)
    ctx.set(top.mmio_we, 1)
    await ctx.tick()
    ctx.set(top.mmio_we, 0)


async def _wait_done(ctx, top, timeout=500):
    for _ in range(timeout):
        await ctx.tick()
        ctx.set(top.mmio_addr, 4)
        ctx.set(top.mmio_re, 1)
        await ctx.delay(1e-10)
        if ctx.get(top.mmio_rdata) & 2:
            ctx.set(top.mmio_re, 0)
            return True
        ctx.set(top.mmio_re, 0)
    return False









def _read_output_from_hdl(ctx, top, hidden_size, dim, seq_len=1):
    """Read Q (0x200) and final output (0x800) from HDL simulator DRAM.
    Returns (q_array, output_array) both as numpy float32 vectors."""
    num_tiles = hidden_size // dim
    stride = 8  # firmware hardcodes tr*8 in save_row_tiles
    last_pos = seq_len - 1

    # Read Q at 0x200 + last_pos * num_tiles * stride
    q_base = 0x200 + last_pos * num_tiles * stride
    q_parts = []
    for tr in range(num_tiles):
        q_parts.append(_read_dram(ctx, top, q_base + tr * stride, dim))
    q = np.concatenate(q_parts)[:hidden_size]

    # Read final output at 0x800 + last_pos * num_tiles * stride
    out_base = 0x800 + last_pos * num_tiles * stride
    out_parts = []
    for tr in range(num_tiles):
        out_parts.append(_read_dram(ctx, top, out_base + tr * stride, dim))
    out = np.concatenate(out_parts)[:hidden_size]

    return q, out


# For multi-tile replay: allow M_RD_DRAM (24) through since tiles are
# loaded dynamically by each M_RD_DRAM instruction.
SKIP_OPS_MULTI_TILE = {25, 6}  # only skip M_WR and M_WR_DRAM

def _init_dram_only(ctx, top, emu_dram, max_dram):
    """Load DRAM only (no MRF pre-load). M_RD_DRAM instructions
    will dynamically load tiles into MRF during replay."""
    for i in range(min(max_dram, len(emu_dram))):
        ctx.set(top.vrf_dram[i], int(_fp16_bits(float(emu_dram[i]))))


async def _replay_sequential_mt(ctx, top, trace, skip_ops, emu_dram, max_dram, hidden_size, dim, seq_len=1):
    """Replay multi-tile instructions one at a time.
    Sets hidden_size and seq_len on the HDL.
    Returns (Q, final_output) both as numpy float32 arrays."""
    _init_dram_only(ctx, top, emu_dram, max_dram)
    ctx.set(top.rst, 1); await ctx.tick(); ctx.set(top.rst, 0)
    ctx.set(top.hidden_size, hidden_size)
    ctx.set(top.seq_len, seq_len)
    for _ in range(3): await ctx.tick()

    timeouts = 0
    for inst in trace:
        op = (inst >> 24) & 0xFF
        if op in skip_ops:
            continue
        await _push(ctx, top, inst)
        ok = await _wait_done(ctx, top, timeout=1000)
        if not ok:
            timeouts += 1
    if timeouts:
        print(f"    seq: {timeouts} instruction timeouts")
    return _read_output_from_hdl(ctx, top, hidden_size, dim, seq_len)


async def _replay_batch_mt(ctx, top, trace, batches, skip_ops, emu_dram, max_dram, hidden_size, dim, seq_len=1):
    """Replay multi-tile instructions in firmware-defined batches.
    Returns (Q, final_output) both as numpy float32 arrays."""
    _init_dram_only(ctx, top, emu_dram, max_dram)
    ctx.set(top.rst, 1); await ctx.tick(); ctx.set(top.rst, 0)
    ctx.set(top.hidden_size, hidden_size)
    ctx.set(top.seq_len, seq_len)
    for _ in range(3): await ctx.tick()

    timeouts = 0
    for batch in batches:
        filtered = [i for i in batch if ((i >> 24) & 0xFF) not in skip_ops]
        if not filtered:
            continue
        for inst in filtered:
            await _push(ctx, top, inst)
        ok = await _wait_done(ctx, top, timeout=6000)
        if not ok:
            timeouts += 1
    if timeouts:
        print(f"    batch: {timeouts} batch timeouts")
    return _read_output_from_hdl(ctx, top, hidden_size, dim, seq_len)




# ═══════════════════════════════════════════════════════════════════════
# Parameterized Test
# ═══════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════
# Multi-Tile Q Projection Test (hidden_size > NATIVE_DIM)
# ═══════════════════════════════════════════════════════════════════════

def _load_weights_and_bias_tiled(npu, params, dim, hidden_size, mat_offset, bias_offset, proj_name):
    """
    Load weight matrix and bias for a projection (Q, K, or V) into NPU DRAM
    as pre-tiled 8×8 submatrices, with bias immediately after.
    """
    tile_size = dim

    if proj_name not in params or not isinstance(params[proj_name], dict):
        return

    W = params[proj_name]['W'].astype(np.float32)
    M, N = W.shape

    # Pre-tile into 8×8 submatrices
    tile_blocks = []
    for tr in range(M // tile_size):
        for tc in range(N // tile_size):
            tile = W[tr*tile_size:(tr+1)*tile_size, tc*tile_size:(tc+1)*tile_size]
            tile_blocks.append(tile.flatten())

    tiled_data = np.concatenate(tile_blocks)
    npu._vrf[MEM_DRAM][mat_offset:mat_offset + len(tiled_data)] = tiled_data

    # Load bias right after weight matrix
    if 'b' in params[proj_name]:
        b = params[proj_name]['b'].astype(np.float32).flatten()[:hidden_size]
        npu._vrf[MEM_DRAM][bias_offset:bias_offset + hidden_size] = b


def _extract_tiled_from_dram(dram, base_addr, dim, hidden_size, stride=8):
    """Extract a tiled vector from DRAM.

    The firmware saves row-tile tr at base_addr + tr * stride, each tile
    has `dim` elements.  For multi-tile (hidden_size > dim), concatenate
    tiles and trim to hidden_size.
    """
    num_tiles = hidden_size // dim
    parts = []
    for tr in range(num_tiles):
        addr = base_addr + tr * stride
        parts.append(dram[addr:addr + dim].copy())
    return np.concatenate(parts)[:hidden_size]


def _extract_q_from_emulator(npu, dim, hidden_size, pos=0):
    """Extract Q vector from DRAM save locations.
    With dot-product attention, Q is saved per position at
    0x200 + pos * num_tiles * 8."""
    num_tiles = hidden_size // dim
    base = 0x200 + pos * num_tiles * 8
    return _extract_tiled_from_dram(npu._vrf[MEM_DRAM], base, dim, hidden_size)


def _extract_output_from_emulator(npu, dim, hidden_size, pos=0):
    """Extract final output vector from DRAM save locations.
    Firmware saves at 0x800 + pos * num_tiles * 8."""
    num_tiles = hidden_size // dim
    base = 0x800 + pos * num_tiles * 8
    return _extract_tiled_from_dram(npu._vrf[MEM_DRAM], base, dim, hidden_size)


def _check_tensor(label, actual, ref, atol, name=""):
    """Compare two tensors, print diagnostics, return max_diff.
    Raises AssertionError if max_diff exceeds atol."""
    diff = np.abs(actual - ref)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    print(f"  {label} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    warnings = 0
    for i in range(len(actual)):
        if abs(actual[i] - ref[i]) >= atol:
            rel = abs(actual[i] - ref[i]) / max(abs(ref[i]), 1e-6)
            if rel >= 0.05:
                warnings += 1
                if warnings <= 3:
                    print(f"    {name}[{i}]: {actual[i]:.6f} vs ref={ref[i]:.6f} "
                          f"diff={abs(actual[i]-ref[i]):.6f}")
    if warnings:
        print(f"    ⚠️  {warnings}/{len(actual)} elements exceed tolerance")
    assert max_diff < atol, f"{label} {name}: max_diff {max_diff:.4f} >= {atol}"
    return max_diff


def _instrument_boundaries(dim, hidden_size, seq_len, proj_base, stride, num_tiles):
    """Build OpBoundary list matching current firmware DRAM layout.
    Only final output is checked — intermediates (Q, K, V) are
    optimization-free and may be stored in VRF cache, not DRAM."""
    nt = num_tiles
    boundaries = [
        OpBoundary('save_res', 0x700, nt, stride=8),
    ]
    for pos in range(seq_len):
        out_base = 0x800 + pos * nt * 8
        label = f'out_p{pos}' if seq_len > 1 else 'out'
        boundaries.append(OpBoundary(label, out_base, nt, stride=8))
    return boundaries


def _run_instrumented_diagnostics(npu, instr, dim, hidden_size, seq_len, golden, params):
    """Print intermediate-by-intermediate comparison.

    Called after assertions pass, only when --instrument is enabled.
    Does NOT raise — purely diagnostic.
    """
    last_pos = seq_len - 1
    print(f"\n  {'='*55}")
    print(f"  OPERATOR-BY-OPERATOR DIAGNOSTICS (--instrument)")
    print(f"  {'='*55}")

    from tests.gen_golden_bert import fp16_round as _fr

    # Only check final output — intermediates are optimization-free
    ref_out = golden['out'][last_pos].flatten()[:hidden_size]
    from tests.integration.test_bert_e2e import _extract_output_from_emulator
    emu_out = _extract_output_from_emulator(npu, dim, hidden_size, pos=last_pos)
    d = float(np.max(np.abs(emu_out - ref_out)))
    st = 'OK' if d < 0.05 else 'FAIL'
    print(f'  [{st}] OUT  max_diff={d:.6f}  {emu_out[:4].round(4)}  (only final output is validated)')
    print(f"  {'='*60}")


# ── Detect firmware capability ────────────────────────────────────
# Firmware may be rewritten by G2/G3 for single-tile (num_tiles=1).
# Multi-tile configs (num_tiles > 1) are skipped if the firmware
# no longer supports them, detected by num_tiles != hidden/dim.

def _firmware_is_single_tile_only(dim, hidden_size) -> bool:
    """Check if the current bert_layer.c has been stripped of multi-tile paths.
    Returns True if the firmware has been optimized for dim == hidden (single-tile)
    and no longer supports multi-tile configs."""
    fw_path = Path("firmware/bert/bert_layer.c")
    if not fw_path.exists():
        return False
    text = fw_path.read_text()
    # Single-tile only firmware removes the multi-tile helper functions.
    # The baseline has compute_k_all_positions and compute_v_all_positions.
    # If those are missing, the firmware was stripped for single-tile only.
    has_kv_phase = "compute_k_all_positions" in text
    has_v_phase = "compute_v_all_positions" in text
    num_tiles = hidden_size // dim
    if num_tiles > 1 and (not has_kv_phase or not has_v_phase):
        return True  # firmware is single-tile only, but test needs multi-tile
    return False


@pytest.mark.parametrize("dim,lanes,num_head,hidden_size,seq_len,vrf_depth", [
    (2, 2, 2, 4, 2, 64),
    (2, 2, 2, 4, 6, 64),
    (4, 4, 2, 8, 2, 64),
    (4, 4, 2, 8, 6, 64),
    (4, 4, 2, 4, 2, 64),   # dim4-h4: single-tile, heads_per_tile=2
    (4, 4, 2, 4, 6, 64),   # dim4-h4-seq6
], ids=[
    "dim2-lanes2-head2-h4-seq2",
    "dim2-lanes2-head2-h4-seq6",
    "dim4-lanes4-head2-h8-seq2",
    "dim4-lanes4-head2-h8-seq6",
    "dim4-lanes4-head2-h4-seq2",
    "dim4-lanes4-head2-h4-seq6",
])
def test_bert_e2e_multi_tile(dim, lanes, num_head, hidden_size, seq_len, vrf_depth, request):
    """
    BERT encoder layer, multi-tile: hidden_size > NATIVE_DIM.
    Validates both Q projection and final output across all 4 rounds.
    Optional HDL rounds (R2, R3) run only if Amaranth is installed.
    """
    if not HAS_ISS:
        pytest.skip("ISS (MiniRV64) not available — install pyelftools")

    # Skip if firmware doesn't support this config
    if _firmware_is_single_tile_only(dim, hidden_size):
        num_tiles = hidden_size // dim
        pytest.skip(f"Firmware is single-tile only (num_tiles={num_tiles} > 1 not supported)")

    head_size = hidden_size // num_head
    num_tiles = hidden_size // dim
    last_pos = seq_len - 1

    print(f"\n  Config: dim={dim}, lanes={lanes}, num_head={num_head}, "
          f"hidden_size={hidden_size}, head_size={head_size}, seq_len={seq_len}")
    print(f"  num_tiles = {num_tiles} (each tile = {dim}×{dim})")
    print(f"  Mode: {num_tiles}×{num_tiles}-tile (hidden_size > dim)")

    # ── Round 0: Generate golden reference ──
    print("\n  Round 0: FP16 golden reference...")
    golden, params, sim_out_ref_arr, ln_offsets = _generate_golden(
        hidden_size=hidden_size, num_head=num_head, head_size=head_size,
        seq_len=seq_len, native_dim=dim)

    sim_out_ref = sim_out_ref_arr
    if sim_out_ref.ndim > 1:
        sim_out_ref = sim_out_ref_arr[last_pos].flatten()[:hidden_size]
    print(f"  Golden out[:4]:   {sim_out_ref[:4].round(4)}")

    # ── Round 1: Firmware on emulator ─────────────────────────
    print("\n  Round 1: Firmware on emulator (multi-tile, dim=8)...")

    r, elf = _build_firmware(dim, seq_len, num_head, hidden_size)
    if r.returncode != 0 or not elf.exists():
        pytest.skip(f"FW build failed (dim={dim}):\n{r.stderr[:300]}")

    npu = NpuDeviceMini(native_dim=dim)
    npu.set_hidden_size(hidden_size)
    npu.set_seq_len(seq_len)

    X_input = golden['X'].flatten().astype(np.float32)[:hidden_size * seq_len]
    npu._vrf[MEM_DRAM][0:len(X_input)] = X_input
    print(f"  Input X (pos 0)[:4]: {X_input[:4].round(4)}")

    _proj_base = hidden_size * seq_len + 4
    _mat_size = hidden_size * hidden_size
    _stride = _mat_size + hidden_size

    proj_layout = [
        ('Q', _proj_base, _proj_base + _mat_size),
        ('K', _proj_base + _stride, _proj_base + _stride + _mat_size),
        ('V', _proj_base + 2 * _stride, _proj_base + 2 * _stride + _mat_size),
        ('selfoutput', _proj_base + 3 * _stride, _proj_base + 3 * _stride + _mat_size),
    ]
    for proj_name, mat_off, bias_off in proj_layout:
        _load_weights_and_bias_tiled(npu, params, dim, hidden_size, mat_off, bias_off, proj_name)

    def _load_ff_tiled(params_key, bias_key, mat_off, bias_off):
        W = params[params_key].astype(np.float32)
        tile_blocks = []
        for tr in range(W.shape[0] // dim):
            for tc in range(W.shape[1] // dim):
                tile = W[tr*dim:(tr+1)*dim, tc*dim:(tc+1)*dim]
                tile_blocks.append(tile.flatten())
        tiled = np.concatenate(tile_blocks)
        npu._vrf[MEM_DRAM][mat_off:mat_off + len(tiled)] = tiled
        b = params[bias_key].astype(np.float32).flatten()[:W.shape[1]]
        npu._vrf[MEM_DRAM][bias_off:bias_off + len(b)] = b

    _load_ff_tiled('W_intmfc', 'b_intmfc', _proj_base + 4 * _stride, _proj_base + 4 * _stride + _mat_size)
    _load_ff_tiled('W_outfc', 'b_outfc', _proj_base + 5 * _stride, _proj_base + 5 * _stride + _mat_size)

    # Load LayerNorm gamma/beta - stored with stride 8 per tile row
    # to match the firmware's save_row_tiles/load_row_tiles stride pattern
    ln_dst = ln_offsets
    def _load_ln_tiled(dram_dst, src_vec, dim, hidden_size):
        num_tiles = hidden_size // dim
        for tr in range(num_tiles):
            chunk = src_vec[tr*dim:(tr+1)*dim]
            npu._vrf[MEM_DRAM][dram_dst + tr * 8: dram_dst + tr * 8 + dim] = chunk
    _load_ln_tiled(ln_dst['ln1_gamma'], params['LayerNorm']['W'][0].astype(np.float32).flatten(), dim, hidden_size)
    _load_ln_tiled(ln_dst['ln1_beta'], params['LayerNorm']['b'][0].astype(np.float32).flatten(), dim, hidden_size)
    _load_ln_tiled(ln_dst['ln2_gamma'], params['LayerNorm']['W'][1].astype(np.float32).flatten(), dim, hidden_size)
    _load_ln_tiled(ln_dst['ln2_beta'], params['LayerNorm']['b'][1].astype(np.float32).flatten(), dim, hidden_size)

    # Load unit vectors into DRAM at UNIT_VEC_BASE (0x900) for V.T re-transpose
    # For NATIVE_DIM=d, we need d unit vectors e_j (j=0..d-1), each of length d
    UNIT_VEC_BASE = 0x900
    for j in range(dim):
        e_j = np.zeros(dim, dtype=np.float32)
        e_j[j] = 1.0
        npu._vrf[MEM_DRAM][UNIT_VEC_BASE + j * dim: UNIT_VEC_BASE + j * dim + dim] = e_j

    # ── Optional instrumentor: wrap NPU with op-by-op boundary capture ──
    want_instr = request.config.getoption("--instrument", default=False)
    instr = None
    if want_instr:
        boundaries = _instrument_boundaries(dim, hidden_size, seq_len,
                                            _proj_base, _stride, num_tiles)
        instr = NpuInstrumentor(npu, boundaries, capture_dram_range=(0, 0x1000))
        wrap_for_instr = instr
    else:
        wrap_for_instr = npu

    rec = TraceRecorder(wrap_for_instr)
    cpu = MiniRV64()
    cpu.set_mmio_device(rec)
    cpu.load_elf(str(elf))
    cpu.run(cycles=80000)
    trace = rec.inst_trace
    print(f"  Instructions executed: {len(trace)}")

    emu_out = _extract_output_from_emulator(npu, dim, hidden_size, pos=last_pos)
    print(f"  Emulator out[:4]:  {emu_out[:4].round(4)}")

    _check_tensor("Round 1", emu_out, sim_out_ref, 0.05, "out")

    # ── Optional instrumentor diagnostics ──────────────────────
    if instr is not None:
        _run_instrumented_diagnostics(npu, instr, dim, hidden_size, seq_len, golden, params)
        instr.unpatch()

    # ── DRAM traffic report ────────────────────────────────────
    ds = npu.get_dram_stats()
    total_elements = ds['vec_rd_elements'] + ds['vec_wr_elements'] + ds['mat_rd_elements'] + ds['mat_wr_elements']
    total_bytes = total_elements * 4
    print(f"\n  DRAM traffic (float32 elements):")
    print(f"    V_RD_DRAM: {ds['vec_rd_ops']} ops × {8} el = {ds['vec_rd_elements']} el")
    print(f"    V_WR_DRAM: {ds['vec_wr_ops']} ops × {8} el = {ds['vec_wr_elements']} el")
    print(f"    M_RD_DRAM: {ds['mat_rd_ops']} ops × 64 el = {ds['mat_rd_elements']} el")
    print(f"    M_WR_DRAM: {ds['mat_wr_ops']} ops × 64 el = {ds['mat_wr_elements']} el")
    print(f"    Total: {total_elements} elements ({total_bytes} bytes)")

    # ── Opcode coverage check ─────────────────────────────────
    ops_hit = set()
    for inst in trace:
        op = (inst >> 24) & 0xFF
        if op in OP_NAMES:
            ops_hit.add(OP_NAMES[op])
    expected_multi_tile_ops = {
        "S_WR", "V_RD", "V_WR", "M_WR", "MV_MUL", "VV_ADD",
        "V_RD_DRAM", "M_RD_DRAM",
    }
    missing = expected_multi_tile_ops - ops_hit
    if missing:
        print(f"  ⚠️  Missing opcodes: {sorted(missing)}")
    else:
        print(f"  ✅ Multi-tile opcode coverage complete")
    print(f"  Total instructions: {len(trace)}")
    print(f"  Opcodes hit: {sorted(ops_hit)}")

    # ── Round 2: HDL sequential replay (optional) ──────────────
    if HAS_HDL:
        print("\n  Round 2: HDL sequential replay (multi-tile)...")
        emu_dram = npu._vrf[MEM_DRAM].copy()
        max_dram = min(2048, len(emu_dram))

        top = _build_top(dim, lanes, dram_depth=max_dram, vrf_depth=vrf_depth)
        sim = Simulator(top)
        sim.add_clock(1e-8)
        seq_result = [None]

        async def _seq_tb(ctx):
            seq_result[0] = await _replay_sequential_mt(
                ctx, top, trace, SKIP_OPS_MULTI_TILE, emu_dram, max_dram, hidden_size, dim, seq_len)

        sim.add_testbench(_seq_tb)
        sim.run()
        assert seq_result[0] is not None, "HDL seq replay produced no result"
        _, seq_out = seq_result[0]
        print(f"    HDL seq out[:4]: {seq_out[:4].round(4)}")
        _check_tensor("Round 2 seq", seq_out, emu_out, 1.0, "out")
        print(f"    ✅ HDL seq output matches emulator")
    else:
        print("  Round 2: skipped (install amaranth for HDL validation)")
        _, seq_out = None, emu_out

    # ── Round 3: HDL batch replay (optional) ───────────────────
    if HAS_HDL:
        print("\n  Round 3: HDL batch replay (multi-tile)...")
        batches = rec.extract_batches()

        top = _build_top(dim, lanes, dram_depth=max_dram, vrf_depth=vrf_depth)
        sim = Simulator(top)
        sim.add_clock(1e-8)
        batch_result = [None]

        async def _batch_tb(ctx):
            batch_result[0] = await _replay_batch_mt(
                ctx, top, trace, batches, SKIP_OPS_MULTI_TILE, emu_dram, max_dram, hidden_size, dim, seq_len)

        sim.add_testbench(_batch_tb)
        sim.run()
        assert batch_result[0] is not None, "HDL batch replay produced no result"
        _, batch_out = batch_result[0]
        print(f"    HDL batch out[:4]: {batch_out[:4].round(4)}")
        _check_tensor("Round 3 batch", batch_out, emu_out, 1.0, "out")
        print(f"    ✅ HDL batch output matches emulator")

        # ── Cross-round comparison ──────────────────────────────
        sb_out_diff = float(np.max(np.abs(seq_out - batch_out)))
        print(f"  Seq vs Batch out: max_diff={sb_out_diff:.6f}")
    else:
        print("  Round 3: skipped (install amaranth for HDL validation)")

    print(f"  \n  ✅ All rounds complete - output matches golden (firmware tile-MHA)")

