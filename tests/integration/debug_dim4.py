"""Debug dim4-h4-seq2: compare K/V from batch vs original."""
import os, sys, subprocess
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'emulator'))
from npu_device_mini import NpuDeviceMini, MEM_DRAM

_SEED = 42

def _build_firmware(dim, seq_len=1, num_head=2, hidden_size=None):
    build_dir = f'build_dim{dim}'
    base = Path('firmware')
    hidden = hidden_size if hidden_size is not None else 2 * dim
    proj_base = hidden * seq_len + 4
    mat_size = hidden * hidden
    stride = mat_size + hidden
    num_tiles = hidden // dim
    env = os.environ.copy()
    env.update({
        'BUILD_DIR': build_dir,
        'NATIVE_DIM': str(dim),
        'SEQ_LEN': str(seq_len),
        '_NUM_TILES': str(num_tiles),
        '_HIDDEN_SIZE': str(hidden),
        'P_BIAS': str(proj_base + mat_size),
        'STRIDE': str(stride),
        'MAT_SIZE': str(mat_size),
        'P_Q': str(proj_base),
        'P_K': str(proj_base + stride),
        'P_V': str(proj_base + 2 * stride),
        'P_SO': str(proj_base + 3 * stride),
        'P_FFNI': str(proj_base + 4 * stride),
        'P_FFNO': str(proj_base + 5 * stride),
        'LN1_GAMMA': str(proj_base + 6 * stride),
        'LN1_BETA': str(proj_base + 6 * stride + num_tiles * 8),
        'LN2_GAMMA': str(proj_base + 6 * stride + 2 * num_tiles * 8),
        'LN2_BETA': str(proj_base + 6 * stride + 3 * num_tiles * 8),
        'SCRATCH': str(proj_base + 6 * stride + 4 * num_tiles * 8),
        'NUM_HEAD': str(num_head),
    })
    r = subprocess.run(['make', 'clean', 'all'], capture_output=True, text=True,
                       cwd=str(base), env=env)
    return r, base / build_dir / 'bert.elf'

def _compute_golden_kv(hidden_size, num_head, head_size, seq_len, native_dim, seed=42):
    """Compute golden K and V using Python (matching emulator fp16 precision)."""
    np.random.seed(seed)
    # Generate random X, Wq, Wk, Wv, bq, bk, bv
    X = np.random.randn(seq_len, hidden_size).astype(np.float32)
    Wk = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    Wv = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    bk = np.random.randn(hidden_size).astype(np.float32)
    bv = np.random.randn(hidden_size).astype(np.float32)

    # Compute K and V with fp16 rounding at each step (matching emulator)
    K = np.zeros((seq_len, hidden_size), dtype=np.float32)
    V = np.zeros((seq_len, hidden_size), dtype=np.float32)

    num_tiles = hidden_size // native_dim
    for pos in range(seq_len):
        k_acc = np.zeros(hidden_size, dtype=np.float32)
        v_acc = np.zeros(hidden_size, dtype=np.float32)
        for tr in range(num_tiles):
            for tc in range(num_tiles):
                tr_start = tr * native_dim
                tr_end = tr_start + native_dim
                tc_start = tc * native_dim
                tc_end = tc_start + native_dim

                # Weight tile
                wk_tile = Wk[tr_start:tr_end, tc_start:tc_end]
                wv_tile = Wv[tr_start:tr_end, tc_start:tc_end]
                # X chunk
                x_chunk = X[pos, tc_start:tc_end]

                # MV_MUL
                k_part = wk_tile @ x_chunk
                v_part = wv_tile @ x_chunk

                # fp16 round after MV_MUL
                k_part = np.float16(k_part).astype(np.float32)
                v_part = np.float16(v_part).astype(np.float32)

                if tc == 0:
                    k_acc[tr_start:tr_end] = k_part
                    v_acc[tr_start:tr_end] = v_part
                else:
                    k_acc[tr_start:tr_end] = np.float16(k_acc[tr_start:tr_end] + k_part).astype(np.float32)
                    v_acc[tr_start:tr_end] = np.float16(v_acc[tr_start:tr_end] + v_part).astype(np.float32)

        # Add bias with fp16 round
        k_acc = np.float16(k_acc + bk).astype(np.float32)
        v_acc = np.float16(v_acc + bv).astype(np.float32)
        K[pos] = k_acc
        V[pos] = v_acc

    return X, Wk, Wv, bk, bv, K, V

# ==== Test config ====
dim = 4
hidden = 4
num_head = 4
seq_len = 2
head_size = hidden // num_head

print(f"Config: dim={dim}, hidden={hidden}, num_head={num_head}, seq_len={seq_len}")
print(f"head_size={head_size}, num_tiles={hidden//dim}")

# Build firmware
Path('firmware/build_dim4').mkdir(exist_ok=True)
r, elf = _build_firmware(dim, seq_len, num_head, hidden)
assert r.returncode == 0, f"Build failed: {r.stderr[:500]}\n{r.stdout[:500]}"

# Compute golden
X, Wk, Wv, bk, bv, K_golden, V_golden = _compute_golden_kv(hidden, num_head, head_size, seq_len, dim)

print(f"\nGolden K pos 0: {K_golden[0]}")
print(f"Golden K pos 1: {K_golden[1]}")
print(f"Golden V pos 0: {V_golden[0]}")
print(f"Golden V pos 1: {V_golden[1]}")

# Setup emulator
npu = NpuDeviceMini(native_dim=dim)
npu.set_hidden_size(hidden)
npu.set_seq_len(seq_len)

# Load X
npu._vrf[MEM_DRAM][0:len(X.flatten())] = X.flatten()

# Load weights and biases
_proj_base = hidden * seq_len + 4
_mat_size = hidden * hidden
_stride = _mat_size + hidden

# K weights
k_mat_off = _proj_base + _stride
k_bias_off = _proj_base + _stride + _mat_size
npu._vrf[MEM_DRAM][k_mat_off:k_mat_off + hidden*hidden] = Wk.flatten()
npu._vrf[MEM_DRAM][k_bias_off:k_bias_off + hidden] = bk

# V weights
v_mat_off = _proj_base + 2 * _stride
v_bias_off = _proj_base + 2 * _stride + _mat_size
npu._vrf[MEM_DRAM][v_mat_off:v_mat_off + hidden*hidden] = Wv.flatten()
npu._vrf[MEM_DRAM][v_bias_off:v_bias_off + hidden] = bv

# Q weights (needed for full FW run, even though we only check K/V)
q_mat_off = _proj_base
q_bias_off = _proj_base + _mat_size
Wq = np.random.randn(hidden, hidden).astype(np.float32)
bq = np.random.randn(hidden).astype(np.float32)
npu._vrf[MEM_DRAM][q_mat_off:q_mat_off + hidden*hidden] = Wq.flatten()
npu._vrf[MEM_DRAM][q_bias_off:q_bias_off + hidden] = bq

# Run firmware
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from iss.mini_rv64 import MiniRV64
from emulator.trace_recorder import TraceRecorder
rec = TraceRecorder(npu)
cpu = MiniRV64()
cpu.set_mmio_device(rec)
cpu.load_elf(str(elf))
cpu.run(cycles=80000)
trace = rec.inst_trace
print(f"Instructions executed: {len(trace)}")

# Check for specific opcodes
OP_NAMES_LOCAL = {
    0: "S_WR", 2: "V_RD", 5: "V_WR", 6: "M_WR", 7: "MV_MUL",
    8: "VV_ADD", 11: "VV_MUL", 18: "VV_ADD_INC", 20: "V_RD_DRAM",
    21: "V_WR_DRAM", 22: "V_RD_DRAM_INC", 23: "V_WR_DRAM_INC",
    24: "M_RD_DRAM", 27: "MV_MUL_INC", 42: "V_GELU", 43: "V_FUNC",
    45: "INST_ISSUE",
}
op_counts = {}
for inst in trace:
    op = (inst >> 24) & 0xFF
    op_counts[op] = op_counts.get(op, 0) + 1
print(f"Opcode counts:")
for op, name in sorted(OP_NAMES_LOCAL.items()):
    if op in op_counts:
        print(f"  {name}: {op_counts[op]}")

# Check DRAM stats
ds = npu.get_dram_stats()
total_bytes = (ds['vec_rd_elements'] + ds['vec_wr_elements'] + ds['mat_rd_elements'] + ds['mat_wr_elements']) * 4
print(f"\nDRAM traffic: {total_bytes} bytes")
print(f"  M_RD_DRAM: {ds['mat_rd_ops']} ops ({ds['mat_rd_elements']} el)")
print(f"  V_RD_DRAM: {ds['vec_rd_ops']} ops ({ds['vec_rd_elements']} el)")
print(f"  V_WR_DRAM: {ds['vec_wr_ops']} ops ({ds['vec_wr_elements']} el)")

# Check hidden_size and status
print(f"Hidden: {npu._hidden_size}, SeqLen: {npu._seq_len}")
print(f"Status: {npu._status}")

# Dump all VRF banks
for mem_id in range(12):
    if mem_id in npu._vrf:
        vrf = npu._vrf[mem_id]
        non_zero = np.where(vrf != 0)[0]
        if len(non_zero) > 0:
            print(f"  VRF[{mem_id}] non-zero at: {non_zero[:20]}, values: {vrf[non_zero[:10]]}")

# Read K/V from VRF[6]
vrf6 = npu._vrf[6]

# K at VRF[6] offset 0..seq_len*num_tiles*native_dim
num_tiles = hidden // dim
k_off = 0
v_off = seq_len * num_tiles * dim

print(f"\nVRF[6][0:16]: {vrf6[0:16]}")
print(f"VRF[6][40:48]: {vrf6[40:48]}")

k_emu = np.concatenate([
    vrf6[k_off + p * num_tiles * dim + tr * dim:
         k_off + p * num_tiles * dim + tr * dim + dim]
    for p in range(seq_len)
    for tr in range(num_tiles)
])[:hidden * seq_len]

v_emu = np.concatenate([
    vrf6[v_off + p * num_tiles * dim + tr * dim:
         v_off + p * num_tiles * dim + tr * dim + dim]
    for p in range(seq_len)
    for tr in range(num_tiles)
])[:hidden * seq_len]

print(f"\nK emu pos 0: {k_emu[0:hidden]}")
print(f"K emu pos 1: {k_emu[hidden:2*hidden]}")
print(f"K golden pos 0: {K_golden[0]}")
print(f"K golden pos 1: {K_golden[1]}")

k_diff = np.max(np.abs(k_emu.reshape(seq_len, hidden) - K_golden))
print(f"K max_diff: {k_diff:.6f}")

print(f"\nV emu pos 0: {v_emu[0:hidden]}")
print(f"V emu pos 1: {v_emu[hidden:2*hidden]}")
print(f"V golden pos 0: {V_golden[0]}")
print(f"V golden pos 1: {V_golden[1]}")

v_diff = np.max(np.abs(v_emu.reshape(seq_len, hidden) - V_golden))
print(f"V max_diff: {v_diff:.6f}")
