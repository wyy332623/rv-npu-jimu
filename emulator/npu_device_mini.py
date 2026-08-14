"""
NPU — Standalone MMIO Device (no PySpike dependency).

Reference-aligned emulator backend. Matches NfuEmulator.cpp dispatch.

No pipeline register: values are threaded locally through execute()
calls within a chain group.  Chain boundaries (INST_ISSUE) discard
the pipeline.
"""

from pathlib import Path
import ctypes
import math
import numpy as np


# ── Register offsets (matching firmware/npu_regs.h) ──────────────────
NPU_INST_FIFO     = 0x00
NPU_STATUS        = 0x04
NPU_RESET         = 0x08
NPU_CHAIN_STATUS  = 0x0C
NPU_DATA_IN_ADDR  = 0x10
NPU_DATA_OUT_ADDR = 0x14
NPU_DATA_IN_SIZE  = 0x18
NPU_DATA_OUT_SIZE = 0x1C
NPU_REG_HIDDEN_SIZE = 0x20
NPU_REG_SEQ_LEN     = 0x24
NPU_SKU_ID        = 0xF0
NPU_VERSION       = 0xF4

STATUS_IDLE = 0x00
STATUS_BUSY = 0x01
STATUS_DONE = 0x02

REG_TILE_ROWS      = 1
REG_TILE_COLS      = 2
REG_ITERATIONS     = 3
REG_VECTOR_LENGTH  = 10

# ── Opcodes ──────────────────────────────────────────────────────────
OP_S_WR   = 0;  OP_S_RD   = 1
OP_V_RD   = 2;  OP_M_RD   = 3
OP_V_WR   = 5;  OP_M_WR   = 6
OP_MV_MUL = 7
OP_VV_ADD = 8;  OP_VV_A_SUB_B = 9;  OP_VV_B_SUB_A = 10
OP_VV_MUL = 11
OP_V_SIGM = 12; OP_V_TANH = 13; OP_V_RELU = 14; OP_VV_MAX = 15
OP_V_RD_DRAM = 20; OP_V_WR_DRAM = 21
OP_V_RD_DRAM_INC = 22; OP_V_WR_DRAM_INC = 23
OP_M_RD_DRAM = 24; OP_M_WR_DRAM = 25
OP_V_RD_INC = 16; OP_V_WR_INC = 17
OP_VV_ADD_INC = 18; OP_VV_MAX_INC = 19
OP_MV_MUL_INC = 27
OP_VV_MUL_INC = 31
OP_VV_A_SUB_B_INC = 32; OP_VV_B_SUB_A_INC = 33
OP_VV_MIN = 30
OP_V_EXP  = 37
OP_V_GELU = 42
OP_V_FUNC = 43
OP_SS_ADD = 44
OP_INST_ISSUE = 45
OP_S_RECIP = 35; OP_S_SQRT = 38; OP_SS_MUL = 40

SUB_SOFTMAX   = 0
SUB_LAYERNORM = 1

# ── Memory targets ───────────────────────────────────────────────────
MEM_DRAM            = 0
MEM_MULTIPLY_VRF    = 1
MEM_NET_OUTPUT_Q    = 2
MEM_NET_INPUT_Q     = 3
MEM_MVM_ACC_VRF     = 13
MEM_SPU_ADD_REDUCE   = 14
MEM_SPU_MAX_REDUCE   = 15
MEM_SPU_ABSMAX_REDUCE = 16
MEM_SPU_BROADCAST    = 17
MEM_VEC_TO_MAT_ROW   = 18

MEM_MATRIX_RF       = 4
MEM_MVM_INITIAL_VRF = 5
MEM_MFU_INITIAL_VRF = 6
MEM_ADDSUB_VRF_0    = 7
MEM_ADDSUB_VRF_1    = 8
MEM_ADDSUB_VRF_2    = 19
MEM_FILL            = 12

VRF_SIZES = {
    MEM_MVM_INITIAL_VRF: 20480,
    MEM_MFU_INITIAL_VRF: 4096,
    MEM_ADDSUB_VRF_0:    1024,
    MEM_ADDSUB_VRF_1:    4096,
    MEM_ADDSUB_VRF_2:    64,
    MEM_MULTIPLY_VRF:    64,
    MEM_MVM_ACC_VRF:    256,
    MEM_FILL:              1,
}

NATIVE_DIM_DEFAULT = 128
NATIVE_DIM = NATIVE_DIM_DEFAULT


class NpuDeviceMini:
    """Reference-aligned NPU MMIO device emulator.

    No pipeline register.  Within a chain, execute() threads
    (pipeline, vpipe_a) as local variables.  Chain boundaries
    (INST_ISSUE) discard them.
    """

    def __init__(self, native_dim=None):
        self.native_dim = native_dim or NATIVE_DIM_DEFAULT
        self._hidden_size = self.native_dim
        self._seq_len = 1
        self._spu_srf = np.zeros(64, dtype=np.float32)
        self._dram_stats = {
            'vec_rd_elements': 0, 'vec_wr_elements': 0,
            'mat_rd_elements': 0, 'mat_wr_elements': 0,
            'vec_rd_ops': 0, 'vec_wr_ops': 0,
            'mat_rd_ops': 0, 'mat_wr_ops': 0,
        }
        self._status = STATUS_IDLE
        self._regs = {}
        self._data = bytearray()
        self._dram_addr = 0
        self._chain_busy = 0    # CHAIN_STATUS: bit0=VMM, bit1=MMM, bit2=MVU
        self._cpu_context = {}
        self._source_map = None

        self._vrf = {mem: np.zeros(sz, dtype=np.float32)
                     for mem, sz in VRF_SIZES.items()}
        self._vrf[MEM_DRAM] = np.zeros(524288, dtype=np.float32)
        self._vrf[MEM_NET_OUTPUT_Q] = np.zeros(1024, dtype=np.float32)
        self._vrf[MEM_NET_INPUT_Q] = np.zeros(1024, dtype=np.float32)

        mrf_n = self.native_dim
        self._mrf = {MEM_MATRIX_RF: np.zeros((mrf_n, mrf_n), dtype=np.float32)}
        self._row_buffer = {}
        self._lib = self._load_library()

    def _load_library(self):
        paths = ["libnpukernels.so", "_build/kernels/libnpukernels.so"]
        for p in paths:
            if Path(p).exists():
                lib = ctypes.CDLL(str(Path(p).resolve()))
                self._setup_ctypes(lib)
                return lib
        raise FileNotFoundError("libnpukernels.so not found (build: make kernels)")

    def _setup_ctypes(self, lib):
        LP = ctypes.POINTER(ctypes.c_float)
        for name, argt in {
            'mv_mul':     [LP, LP, LP, ctypes.c_int, ctypes.c_int, ctypes.c_int],
            'gelu':       [LP, LP, ctypes.c_int],
            'relu':       [LP, LP, ctypes.c_int],
            'sigmoid':    [LP, LP, ctypes.c_int],
            'softmax':    [LP, LP, ctypes.c_int],
            'layernorm':  [LP, LP, LP, LP, ctypes.c_int, ctypes.c_float],
            'vec_add':    [LP, LP, LP, ctypes.c_int],
            'vec_sub':    [LP, LP, LP, ctypes.c_int],
            'vec_mul':    [LP, LP, LP, ctypes.c_int],
            'float_to_bfp':  [LP, ctypes.POINTER(ctypes.c_int32),
                              ctypes.POINTER(ctypes.c_int32),
                              ctypes.c_int, ctypes.c_int, ctypes.c_int],
        }.items():
            f = getattr(lib, name, None)
            if f:
                f.argtypes = argt
                f.restype = None

    def _ptr(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def set_hidden_size(self, hs):
        self._hidden_size = hs

    def set_seq_len(self, sl):
        self._seq_len = sl

    def get_dram_stats(self):
        return dict(self._dram_stats)

    def set_cpu_context(self, *, pc=None, cycle=None, inst_count=None):
        """Attach ISS provenance to the next MMIO-triggered NPU command."""
        self._cpu_context = {
            "pc": pc, "cycle": cycle, "inst_count": inst_count,
        }

    def set_source_map(self, source_map):
        """Install a best-effort ELF PC-to-source map for EventTracer."""
        self._source_map = source_map

    NPU_DRAM_BASE = 0x40
    NPU_DRAM_END  = 0x8000
    NPU_SRF_BASE  = 0x8000
    NPU_SRF_END   = 0x8100

    def _dram_offset(self, addr):
        return (addr - self.NPU_DRAM_BASE) // 4

    def _srf_index(self, addr):
        return (addr - self.NPU_SRF_BASE) // 4

    def load(self, addr: int, size: int) -> bytes:
        if addr == NPU_STATUS:
            return self._status.to_bytes(4, 'little')
        elif addr == NPU_CHAIN_STATUS:
            return self._chain_busy.to_bytes(4, 'little')
        elif addr == NPU_DATA_OUT_SIZE:
            return (len(self._data)).to_bytes(4, 'little')
        elif addr == NPU_REG_HIDDEN_SIZE:
            return self._hidden_size.to_bytes(4, 'little')
        elif addr == NPU_REG_SEQ_LEN:
            return self._seq_len.to_bytes(4, 'little')
        elif addr == NPU_SKU_ID:
            return (1).to_bytes(4, 'little')
        elif addr == NPU_VERSION:
            return (0x0500).to_bytes(4, 'little')
        if self.NPU_DRAM_BASE <= addr < self.NPU_DRAM_END:
            off = self._dram_offset(addr)
            dram = self._vrf.get(MEM_DRAM)
            if dram is not None and 0 <= off < len(dram):
                return np.float32(dram[off]).tobytes()
            return b'\x00' * 4
        if self.NPU_SRF_BASE <= addr < self.NPU_SRF_END:
            idx = self._srf_index(addr)
            if 0 <= idx < len(self._spu_srf):
                return np.float32(self._spu_srf[idx]).tobytes()
            return b'\x00' * 4
        return b'\x00' * size

    def store(self, addr: int, data: bytes):
        val = int.from_bytes(data, 'little')
        if addr == NPU_INST_FIFO:
            self._push_instruction(val)
            return
        elif addr == NPU_RESET:
            if val:
                self._reset()
            return
        if self.NPU_DRAM_BASE <= addr < self.NPU_DRAM_END:
            off = self._dram_offset(addr)
            dram = self._vrf.get(MEM_DRAM)
            if dram is not None and 0 <= off < len(dram):
                f32 = np.frombuffer(data, dtype=np.float32)[0]
                dram[off] = f32
            return
        if self.NPU_SRF_BASE <= addr < self.NPU_SRF_END:
            idx = self._srf_index(addr)
            if 0 <= idx < len(self._spu_srf):
                f32 = np.frombuffer(data, dtype=np.float32)[0]
                self._spu_srf[idx] = f32
            return

    def tick(self):
        pass

    def set_data(self, data: bytes):
        self._data = bytearray(data)

    def _reset(self):
        self._regs.clear()
        for mem, sz in VRF_SIZES.items():
            self._vrf[mem] = np.zeros(sz, dtype=np.float32)
        for mem in [MEM_DRAM, MEM_NET_OUTPUT_Q, MEM_NET_INPUT_Q]:
            self._vrf[mem] = np.zeros(len(self._vrf[mem]), dtype=np.float32)
        for mem in self._mrf:
            self._mrf[mem] = np.zeros(self._mrf[mem].shape, dtype=np.float32)
        self._data = bytearray()
        self._status = STATUS_IDLE
        self._chain_busy = 0
        self._row_buffer = {}

    # ── Instruction dispatch ───────────────────────────────────────
    #
    # Pipeline values are local to a chain.  They persist between
    # instructions via _chain_pipeline / _chain_vpipe_a, but are
    # NOT stored as instance attributes for the chain to own.
    # INST_ISSUE discards them.

    def _push_instruction(self, inst: int):
        self._status = STATUS_BUSY
        opcode = (inst >> 24) & 0xFF
        opd0_8  = (inst >> 16) & 0xFF
        opd1_16 = inst & 0xFFFF
        if opcode >= 20:
            opd0, opd1 = opd0_8, opd1_16
        else:
            opd0, opd1 = opd0_8, opd1_16

        pipeline = getattr(self, '_chain_pipeline', None)
        vpipe_a = getattr(self, '_chain_vpipe_a', None)
        pipeline_out, vpipe_a_out = self._execute(
            opcode, opd0, opd1,
            full_operand=(inst & 0xFFFFFF) if opcode >= 20 else 0,
            pipeline=pipeline, vpipe_a=vpipe_a,
        )
        if opcode == OP_INST_ISSUE:
            self._chain_pipeline = None
            self._chain_vpipe_a = None
            self._chain_busy = 0
        else:
            self._chain_pipeline = pipeline_out
            self._chain_vpipe_a = vpipe_a_out
        self._status = STATUS_DONE

    def _execute(self, opcode: int, opd0: int, opd1: int,
                 full_operand: int = 0,
                 pipeline=None, vpipe_a=None):
        """Returns (pipeline_out, vpipe_a_out)."""
        if opcode == OP_S_WR:
            self._s_wr(opcode, opd0, opd1)
            return pipeline, vpipe_a

        inc_map = {
            OP_V_RD_INC: OP_V_RD, OP_V_WR_INC: OP_V_WR,
            OP_V_RD_DRAM_INC: OP_V_RD_DRAM, OP_V_WR_DRAM_INC: OP_V_WR_DRAM,
            OP_MV_MUL_INC: OP_MV_MUL,
            OP_VV_ADD_INC: OP_VV_ADD, OP_VV_MAX_INC: OP_VV_MAX,
            OP_VV_MUL_INC: OP_VV_MUL,
            OP_VV_A_SUB_B_INC: OP_VV_A_SUB_B, OP_VV_B_SUB_A_INC: OP_VV_B_SUB_A,
        }
        base_opcode = inc_map.get(opcode, None)
        if base_opcode is not None:
            inc = opd1
            tr = self._regs.get(1, 1)
            tc = self._regs.get(2, 1)
            iters = self._regs.get(3, 1)
            vec_count = iters * tc
            old_addr = self._dram_addr
            p, v = pipeline, vpipe_a
            for _ in range(vec_count):
                p, v = self._execute(
                    base_opcode, opd0, 0,
                    full_operand=old_addr + inc,
                    pipeline=p, vpipe_a=v,
                )
                old_addr += inc
            self._dram_addr = old_addr
            return p, v

        if opcode == OP_V_RD:
            return self._v_rd(opd0, opd1, pipeline=pipeline, vpipe_a=vpipe_a)
        if opcode == OP_V_RD_DRAM:
            return self._v_rd_dram(full_operand, pipeline=pipeline, vpipe_a=vpipe_a)
        if opcode == OP_M_RD:
            # Only VecToMatRow variant — DRAM variant uses OP_M_RD_DRAM
            return self._m_rd(opd0, full_operand, pipeline=pipeline, vpipe_a=vpipe_a)
        if opcode == OP_M_RD_DRAM:
            self._chain_busy |= 0b0010  # MMM busy
            self._m_rd_dram(full_operand)
            return pipeline, vpipe_a
        if opcode == OP_V_WR:
            self._v_wr(opd0, opd1, pipeline=pipeline)
            return pipeline, vpipe_a  # V_WR broadcasts: writes to target *and* keeps pipeline
        if opcode == OP_V_WR_DRAM:
            self._v_wr_dram(full_operand, pipeline=pipeline)
            return pipeline, vpipe_a  # same: V_WR_DRAM writes *and* keeps pipeline
        if opcode == OP_M_WR:
            return pipeline, vpipe_a
        if opcode == OP_M_WR_DRAM:
            self._m_wr_dram(full_operand)
            return pipeline, vpipe_a
        if opcode == OP_MV_MUL:
            self._chain_busy |= 0b0100  # MVU busy
            return self._mv_mul(pipeline=pipeline), None
        if opcode == OP_VV_MUL:
            self._chain_busy |= 0b0001  # VMM busy
            return self._vv_binop(self._vv_mul_impl, vpipe_a, pipeline), None
        if opcode in (OP_VV_ADD, OP_VV_A_SUB_B, OP_VV_B_SUB_A,
                       OP_VV_MIN, OP_VV_MAX):
            self._chain_busy |= 0b0001  # VMM busy
            return self._vv_add_sub_impl(opcode, vpipe_a, pipeline), None
        if opcode in (OP_V_SIGM, OP_V_TANH, OP_V_RELU, OP_V_GELU, OP_V_EXP):
            self._chain_busy |= 0b0001  # VMM busy
            return self._v_activation(opcode, pipeline), None
        if opcode == OP_V_FUNC:
            self._chain_busy |= 0b0001  # VMM busy
            if opd0 == SUB_SOFTMAX:
                return self._v_softmax(pipeline), None
            elif opd0 == SUB_LAYERNORM:
                return self._v_layernorm(pipeline), None
            return pipeline, vpipe_a
        if opcode == OP_INST_ISSUE:
            return None, None
        if opcode == OP_SS_ADD:
            return pipeline, vpipe_a
        if opcode in (OP_S_RECIP, OP_S_SQRT, OP_SS_MUL):
            # scalar op — no VMM/MMM/MVU busy
            self._spu_func(opcode, opd0, opd1)
            return pipeline, vpipe_a
        return pipeline, vpipe_a

    # ── Opcode handlers ────────────────────────────────────────────
    #
    # All handlers take pipeline/vpipe_a as arguments and return
    # (pipeline_out, vpipe_a_out).  Writes (V_WR, V_WR_DRAM) consume
    # the pipeline and return (None, None).

    def _s_wr(self, opcode, opd0, opd1):
        self._regs[opd0] = opd1

    def _v_rd(self, opd0, opd1, pipeline=None, vpipe_a=None):
        """Vector read: load VRF → pipeline, save old pipeline as vpipe_a."""
        self._chain_busy |= 0b0001  # VMM busy
        mem_target, addr = opd0, opd1
        if mem_target == MEM_FILL:
            val = np.float32(np.frombuffer(
                np.uint16([addr]).tobytes(), dtype=np.float16)[0])
            return np.full(self.native_dim, val, dtype=np.float32), pipeline
        if mem_target == MEM_SPU_BROADCAST:
            val = self._spu_srf[addr] if addr < len(self._spu_srf) else 0.0
            return np.full(self.native_dim, val, dtype=np.float32), pipeline
        vrf = self._vrf.get(mem_target)
        if vrf is not None:
            n = min(self.native_dim, max(0, len(vrf) - addr))
            result = np.zeros(self.native_dim, dtype=np.float32)
            if n > 0:
                result[:n] = np.float16(vrf[addr:addr + n]).astype(np.float32)
            return result, pipeline
        return np.zeros(self.native_dim, dtype=np.float32), pipeline

    def _v_rd_dram(self, full_operand, pipeline=None, vpipe_a=None):
        """Vector read from DRAM: DRAM[addr] → pipeline."""
        self._chain_busy |= 0b0001  # VMM busy
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None:
            addr = full_operand
            n = min(self.native_dim, len(dram) - addr) if addr < len(dram) else 0
            result = np.zeros(self.native_dim, dtype=np.float32)
            if n > 0:
                self._dram_stats['vec_rd_elements'] += n
                self._dram_stats['vec_rd_ops'] += 1
                rmask = self._regs.get(15, 0xFF)
                for i in range(n):
                    if (rmask >> (i % 8)) & 1:
                        result[i] = np.float16(dram[addr + i]).astype(np.float32)
            return result, pipeline
        return np.zeros(self.native_dim, dtype=np.float32), pipeline

    def _v_wr(self, opd0, opd1, pipeline=None):
        """Vector write: pipeline → VRF (or row buffer / SPU reduce)."""
        if pipeline is None:
            return
        self._chain_busy |= 0b0001  # VMM busy
        mem_target, addr = opd0, opd1
        if mem_target == MEM_VEC_TO_MAT_ROW:
            key = 0
            if key not in self._row_buffer:
                self._row_buffer[key] = []
            self._row_buffer[key].append(pipeline.copy())
            return
        if mem_target == MEM_SPU_ADD_REDUCE:
            if addr < len(self._spu_srf):
                self._spu_srf[addr] = float(np.sum(pipeline)) + self._spu_srf[addr]
            return
        if mem_target in (MEM_SPU_MAX_REDUCE, MEM_SPU_ABSMAX_REDUCE):
            if addr < len(self._spu_srf):
                data = pipeline if mem_target == MEM_SPU_MAX_REDUCE else np.abs(pipeline)
                self._spu_srf[addr] = max(float(np.max(data)), self._spu_srf[addr])
            return
        vrf = self._vrf.setdefault(mem_target,
                                    np.zeros(self.native_dim * 8, dtype=np.float32))
        wmask = self._regs.get(16, 0xFF)
        n = min(self.native_dim, max(0, len(vrf) - addr))
        if n > 0:
            for i in range(n):
                if (wmask >> (i % 8)) & 1:
                    vrf[addr + i] = np.float16(pipeline[i])

    def _v_wr_dram(self, full_operand, pipeline=None):
        """Vector write to DRAM: pipeline → DRAM[addr]."""
        if pipeline is None:
            return
        self._chain_busy |= 0b0001  # VMM busy
        addr = full_operand
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None:
            n = min(self.native_dim, len(pipeline), len(dram) - addr) if addr < len(dram) else 0
            if n > 0:
                self._dram_stats['vec_wr_elements'] += n
                self._dram_stats['vec_wr_ops'] += 1
                dram[addr:addr + n] = np.float16(pipeline[:n])

    def _m_rd(self, opd0, full_operand, pipeline=None, vpipe_a=None):
        """Matrix read from row buffer (VecToMatRow) into MRF."""
        self._chain_busy |= 0b0010  # MMM busy
        key = 0
        rows = self._row_buffer.get(key, [])
        n = self.native_dim
        mat = np.zeros((n, n), dtype=np.float32)
        for i, row in enumerate(rows[:n]):
            mat[i, :len(row)] = row[:n]
        self._mrf[MEM_MATRIX_RF] = mat.copy()
        self._row_buffer[key] = []
        return pipeline, vpipe_a

    def _m_rd_dram(self, full_operand):
        """Matrix read from DRAM into MRF."""
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None:
            addr = full_operand
            if addr < len(dram):
                n = self._regs.get(REG_TILE_ROWS, 1) * self.native_dim
                mrf_size = n * n
                if addr + mrf_size <= len(dram):
                    self._dram_stats['mat_rd_elements'] += mrf_size
                    self._dram_stats['mat_rd_ops'] += 1
                    mat = np.float16(dram[addr:addr + mrf_size]).astype(np.float32).reshape(n, n).copy()
                    self._mrf[MEM_MATRIX_RF] = mat

    def _m_wr_dram(self, full_operand):
        """Matrix write: MRF → DRAM."""
        mrf = self._mrf.get(MEM_MATRIX_RF)
        if mrf is not None:
            flat = mrf.flatten()
            addr = full_operand
            dram = self._vrf.get(MEM_DRAM)
            if dram is not None and addr + len(flat) <= len(dram):
                self._dram_stats['mat_wr_elements'] += len(flat)
                self._dram_stats['mat_wr_ops'] += 1
                dram[addr:addr + len(flat)] = np.float16(flat)

    # ── Compute helpers ────────────────────────────────────────────

    @staticmethod
    def _round_fp16(arr):
        return arr.astype(np.float16).astype(np.float32) if arr is not None else None

    def _store_to_ivrf(self, pipeline):
        if pipeline is not None:
            p = self._round_fp16(pipeline)
            ivrf = self._vrf.get(MEM_MVM_INITIAL_VRF)
            if ivrf is not None:
                n = min(len(ivrf), len(p))
                ivrf[:n] = p[:n]

    def _mv_mul(self, pipeline=None):
        """MRF × pipeline → result."""
        mrf = self._mrf.get(MEM_MATRIX_RF)
        if mrf is not None and pipeline is not None:
            mrfMask = self._regs.get(17, 0xFF)
            n_rows = mrf.shape[0]
            mask_dim = max(1, n_rows // 8)
            masked_mrf = mrf.copy()
            for bit_idx in range(8):
                if not (mrfMask >> bit_idx) & 1:
                    sr = bit_idx * mask_dim
                    er = min(sr + mask_dim, n_rows)
                    if sr < n_rows:
                        masked_mrf[sr:er, :] = 0.0
            rows, cols = masked_mrf.shape
            n = min(len(pipeline), cols)
            result = np.zeros(rows, dtype=np.float32)
            self._lib.mv_mul(self._ptr(masked_mrf), self._ptr(pipeline),
                             self._ptr(result), rows, n, 0)
            return self._round_fp16(result)
        return pipeline

    def _vv_binop(self, impl_fn, a, b):
        if a is not None and b is not None:
            n = min(len(a), len(b))
            r = np.zeros(n, dtype=np.float32)
            impl_fn(a, b, r, n)
            return self._round_fp16(r)
        return b if b is not None else a

    def _vv_mul_impl(self, a, b, r, n):
        self._lib.vec_mul(self._ptr(a), self._ptr(b), self._ptr(r), n)

    def _vv_add_sub_impl(self, opcode, a, b):
        if a is not None and b is not None:
            n = min(len(a), len(b))
            r = np.zeros(n, dtype=np.float32)
            if opcode == OP_VV_ADD:
                self._lib.vec_add(self._ptr(a), self._ptr(b), self._ptr(r), n)
            elif opcode == OP_VV_A_SUB_B:
                self._lib.vec_sub(self._ptr(a), self._ptr(b), self._ptr(r), n)
            elif opcode == OP_VV_B_SUB_A:
                self._lib.vec_sub(self._ptr(b), self._ptr(a), self._ptr(r), n)
            elif opcode == OP_VV_MAX:
                self._lib.vec_max(self._ptr(a), self._ptr(b), self._ptr(r), n)
            else:
                self._lib.vec_add(self._ptr(a), self._ptr(b), self._ptr(r), n)
            return self._round_fp16(r)
        return b if b is not None else a

    def _v_activation(self, opcode, pipeline):
        if pipeline is not None:
            n = len(pipeline)
            r = np.zeros(n, dtype=np.float32)
            if opcode == OP_V_GELU:
                self._lib.gelu(self._ptr(pipeline), self._ptr(r), n)
            elif opcode == OP_V_RELU:
                self._lib.relu(self._ptr(pipeline), self._ptr(r), n)
            elif opcode == OP_V_SIGM:
                self._lib.sigmoid(self._ptr(pipeline), self._ptr(r), n)
            elif opcode == OP_V_TANH:
                r[:] = np.tanh(pipeline)
            elif opcode == OP_V_EXP:
                r[:] = np.exp(pipeline)
            result = self._round_fp16(r)
            self._store_to_ivrf(result)
            return result
        return pipeline

    def _v_softmax(self, pipeline):
        if pipeline is not None:
            n = len(pipeline)
            r = np.zeros(n, dtype=np.float32)
            self._lib.softmax(self._ptr(pipeline), self._ptr(r), n)
            result = self._round_fp16(r)
            self._store_to_ivrf(result)
            return result
        return pipeline

    def _v_layernorm(self, pipeline):
        if pipeline is not None:
            n = len(pipeline)
            gamma = self._vrf.get(5, np.ones(n, dtype=np.float32))[:n]
            beta = self._vrf.get(7, np.zeros(n, dtype=np.float32))[:n]
            r = np.zeros(n, dtype=np.float32)
            self._lib.layernorm(self._ptr(pipeline), self._ptr(gamma),
                                self._ptr(beta), self._ptr(r), n, 1e-12)
            result = self._round_fp16(r)
            self._store_to_ivrf(result)
            return result
        return pipeline

    def _spu_func(self, opcode, opd0, opd1):
        # scalar op — no VMM/MMM/MVU busy
        src = opd1 & 0x3F
        dst = opd0 & 0x3F
        val = self._spu_srf[src] if src < len(self._spu_srf) else 0.0
        if opcode == OP_S_RECIP:
            result = 1.0 / val if val != 0.0 else float('inf')
        elif opcode == OP_S_SQRT:
            result = math.sqrt(val) if val >= 0.0 else float('nan')
        elif opcode == OP_SS_MUL:
            dst_val = self._spu_srf[dst] if dst < len(self._spu_srf) else 0.0
            result = dst_val * val
        else:
            result = 0.0
        if dst < len(self._spu_srf):
            self._spu_srf[dst] = float(result)

    def snapshot(self):
        """Return NPU state (no pipeline — it's local to the chain)."""
        return {
            'dram_addr': self._dram_addr,
            'regs': dict(self._regs),
            'spu_srf': self._spu_srf.copy(),
            'vrf': {mem: arr.copy() for mem, arr in self._vrf.items()},
            'mrf': {mem: arr.copy() for mem, arr in self._mrf.items()},
        }
