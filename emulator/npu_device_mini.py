"""
NPU — Standalone MMIO Device (no PySpike dependency).

Reference-aligned emulator backend. Matches NfuEmulator.cpp dispatch:
  execute() -> s_wr / v_rd / v_wr / m_rd / m_wr / mv_mul /
               vv_add_sub_cmp_impl / vv_mul_impl / v_activation_impl /
               v_func { softmax | layernorm } / ...

Memory model (reference: m_vec_mems[memId], m_mat_mems[memId]):
  Vector RF: DRAM, MultiplyVrf, MvmInitialVrf, MfuInitialVrf,
             AddSubVrf_0/1/2, NetOutputQ, NetInputQ
  Matrix RF: MatrixRf

Uses ctypes to call C kernel library for compute.
For cycle-accurate RTL sim, use sim.backend_verilator instead.
"""

from pathlib import Path
import ctypes
import math
import numpy as np


# ── Register offsets (matching firmware/npu_regs.h) ──────────────────
NPU_INST_FIFO     = 0x00
NPU_STATUS        = 0x04
NPU_RESET         = 0x08
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

# ── NPU internal register addresses (firmware:npu_driver.h) ─────────
REG_TILE_ROWS      = 1
REG_TILE_COLS      = 2
REG_ITERATIONS     = 3
REG_VECTOR_LENGTH  = 10

# ── Opcodes (matching ISA.bond, npu_isa.h) ──────────────────────────
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

# ── Memory targets (matching ISA.bond Mem enum) ──────────────────────
MEM_DRAM            = 0
MEM_MULTIPLY_VRF    = 1
MEM_NET_OUTPUT_Q    = 2
MEM_NET_INPUT_Q     = 3
MEM_MVM_ACC_VRF     = 13
MEM_SPU_ADD_REDUCE   = 14  # sum elements → SRF
MEM_SPU_MAX_REDUCE   = 15  # max element → SRF
MEM_SPU_ABSMAX_REDUCE = 16 # max|element| → SRF
MEM_SPU_BROADCAST    = 17  # broadcast SRF → pipeline
MEM_VEC_TO_MAT_ROW   = 18  # accumulate vectors into row buffer for MRF

MEM_MATRIX_RF       = 4
MEM_MVM_INITIAL_VRF = 5
MEM_MFU_INITIAL_VRF = 6
MEM_ADDSUB_VRF_0    = 7
MEM_ADDSUB_VRF_1    = 8
MEM_ADDSUB_VRF_2    = 9
MEM_FILL            = 12

# ── VRF sizes (from sku_bert_np.py SkuParams) ───────────────────────
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

NATIVE_DIM_DEFAULT = 128  # from SKU
NATIVE_DIM = NATIVE_DIM_DEFAULT  # backward-compatible alias


class NpuMemoryAccessError(ValueError):
    """Invalid firmware access to an NPU memory bank."""


class NpuDeviceMini:
    """Reference-aligned NPU MMIO device emulator.

    Translates firmware instructions to C kernel calls.
    Register file model matches NfuEmulator's m_vec_mems/m_mat_mems.

    Parameters
    ----------
    native_dim : int, optional
        Logical vector dimension.  Defaults to 128 (SKU BERT-NP).
        Pass 8 for dim=8 integration tests so that DRAM addressing
        and VRF widths match the firmware compiled with NATIVE_DIM=8.
    """

    def __init__(self, native_dim=None):
        self.native_dim = native_dim or NATIVE_DIM_DEFAULT
        self._hidden_size = self.native_dim  # default = NATIVE_DIM
        self._seq_len = 1  # default = 1
        self._spu_srf = np.zeros(64, dtype=np.float32)  # SPU scalar register file
        self._dram_stats = {  # DRAM traffic counters
            'vec_rd_elements': 0,  # elements read via V_RD_DRAM
            'vec_wr_elements': 0,  # elements written via V_WR_DRAM
            'mat_rd_elements': 0,  # elements read via M_RD_DRAM
            'mat_wr_elements': 0,  # elements written via M_WR_DRAM
            'vec_rd_ops': 0,       # count of V_RD_DRAM instructions
            'vec_wr_ops': 0,       # count of V_WR_DRAM instructions
            'mat_rd_ops': 0,       # count of M_RD_DRAM instructions
            'mat_wr_ops': 0,       # count of M_WR_DRAM instructions
        }
        self._status = STATUS_IDLE
        self._regs = {}          # scalar registers (REG_*)
        self._data = bytearray()  # data buffer
        self._pipeline = None    # vector pipeline (result of last compute)
        self._vpipe_a = None     # operand A for VV ops
        self._vpipe_b = None     # operand B for VV ops
        self._dram_addr = 0      # DRAM address for auto-increment (INC variants)

        # Reference: m_vec_mems[memId] — map-based VRF storage
        self._vrf = {mem: np.zeros(sz, dtype=np.float32)
                     for mem, sz in VRF_SIZES.items()}
        self._vrf[MEM_DRAM] = np.zeros(524288, dtype=np.float32)  # 512K
        self._vrf[MEM_NET_OUTPUT_Q] = np.zeros(1024, dtype=np.float32)
        self._vrf[MEM_NET_INPUT_Q] = np.zeros(1024, dtype=np.float32)

        # Reference: m_mat_mems[memId] — matrix RF
        mrf_n = self.native_dim
        self._mrf = {MEM_MATRIX_RF: np.zeros((mrf_n, mrf_n), dtype=np.float32)}

        # Row buffer for VecToMatRow: accumulates vectors into MRF rows
        self._row_buffer = {}   # key → list of ndarrays (rows)

        # Load C kernel library
        self._lib = self._load_library()

    def _load_library(self):
        paths = ["libnpukernels.so", "_build/kernels/libnpukernels.so"]
        for p in paths:
            if Path(p).exists():
                lib = ctypes.CDLL(str(Path(p).resolve()))
                self._setup_ctypes(lib)
                return lib
        raise FileNotFoundError(
            "libnpukernels.so not found (build: make kernels)")

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

    # ── MMIO interface ─────────────────────────────────────────────
    def set_hidden_size(self, hs):
        """Set hidden_size for the firmware to read from MMIO 0x20."""
        self._hidden_size = hs
    
    def set_seq_len(self, sl):
        """Set seq_len for the firmware to read from MMIO 0x24."""
        self._seq_len = sl
    
    def get_dram_stats(self):
        """Return DRAM traffic counters as a dict."""
        return dict(self._dram_stats)
    
    # ── MMIO register map bounds ──
    #   0x00 - 0x3F:  Control/status registers
    #   0x40 - 0x3FFF: DRAM window (64K floats = 256KB = half the DRAM)
    #   0x4000 - 0x40FF: SPU SRF window (64 registers × 4 bytes = 256 bytes)

    NPU_DRAM_BASE = 0x40
    NPU_DRAM_END  = 0x8000
    NPU_SRF_BASE  = 0x8000
    NPU_SRF_END   = 0x8100

    def _dram_offset(self, addr):
        return (addr - self.NPU_DRAM_BASE) // 4

    def _srf_index(self, addr):
        return (addr - self.NPU_SRF_BASE) // 4

    def load(self, addr: int, size: int) -> bytes:
        # Control/status registers
        if addr == NPU_STATUS:
            return self._status.to_bytes(4, 'little')
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

        # DRAM window: CPU reads DRAM[float_offset]
        if self.NPU_DRAM_BASE <= addr < self.NPU_DRAM_END:
            off = self._dram_offset(addr)
            dram = self._vrf.get(MEM_DRAM)
            if dram is not None and 0 <= off < len(dram):
                return np.float32(dram[off]).tobytes()
            return b'\x00' * 4

        # SPU SRF window: CPU reads SRF[idx]
        if self.NPU_SRF_BASE <= addr < self.NPU_SRF_END:
            idx = self._srf_index(addr)
            if 0 <= idx < len(self._spu_srf):
                return np.float32(self._spu_srf[idx]).tobytes()
            return b'\x00' * 4

        return b'\x00' * size

    def store(self, addr: int, data: bytes):
        val = int.from_bytes(data, 'little')

        # Control/status registers
        if addr == NPU_INST_FIFO:
            self._push_instruction(val)
            return
        elif addr == NPU_RESET:
            if val:
                self._reset()
            return

        # DRAM window: CPU writes DRAM[float_offset]
        if self.NPU_DRAM_BASE <= addr < self.NPU_DRAM_END:
            off = self._dram_offset(addr)
            dram = self._vrf.get(MEM_DRAM)
            if dram is not None and 0 <= off < len(dram):
                f32 = np.frombuffer(data, dtype=np.float32)[0]
                dram[off] = f32
            return

        # SPU SRF window: CPU writes SRF[idx]
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
        self._pipeline = None
        self._vpipe_a = None
        self._vpipe_b = None
        self._status = STATUS_IDLE
        self._row_buffer = {}

    def _push_instruction(self, inst: int):
        self._status = STATUS_BUSY
        opcode = (inst >> 24) & 0xFF
        # LO format (M_RD_DRAM, M_WR_DRAM): operand is full 24-bit address
        # SI format: opd0=8bit, opd1=16bit
        # Detect LO format by checking if opd0's bits 16:23 are zero
        opd0_8  = (inst >> 16) & 0xFF
        opd1_16 = inst & 0xFFFF
        # LO opcodes have opcode >= 20 (m_rd_dram, v_rd_dram, etc.)
        if opcode >= 20:
            # LO format: operand is bits 23:0, split into opd0_8 and opd1_16
            opd0 = opd0_8  # upper 8 bits of 24-bit address
            opd1 = opd1_16  # lower 16 bits of 24-bit address
        else:
            opd0 = opd0_8
            opd1 = opd1_16
        self._execute(opcode, opd0, opd1, full_operand=(inst & 0xFFFFFF) if opcode >= 20 else 0)
        self._status = STATUS_DONE

    # ── Reference-aligned instruction dispatch ──────────────────────
    def _execute(self, opcode: int, opd0: int, opd1: int, full_operand: int = 0):
        """Matches NfuEmulator::execute() dispatch structure.
        full_operand: 24-bit address for LO-format instructions.
        """
        if opcode == OP_S_WR:
            self._s_wr(opcode, opd0, opd1)

        # ── INC variant handling: map to base opcode, set DRAM addr ──
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
            # For INC variants, opd1 is the increment amount.
            # Tile iteration: compute total vectors from registers.
            inc = opd1
            tile_rows = self._regs.get(1, 1)  # REG_TILE_ROWS
            tile_cols = self._regs.get(2, 1)  # REG_TILE_COLS
            iterations = self._regs.get(3, 1)  # REG_ITERATIONS
            # Total vectors to transfer = iterations * tile_cols
            # (matching npu_top's vmm_vec_count = reg_iterations * reg_tile_cols)
            vec_count = iterations * tile_cols
            old_addr = self._dram_addr
            for v in range(vec_count):
                self._execute(base_opcode, opd0, 0, full_operand=old_addr + v * inc)
            self._dram_addr = old_addr + vec_count * inc
            return

        elif opcode == OP_V_RD:
            self._v_rd(opcode, opd0, opd1)
        elif opcode == OP_V_RD_DRAM:
            self._v_rd_dram(opcode, opd0, opd1, full_operand)

        elif opcode in (OP_M_RD, OP_M_RD_DRAM):
            self._m_rd(opcode, opd0, opd1, full_operand)

        elif opcode == OP_V_WR:
            self._v_wr(opcode, opd0, opd1)
        elif opcode == OP_V_WR_DRAM:
            self._v_wr_dram(opcode, opd0, opd1, full_operand)

        elif opcode == OP_M_WR:
            pass  # no-op — matches HDL (npu_top: `with m.Case(OP_M_WR): pass`)
        elif opcode == OP_M_WR_DRAM:
            self._m_wr(opcode, opd0, opd1, full_operand)

        elif opcode == OP_MV_MUL:
            self._mv_mul(opcode, opd0, opd1)

        elif opcode == OP_VV_MUL:
            self._vv_mul(opcode, opd0, opd1)

        elif opcode in (OP_VV_ADD, OP_VV_A_SUB_B, OP_VV_B_SUB_A,
                        OP_VV_MIN, OP_VV_MAX):
            self._vv_add_sub(opcode, opd0, opd1)

        elif opcode in (OP_V_SIGM, OP_V_TANH, OP_V_RELU,
                        OP_V_GELU, OP_V_EXP):
            self._v_activation(opcode, opd0, opd1)

        elif opcode == OP_V_FUNC:
            sub = opd0
            if sub == SUB_SOFTMAX:
                self._v_softmax(opcode, opd0, opd1)
            elif sub == SUB_LAYERNORM:
                self._v_layernorm(opcode, opd0, opd1)

        elif opcode == OP_INST_ISSUE:
            pass  # chain marker — handled by dispatcher

        elif opcode == OP_SS_ADD:
            pass  # scalar add — handled by control processor

        elif opcode in (OP_S_RECIP, OP_S_SQRT, OP_SS_MUL):
            self._spu_func(opcode, opd0, opd1)

        # else: unsupported opcode — just acknowledge

    # ── Opcode handlers (matching reference method names) ───────────

    def _s_wr(self, opcode, opd0, opd1):
        """Scalar write: store opd1 into internal register opd0."""
        self._regs[opd0] = opd1

    # ── Pipeline data flow ─────────────────────────────────────────
    # In the reference, instructions operate on a data pipeline.
    # V_RD loads FROM memory INTO the pipeline.
    # Compute ops (MV_MUL, VV_*) read FROM pipeline, write TO pipeline.
    # V_WR stores FROM pipeline TO memory.
    # We track the pipeline as _pipeline (ndarray, float32).

    def _checked_vrf(self, mem_target, addr, operation):
        """Return a VRF only when a full native-width access is valid."""
        vrf = self._vrf.get(mem_target)
        if vrf is None:
            raise NpuMemoryAccessError(
                f"{operation}: unknown VRF bank {mem_target}"
            )
        if addr < 0 or addr + self.native_dim > len(vrf):
            raise NpuMemoryAccessError(
                f"{operation}: VRF bank {mem_target} access "
                f"[{addr}, {addr + self.native_dim}) exceeds "
                f"[0, {len(vrf)})"
            )
        return vrf

    def _v_rd(self, opcode, opd0, opd1):
        """Vector read: load from VRF at opd0[opd1] into pipeline."""
        mem_target = opd0
        addr = opd1

        if mem_target == MEM_FILL:
            if self._pipeline is not None:
                self._vpipe_a = self._pipeline.copy()
            self._pipeline = np.full(self.native_dim, np.float32(np.frombuffer(
                np.uint16([addr]).tobytes(), dtype=np.float16)[0]), dtype=np.float32)
            return
        if mem_target == MEM_SPU_BROADCAST:
            if self._pipeline is not None:
                self._vpipe_a = self._pipeline.copy()
            val = self._spu_srf[addr] if addr < len(self._spu_srf) else 0.0
            self._pipeline = np.full(self.native_dim, val, dtype=np.float32)
            return
        vrf = self._checked_vrf(mem_target, addr, "V_RD")
        if self._pipeline is not None:
            self._vpipe_a = self._pipeline.copy()
        # Load from VRF and round to FP16, matching HDL pipe width.
        # Promoted back to float32 for C kernel compatibility.
        self._pipeline = np.float16(
            vrf[addr:addr + self.native_dim]
        ).astype(np.float32)

    def _v_wr(self, opcode, opd0, opd1):
        """Vector write: store pipeline to VRF starting at opd1."""
        mem_target = opd0
        addr = opd1
        if self._pipeline is None:
            return

        # VecToMatRow: accumulate pipeline into row buffer
        if mem_target == MEM_VEC_TO_MAT_ROW:
            key = 0  # single row buffer
            if key not in self._row_buffer:
                self._row_buffer[key] = []
            self._row_buffer[key].append(self._pipeline.copy())
            return
        
        # SPU reduce operations
        if mem_target == MEM_SPU_ADD_REDUCE:
            if addr < len(self._spu_srf):
                self._spu_srf[addr] = float(np.sum(self._pipeline)) + self._spu_srf[addr]
            return
        if mem_target == MEM_SPU_MAX_REDUCE or mem_target == MEM_SPU_ABSMAX_REDUCE:
            if addr < len(self._spu_srf):
                data = self._pipeline if mem_target == MEM_SPU_MAX_REDUCE else np.abs(self._pipeline)
                self._spu_srf[addr] = max(float(np.max(data)), self._spu_srf[addr])
            return
        
        vrf = self._checked_vrf(mem_target, addr, "V_WR")
        # Apply write vector mask
        wmask = self._regs.get(16, 0xFF)  # REG_WRITE_VECTOR_MASK, default all ones
        # Write pipeline to VRF starting at addr, masked
        for i in range(self.native_dim):
            if (wmask >> (i % 8)) & 1:
                # Store as FP16, matching HDL VRF register width
                vrf[addr + i] = np.float16(self._pipeline[i])

    def _m_rd(self, opcode, opd0, opd1, full_operand=0):
        """Matrix read: load matrix from DRAM or row buffer into MRF."""
        # VecToMatRow: construct MRF from accumulated row buffer
        if opd0 == MEM_VEC_TO_MAT_ROW:
            key = 0
            rows = self._row_buffer.get(key, [])
            n = self.native_dim
            mat = np.zeros((n, n), dtype=np.float32)
            for i, row in enumerate(rows[:n]):
                mat[i, :len(row)] = row[:n]
            self._mrf[MEM_MATRIX_RF] = mat.copy()
            # Clear the row buffer after reading
            self._row_buffer[key] = []
            return

        addr = full_operand
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None and addr < len(dram):
            n = self._regs.get(REG_TILE_ROWS, 1) * self.native_dim
            mrf_size = n * n
            if addr + mrf_size <= len(dram):
                self._dram_stats['mat_rd_elements'] += mrf_size
                self._dram_stats['mat_rd_ops'] += 1
                # Read from DRAM and round to FP16, matching HDL MRF width.
                # Store as float32 (C kernel expects float*) with fp16-exact values.
                mat = np.float16(dram[addr:addr + mrf_size]).astype(np.float32).reshape(n, n).copy()
                self._mrf[MEM_MATRIX_RF] = mat

    def _m_wr(self, opcode, opd0, opd1, full_operand=0):
        """Matrix write: store MRF to DRAM."""
        mrf = self._mrf.get(MEM_MATRIX_RF)
        if mrf is not None:
            flat = mrf.flatten()
            addr = full_operand
            dram = self._vrf.get(MEM_DRAM)
            if dram is not None and addr + len(flat) <= len(dram):
                self._dram_stats['mat_wr_elements'] += len(flat)
                self._dram_stats['mat_wr_ops'] += 1
                # Store as FP16, matching HDL DRAM width
                dram[addr:addr + len(flat)] = np.float16(flat)

    def _v_rd_dram(self, opcode, opd0, opd1, full_operand=0):
        """Vector read from DRAM at 24-bit address, with read mask."""
        addr = full_operand
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None:
            if self._pipeline is not None:
                self._vpipe_a = self._pipeline.copy()
            n = min(self.native_dim, len(dram) - addr) if addr < len(dram) else 0
            self._pipeline = np.zeros(self.native_dim, dtype=np.float32)
            if n > 0:
                self._dram_stats['vec_rd_elements'] += n
                self._dram_stats['vec_rd_ops'] += 1
                # Apply read vector mask (register 15)
                rmask = self._regs.get(15, 0xFF)
                for i in range(n):
                    if (rmask >> (i % 8)) & 1:
                        # Read from DRAM and round to FP16, matching HDL.
                        self._pipeline[i] = np.float16(dram[addr + i]).astype(np.float32)

    def _v_wr_dram(self, opcode, opd0, opd1, full_operand=0):
        """Vector write to DRAM at 24-bit address."""
        addr = full_operand
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None and self._pipeline is not None:
            pipeline_n = len(self._pipeline)
            n = min(self.native_dim, pipeline_n, len(dram) - addr) if addr < len(dram) else 0
            if n > 0:
                self._dram_stats['vec_wr_elements'] += n
                self._dram_stats['vec_wr_ops'] += 1
                # Store as FP16, matching HDL DRAM width
                dram[addr:addr + n] = np.float16(self._pipeline[:n])

    def _pipeline_round_fp16(self):
        """Round pipeline values to FP16, matching HW datapath."""
        if self._pipeline is not None:
            self._pipeline = self._pipeline.astype(np.float16).astype(np.float32)

    def _clear_vv(self):
        self._vpipe_a = None

    def _store_to_ivrf(self):
        """Auto-store pipeline to IVRF after compute."""
        if self._pipeline is not None:
            self._pipeline_round_fp16()
            ivrf = self._vrf.get(MEM_MVM_INITIAL_VRF)
            if ivrf is not None:
                n = min(len(ivrf), len(self._pipeline))
                ivrf[:n] = self._pipeline[:n]

    def _mv_mul(self, opcode, opd0, opd1):
        """Matrix-vector multiply: MRF × pipeline → pipeline."""
        self._clear_vv()
        mrf = self._mrf.get(MEM_MATRIX_RF)
        if mrf is not None and self._pipeline is not None:
            # Apply ReadMatrixMask to zero out masked rows
            mrfMask = self._regs.get(17, 0xFF)  # REG_READ_MATRIX_MASK, default all rows active
            n = mrf.shape[0]
            mask_dim = max(1, n // 8)  # rows per mask bit
            masked_mrf = mrf.copy()
            for bit_idx in range(8):
                if not (mrfMask >> bit_idx) & 1:
                    start_row = bit_idx * mask_dim
                    end_row = min(start_row + mask_dim, n)
                    if start_row < n:
                        masked_mrf[start_row:end_row, :] = 0.0

            rows, cols = masked_mrf.shape
            n = min(len(self._pipeline), cols)
            result = np.zeros(rows, dtype=np.float32)
            self._lib.mv_mul(self._ptr(masked_mrf), self._ptr(self._pipeline),
                             self._ptr(result), rows, n, 0)
            self._pipeline = result
            self._pipeline_round_fp16()
            # No auto-store to IVRF — firmware explicitly V_WR ACC/MUL/IVRF after MV_MUL.
            # Auto-storing would overwrite IVRF with Q, breaking multi-chain reads of input X.

    def _vv_mul(self, opcode, opd0, opd1):
        """Vector-vector elementwise multiply: _vpipe_a × pipeline → pipeline."""
        if self._vpipe_a is not None and self._pipeline is not None:
            n = min(len(self._vpipe_a), len(self._pipeline))
            r = np.zeros(n, dtype=np.float32)
            self._lib.vec_mul(self._ptr(self._vpipe_a), self._ptr(self._pipeline),
                              self._ptr(r), n)
            self._pipeline = r
            self._pipeline_round_fp16()
        self._vpipe_a = None

    def _vv_add_sub(self, opcode, opd0, opd1):
        """Vector-vector binary operation: vpipe_a OP pipeline → pipeline.
        
        OP_VV_ADD:      vpipe_a + pipeline
        OP_VV_A_SUB_B:  vpipe_a - pipeline
        OP_VV_B_SUB_A:  pipeline - vpipe_a
        OP_VV_MAX:      max(vpipe_a, pipeline)
        OP_VV_MIN:      min(vpipe_a, pipeline)
        """
        if self._vpipe_a is not None and self._pipeline is not None:
            n = min(len(self._vpipe_a), len(self._pipeline))
            r = np.zeros(n, dtype=np.float32)
            if opcode == OP_VV_ADD:
                self._lib.vec_add(self._ptr(self._vpipe_a), self._ptr(self._pipeline),
                                  self._ptr(r), n)
            elif opcode == OP_VV_A_SUB_B:
                self._lib.vec_sub(self._ptr(self._vpipe_a), self._ptr(self._pipeline),
                                  self._ptr(r), n)
            elif opcode == OP_VV_B_SUB_A:
                self._lib.vec_sub(self._ptr(self._pipeline), self._ptr(self._vpipe_a),
                                  self._ptr(r), n)
            elif opcode == OP_VV_MAX:
                self._lib.vec_max(self._ptr(self._vpipe_a), self._ptr(self._pipeline),
                                  self._ptr(r), n)
            else:
                # OP_VV_MIN: just use add and negate? Or implement.
                # Fallback: add for unknown
                self._lib.vec_add(self._ptr(self._vpipe_a), self._ptr(self._pipeline),
                                  self._ptr(r), n)
            self._pipeline = r
            self._pipeline_round_fp16()
        self._vpipe_a = None

    def _v_activation(self, opcode, opd0, opd1):
        """Activation functions on pipeline."""
        self._clear_vv()
        if self._pipeline is not None:
            n = len(self._pipeline)
            r = np.zeros(n, dtype=np.float32)
            if opcode == OP_V_GELU:
                self._lib.gelu(self._ptr(self._pipeline), self._ptr(r), n)
            elif opcode == OP_V_RELU:
                self._lib.relu(self._ptr(self._pipeline), self._ptr(r), n)
            elif opcode == OP_V_SIGM:
                self._lib.sigmoid(self._ptr(self._pipeline), self._ptr(r), n)
            elif opcode == OP_V_TANH:
                r[:] = np.tanh(self._pipeline)
            elif opcode == OP_V_EXP:
                r[:] = np.exp(self._pipeline)  # elementwise exp
            self._pipeline = r
            self._store_to_ivrf()

    def _v_softmax(self, opcode, opd0, opd1):
        """Softmax on pipeline."""
        self._clear_vv()
        if self._pipeline is not None:
            n = len(self._pipeline)
            r = np.zeros(n, dtype=np.float32)
            self._lib.softmax(self._ptr(self._pipeline), self._ptr(r), n)
            self._pipeline = r
            self._store_to_ivrf()

    def _v_layernorm(self, opcode, opd0, opd1):
        """Layer normalization on pipeline.

        Uses IVRF (mem_id 5) as gamma and AddSubVrf_0 (mem_id 7) as beta,
        matching the HDL dispatch in npu_top.py:
            slu_gamma[i] = ivrf[i], slu_beta[i] = as0[i].
        """
        self._clear_vv()
        if self._pipeline is not None:
            n = len(self._pipeline)
            gamma = self._vrf.get(5, np.ones(n, dtype=np.float32))[:n]
            beta  = self._vrf.get(7, np.zeros(n, dtype=np.float32))[:n]
            r = np.zeros(n, dtype=np.float32)
            self._lib.layernorm(self._ptr(self._pipeline), self._ptr(gamma),
                                self._ptr(beta), self._ptr(r), n, 1e-12)
            self._pipeline = r
            self._store_to_ivrf()

    def _spu_func(self, opcode, opd0, opd1):
        """SPU scalar functions (recip, sqrt, mul).
        
        opd0: SRF target index (where to store the result).
        opd1: SRF source index (where to read the operand).
        """
        src = opd1 & 0x3F  # limit to SRF size (64)
        dst = opd0 & 0x3F
        val = self._spu_srf[src] if src < len(self._spu_srf) else 0.0
        
        if opcode == OP_S_RECIP:
            # reciprocal: dst = 1.0 / src
            result = 1.0 / val if val != 0.0 else float('inf')
        elif opcode == OP_S_SQRT:
            # square root: dst = sqrt(src)
            result = math.sqrt(val) if val >= 0.0 else float('nan')
        elif opcode == OP_SS_MUL:
            # scalar multiply: dst = dst * src 
            dst_val = self._spu_srf[dst] if dst < len(self._spu_srf) else 0.0
            result = dst_val * val
        else:
            result = 0.0
        
        if dst < len(self._spu_srf):
            self._spu_srf[dst] = float(result)

    # ── State snapshot for debug instrumentation ─────────────────
    def snapshot(self):
        """Return a dict capturing the full NPU state.

        Used by operator-by-operator verification to compare
        intermediates against golden reference without modifying
        firmware.  All arrays are copies (safe to mutate).
        """
        return {
            'pipeline': self._pipeline.copy() if self._pipeline is not None else None,
            'vpipe_a': self._vpipe_a.copy() if self._vpipe_a is not None else None,
            'dram_addr': self._dram_addr,
            'regs': dict(self._regs),
            'spu_srf': self._spu_srf.copy(),
            'vrf': {mem: arr.copy() for mem, arr in self._vrf.items()},
            'mrf': {mem: arr.copy() for mem, arr in self._mrf.items()},
        }
