"""
Shared NpuFP32 class for FP32 NPU tests.

Subclasses NpuDeviceMini to disable all FP16 truncation,
enabling clean float32 testing of NPU instruction sequences.
Also adds convenience methods for building instruction streams.
"""

import numpy as np
from emulator.npu_device_mini import (
    NpuDeviceMini,
    MEM_DRAM, MEM_MULTIPLY_VRF, MEM_MVM_INITIAL_VRF,
    MEM_ADDSUB_VRF_0, MEM_MATRIX_RF, MEM_FILL,
    MEM_VEC_TO_MAT_ROW, MEM_SPU_ADD_REDUCE, MEM_SPU_MAX_REDUCE,
    MEM_SPU_ABSMAX_REDUCE, MEM_SPU_BROADCAST,
    OP_V_RD_DRAM, OP_V_WR_DRAM,
    OP_V_RD, OP_V_WR,
    OP_M_RD_DRAM, OP_M_WR, OP_M_RD, OP_MV_MUL,
    OP_VV_ADD, OP_VV_A_SUB_B, OP_VV_B_SUB_A, OP_VV_MUL,
    OP_V_RELU, OP_V_EXP, OP_V_FUNC,
    OP_S_WR, OP_S_RECIP, OP_S_SQRT, OP_SS_MUL,
    SUB_SOFTMAX,
)


def si(op, opd0, opd1):
    """Encode SI-format instruction word."""
    return ((op & 0xFF) << 24) | ((opd0 & 0xFF) << 16) | (opd1 & 0xFFFF)


def lo(op, addr):
    """Encode LO-format instruction word (24-bit address)."""
    return ((op & 0xFF) << 24) | (addr & 0xFFFFFF)


class NpuFP32(NpuDeviceMini):
    """NpuDeviceMini with all FP16 truncation disabled for clean float32 testing."""

    def __init__(self, native_dim=4):
        super().__init__(native_dim=native_dim)
        self._round_fp16 = lambda arr: arr if arr is not None else None
        self._store_to_ivrf = lambda p: None

    # ── Override all FP16-truncating methods ──
    #
    # New API: each handler takes (pipeline, vpipe_a) and returns
    # (pipeline_out, vpipe_a_out).

    def _v_rd_dram(self, full_operand, pipeline=None, vpipe_a=None):
        addr = full_operand
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None:
            n = min(self.native_dim, len(dram) - addr) if addr < len(dram) else 0
            result = np.zeros(self.native_dim, dtype=np.float32)
            if n > 0:
                result[:n] = dram[addr:addr + n]
            return result, pipeline
        return np.zeros(self.native_dim, dtype=np.float32), pipeline

    def _v_wr_dram(self, full_operand, pipeline=None):
        if pipeline is None:
            return
        addr = full_operand
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None:
            n = min(self.native_dim, len(pipeline), len(dram) - addr) if addr < len(dram) else 0
            if n > 0:
                dram[addr:addr + n] = pipeline[:n]

    def _m_rd(self, opd0, full_operand, pipeline=None, vpipe_a=None):
        """Override: VecToMatRow uses parent."""
        return super()._m_rd(opd0, full_operand, pipeline=pipeline, vpipe_a=vpipe_a)

    def _m_rd_dram(self, full_operand):
        """Override: DRAM → MRF with direct copy, no FP16 rounding."""
        addr = full_operand
        dram = self._vrf.get(MEM_DRAM)
        if dram is not None and addr < len(dram):
            n = self._regs.get(1, 1) * self.native_dim
            mrf_size = n * n
            if addr + mrf_size <= len(dram):
                self._dram_stats['mat_rd_elements'] += mrf_size
                self._dram_stats['mat_rd_ops'] += 1
                mat = dram[addr:addr + mrf_size].reshape(n, n).copy()
                self._mrf[MEM_MATRIX_RF] = mat

    def _v_rd(self, opd0, opd1, pipeline=None, vpipe_a=None):
        mem_target, addr = opd0, opd1
        if mem_target == MEM_FILL:
            val = np.frombuffer(np.uint16([addr]).tobytes(), dtype=np.float16)[0]
            return np.full(self.native_dim, float(val), dtype=np.float32), pipeline
        if mem_target == MEM_SPU_BROADCAST:
            val = self._spu_srf[addr] if addr < len(self._spu_srf) else 0.0
            return np.full(self.native_dim, val, dtype=np.float32), pipeline
        vrf = self._vrf.get(mem_target)
        if vrf is not None:
            n = min(self.native_dim, max(0, len(vrf) - addr))
            result = np.zeros(self.native_dim, dtype=np.float32)
            if n > 0:
                result[:n] = vrf[addr:addr + n]
            return result, pipeline
        return np.zeros(self.native_dim, dtype=np.float32), pipeline

    def _v_wr(self, opd0, opd1, pipeline=None):
        if pipeline is None:
            return
        mem_target, addr = opd0, opd1
        # SPU reduce
        if mem_target == MEM_SPU_ADD_REDUCE:
            if addr < len(self._spu_srf):
                self._spu_srf[addr] = float(np.sum(pipeline)) + self._spu_srf[addr]
            return
        if mem_target in (MEM_SPU_MAX_REDUCE, MEM_SPU_ABSMAX_REDUCE):
            if addr < len(self._spu_srf):
                data = pipeline if mem_target == MEM_SPU_MAX_REDUCE else np.abs(pipeline)
                self._spu_srf[addr] = max(float(np.max(data)), self._spu_srf[addr])
            return
        # VecToMatRow
        if mem_target == MEM_VEC_TO_MAT_ROW:
            key = 0
            if key not in self._row_buffer:
                self._row_buffer[key] = []
            self._row_buffer[key].append(pipeline.copy())
            return
        # General VRF store (no FP16 truncation)
        vrf = self._vrf.setdefault(mem_target, np.zeros(self.native_dim * 8, dtype=np.float32))
        n = min(self.native_dim, max(0, len(vrf) - addr))
        if n > 0:
            vrf[addr:addr + n] = pipeline[:n]

    # ── Convenience methods ──

    def load_dram(self, dram):
        """Copy weight DRAM into NPU DRAM."""
        self._vrf[MEM_DRAM][:len(dram)] = dram.copy()

    def get_dram(self, addr, n=4):
        """Read n elements from DRAM starting at addr."""
        return self._vrf[MEM_DRAM][addr:addr + n].copy()

    def set_dram(self, addr, arr):
        """Write array to DRAM starting at addr."""
        n = min(len(arr), len(self._vrf[MEM_DRAM]) - addr)
        self._vrf[MEM_DRAM][addr:addr + n] = arr[:n].astype(np.float32)

    def send(self, inst):
        """Push one instruction through the FIFO."""
        self._push_instruction(inst)

    def send_si(self, op, opd0, opd1):
        self.send(si(op, opd0, opd1))

    def send_lo(self, op, addr):
        self.send(lo(op, addr))

    def opcode_coverage(self):
        """Return set of opcode names used (from ops list)."""
        if not hasattr(self, '_ops_log'):
            return set()
        return set(name for name, _ in self._ops_log)

    # ── High-level instruction helpers ──

    def mvm(self, mat_addr, vec_addr, sink_vrf=MEM_MULTIPLY_VRF):
        """MV_MUL from DRAM weight tile and DRAM vector, store to sink_vrf."""
        self.send_lo(OP_M_RD_DRAM, mat_addr)
        self.send_si(OP_M_WR, MEM_MATRIX_RF, 0)
        self.send_lo(OP_V_RD_DRAM, vec_addr)
        self.send_si(OP_V_WR, MEM_MVM_INITIAL_VRF, 0)
        self.send_si(OP_V_RD, MEM_MVM_INITIAL_VRF, 0)
        self.send_si(OP_MV_MUL, 0, 0)
        self.send_si(OP_V_WR, sink_vrf, 0)

    def load_vec_to_mat(self, addr):
        """Load one DRAM vector into VecToMatRow buffer."""
        self.send_lo(OP_V_RD_DRAM, addr)
        self.send_si(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0)

    def mat_from_vec_to_mat(self):
        """Transfer row buffer into MRF."""
        self.send_si(OP_M_RD, MEM_VEC_TO_MAT_ROW, 0)

    def broadcast_srf(self, idx):
        """Broadcast SRF[idx] to all pipeline elements."""
        self.send_si(OP_V_RD, MEM_SPU_BROADCAST, idx)

    def spu_max_reduce(self, dst):
        """Accumulate max of pipeline into SRF[dst]."""
        self.send_si(OP_V_WR, MEM_SPU_MAX_REDUCE, dst)

    def spu_add_reduce(self, dst):
        """Accumulate sum of pipeline into SRF[dst]."""
        self.send_si(OP_V_WR, MEM_SPU_ADD_REDUCE, dst)
