"""
NPU — MMIO Trace Recorder and Cross-Backend Replay

Records the instruction stream sent by RISC-V firmware to the NPU MMIO
interface, then replays those same instructions against the Amaranth
NpuTop simulator in batch mode — matching real hardware behavior.

Key insight: on real hardware, the firmware pushes multiple instructions
into the FIFO without waiting (fire-and-forget), then calls npu_wait_done()
to synchronize.  The trace recorder captures instruction writes AND
status reads, allowing us to identify wait_done boundaries and replay
in instruction batches rather than one-at-a-time.
"""

from typing import Callable, List, Tuple

# NPU MMIO register offsets (matching npu_regs.h)
NPU_INST_FIFO = 0x00
NPU_STATUS    = 0x04
NPU_RESET     = 0x08

# NPU data exchange windows (must match npu_device_mini.py)
NPU_DRAM_BASE = 0x40
NPU_DRAM_END  = 0x8000
NPU_SRF_BASE  = 0x8000
NPU_SRF_END   = 0x8100

# Operation types for full trace
OP_WR_INST  = 0  # write to NPU_INST_FIFO
OP_RD_STAT  = 1  # read from NPU_STATUS
OP_WR_DRAM  = 2  # write to NPU DRAM window (addr, data_bytes stored in value/extra)
OP_WR_SRF   = 3  # write to NPU SRF window (addr, data_bytes)
OP_OTHER    = 4  # other MMIO access


class TraceRecorder:
    """Wraps an NPU device and records the full MMIO operation sequence.

    Records both instruction writes (to NPU_INST_FIFO) and status reads
    (from NPU_STATUS) in order, so the replay engine can identify
    npu_wait_done() boundaries and replay in instruction batches.
    """

    def __init__(self, inner_device):
        self._inner = inner_device
        # Full ordered trace: list of (op_type, value) tuples
        #   OP_WR_INST → value is the instruction word
        #   OP_RD_STAT → value is the status result read back
        #   OP_OTHER   → value is 0
        self.full_trace: List[Tuple[int, int]] = []
        self.inst_trace: List[int] = []
        self.dram_writes: List[Tuple[int, int]] = []  # (float_offset, raw_u32)
        self.srf_writes: List[Tuple[int, int]] = []   # (srf_idx, raw_u32)
        self.status_reads: int = 0
        self.inst_writes: int = 0
        self.other_accesses: int = 0

    # ── Delegated MMIO interface ────────────────────────────────────

    def load(self, addr: int, size: int) -> bytes:
        result = self._inner.load(addr, size)
        if addr == NPU_STATUS:
            val = int.from_bytes(result, 'little')
            self.full_trace.append((OP_RD_STAT, val))
            self.status_reads += 1
        else:
            self.full_trace.append((OP_OTHER, 0))
            self.other_accesses += 1
        return result

    def store(self, addr: int, data: bytes):
        if addr == NPU_INST_FIFO:
            val = int.from_bytes(data, 'little')
            self.full_trace.append((OP_WR_INST, val))
            self.inst_trace.append(val)
            self.inst_writes += 1
            self._inner.store(addr, data)
            return
        elif addr == NPU_RESET:
            val = int.from_bytes(data, 'little')
            if val:
                self.full_trace.clear()
                self.inst_trace.clear()
            self.full_trace.append((OP_OTHER, 0))
            self.other_accesses += 1
            self._inner.store(addr, data)
            return

        # Capture DRAM and SRF window writes for replay
        if NPU_DRAM_BASE <= addr < NPU_DRAM_END:
            # DRAM write: store (float_offset, float_value)
            float_off = (addr - NPU_DRAM_BASE) // 4
            f32 = int.from_bytes(data, 'little', signed=False)
            self.dram_writes.append((float_off, f32))
            self.full_trace.append((OP_WR_DRAM, float_off))
            self._inner.store(addr, data)
            return

        if NPU_SRF_BASE <= addr < NPU_SRF_END:
            idx = (addr - NPU_SRF_BASE) // 4
            f32 = int.from_bytes(data, 'little', signed=False)
            self.srf_writes.append((idx, f32))
            self.full_trace.append((OP_WR_SRF, idx))
            self._inner.store(addr, data)
            return

        self.full_trace.append((OP_OTHER, 0))
        self.other_accesses += 1
        self._inner.store(addr, data)

    def tick(self):
        if hasattr(self._inner, 'tick'):
            self._inner.tick()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # ── Replay batch extraction ────────────────────────────────────

    def extract_batches(self) -> List[List[int]]:
        """Extract instruction batches separated by npu_wait_done().

        npu_wait_done() polls STATUS until DONE.  On the emulator each
        poll returns DONE immediately, so npu_wait_done() produces one
        STATUS read.  npu_send_inst() also performs one STATUS read
        (checking FULL) before each INST write.

        A batch boundary is detected by two consecutive STATUS reads
        without an INST_FIFO write between them:

          ... WR_INST  (last instruction before wait_done)
          ... RD_STAT  (npu_wait_done: reads STATUS, sees DONE)
          ... RD_STAT  (next npu_send_inst: reads STATUS, sees FULL=0)
          ... WR_INST  (next instruction)

        The two consecutive RD_STAT operations mark the boundary.

        Returns:
            List of instruction lists, one per batch.
        """
        batches = []
        current_batch = []

        for i in range(len(self.full_trace)):
            op_type, value = self.full_trace[i]
            if op_type == OP_WR_INST:
                current_batch.append(value)
            elif op_type == OP_RD_STAT and current_batch:
                # Two consecutive status reads = npu_wait_done boundary
                # (Skip the initial IDLE poll which has no preceding inst)
                if (i + 1 < len(self.full_trace)
                        and self.full_trace[i + 1][0] == OP_RD_STAT):
                    batches.append(current_batch)
                    current_batch = []

        if current_batch:
            batches.append(current_batch)

        return batches

    # ── Report ─────────────────────────────────────────────────────

    def report(self):
        batches = self.extract_batches()
        print(f"  TraceRecorder: {self.inst_writes} instructions,"
              f" {self.status_reads} status reads,"
              f" {len(batches)} batches"
              f" ({' '.join(str(len(b)) for b in batches)} inst/batch)")


def replay_batches(batches: List[List[int]], push_fn: Callable,
                   wait_fn: Callable, timeout: int = 200):
    """Replay instruction batches like real hardware.

    Each batch is pushed fire-and-forget (all instructions go into the
    FIFO without waiting), then wait_fn() is called to synchronize.

    Args:
        batches:  list of instruction lists from extract_batches()
        push_fn:  async callable(inst) that writes one instruction
        wait_fn:  async callable() that polls STATUS until DONE
        timeout:  max ticks per batch
    """
    for batch in batches:
        for inst in batch:
            push_fn(inst)
        wait_fn()

    print(f"  Replay: {sum(len(b) for b in batches)} instructions"
          f" in {len(batches)} batches")
