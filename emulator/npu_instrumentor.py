"""
NPU — Operator-by-Operator Debug Instrumentor.

Wraps NpuDeviceMini to capture intermediate state at operator boundaries
during firmware execution.  No firmware changes required — boundaries are
detected by monitoring DRAM write addresses.

Usage:
    npu = NpuDeviceMini(native_dim=dim)
    boundaries = [OpBoundary('Q', 0x200, num_tiles, stride=8), ...]
    instr = NpuInstrumentor(npu, boundaries)
    rec = TraceRecorder(instr)
    cpu = MiniRV64()
    cpu.set_mmio_device(rec)
    cpu.load_elf(str(elf))
    cpu.run(cycles=50000)

    # After execution:
    snapshots = instr.snapshots  # label → [snapshot, ...]
    instr.unpatch()
"""

from typing import Dict, List, Optional, Tuple

from emulator.npu_device_mini import MEM_DRAM


# ── Boundary definition ──────────────────────────────────────────────

class OpBoundary:
    """An operator boundary: triggered when firmware writes a specific
    DRAM address range.

    The boundary fires on the LAST tile row write, so that all tile
    rows of the intermediate result are in DRAM before we snapshot.

    Attributes:
        label:   Human-readable name (e.g., 'Q', 'K', 'V', 'save_res').
        base:    DRAM base address of the tiled vector.
        last:    DRAM address of the last tile row (= base + (num_tiles-1)*stride).
        stride:  Byte stride between tile rows in DRAM.
    """
    __slots__ = ('label', 'base', 'last', 'stride')

    def __init__(self, label: str, base: int, num_tiles: int = 1,
                 stride: int = 8):
        self.label = label
        self.base = base
        self.last = base + (num_tiles - 1) * stride
        self.stride = stride


# ── Instrumentor ─────────────────────────────────────────────────────

class NpuInstrumentor:
    """Wraps an NPU device and captures state snapshots at operator
    boundaries.

    Detection mechanism: hooks into _v_wr_dram to watch for writes to
    boundary addresses.  When a boundary address is hit, captures a
    full snapshot of the NPU state.

    Works with TraceRecorder layered on top — TraceRecorder calls
    store() → _inner.store() → ... → eventually _v_wr_dram on the
    real NpuDeviceMini.
    """

    def __init__(self, inner_device, boundaries: List[OpBoundary],
                 capture_dram_range: Optional[Tuple[int, int]] = None):
        """
        Args:
            inner_device:       The NpuDeviceMini instance (or wrapper).
            boundaries:         List of OpBoundary objects defining when to
                                capture snapshots.
            capture_dram_range: If set, (start, length) of DRAM region to
                                include in snapshot.  None = full DRAM.
        """
        self._inner = inner_device
        self._boundary_by_last = {b.last: b for b in boundaries}
        self._capture_dram_range = capture_dram_range

        # Results: label → list of snapshots (one per position for seq>1)
        self.snapshots: Dict[str, List[dict]] = {b.label: [] for b in boundaries}

        # Instruction counter for correlation
        self._inst_count = 0

        # Patch the NPU's _v_wr_dram to intercept DRAM writes
        self._original_v_wr_dram = inner_device._v_wr_dram
        inner_device._v_wr_dram = self._patched_v_wr_dram

        # Also patch _push_instruction to count instructions
        self._original_push = inner_device._push_instruction
        inner_device._push_instruction = self._patched_push

    # ── Delegated interface ──────────────────────────────────────

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # ── Patched methods ──────────────────────────────────────────

    def _patched_push(self, inst: int):
        self._inst_count += 1
        self._original_push(inst)

    def _patched_v_wr_dram(self, full_operand, pipeline=None):
        """Intercept V_WR_DRAM to detect operator boundaries.

        When the firmware writes to a boundary address, capture a
        snapshot AFTER the write completes so that DRAM contains
        the result for comparison against golden reference.
        """
        # Delegate to original first (write completes)
        self._original_v_wr_dram(full_operand, pipeline=pipeline)

        # Now check if the address is the last tile row of a boundary
        addr = full_operand
        boundary = self._boundary_by_last.get(addr)
        if boundary is not None:
            self._capture_snapshot(boundary)

    # ── Snapshot capture ─────────────────────────────────────────

    def _capture_snapshot(self, boundary: OpBoundary):
        """Capture current NPU state and store under boundary.label."""
        snap = self._inner.snapshot()

        # Add metadata
        snap['_meta'] = {
            'label': boundary.label,
            'base': boundary.base,
            'inst_count': self._inst_count,
        }

        # Trim DRAM if range specified
        if self._capture_dram_range is not None:
            start, length = self._capture_dram_range
            dram = snap['vrf'].get(MEM_DRAM)
            if dram is not None:
                snap['vrf'][MEM_DRAM] = dram[start:start + length].copy()

        self.snapshots[boundary.label].append(snap)

    # ── Unpatch (for cleanup) ────────────────────────────────────

    def unpatch(self):
        """Restore original methods on the inner device."""
        self._inner._v_wr_dram = self._original_v_wr_dram
        self._inner._push_instruction = self._original_push
