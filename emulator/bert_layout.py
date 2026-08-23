"""Versioned DRAM layouts for the BERT firmware and its host-side tools."""

from __future__ import annotations

from dataclasses import dataclass


LEGACY_LAYOUT = "legacy-v1"
PACKED_LAYOUT = "packed-v2"
SUPPORTED_LAYOUTS = frozenset({LEGACY_LAYOUT, PACKED_LAYOUT})

DRAM_ELEMENT_CAPACITY = 524_288
LO_ADDRESS_CAPACITY = 1 << 24


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class BertDramLayout:
    """Element-addressed DRAM layout shared by firmware builders and tests."""

    version: str
    native_dim: int
    hidden_size: int
    seq_len: int
    num_tiles: int
    tile_stride: int
    input_base: int
    projection_base: int
    matrix_size: int
    projection_stride: int
    ln1_gamma: int
    ln1_beta: int
    ln2_gamma: int
    ln2_beta: int
    ln_param_span: int
    save_q_base: int
    save_k_base: int
    save_v_base: int
    attention_scratch: int
    layernorm_scratch: int
    scratch_z: int
    scratch_ln1: int
    scratch_gelu: int
    so_scratch: int
    save_res_base: int
    save_out_base: int
    unit_vec_base: int
    end_address: int

    @property
    def position_span(self) -> int:
        return self.num_tiles * self.tile_stride

    @property
    def sequence_span(self) -> int:
        return self.seq_len * self.position_span

    def position_address(self, base: int, position: int, tile: int = 0) -> int:
        return base + position * self.position_span + tile * self.tile_stride

    def build_environment(self) -> dict[str, str]:
        """Return the Make variables consumed by ``firmware/Makefile``."""
        return {
            "NATIVE_DIM": str(self.native_dim),
            "SEQ_LEN": str(self.seq_len),
            "_HIDDEN_SIZE": str(self.hidden_size),
            "_PROJ_BASE": str(self.projection_base),
            "_MAT_SIZE": str(self.matrix_size),
            "_STRIDE": str(self.projection_stride),
            "_NUM_TILES": str(self.num_tiles),
            "_TILE_STRIDE": str(self.tile_stride),
            "_LN1_GAMMA": str(self.ln1_gamma),
            "_LN1_BETA": str(self.ln1_beta),
            "_LN2_GAMMA": str(self.ln2_gamma),
            "_LN2_BETA": str(self.ln2_beta),
            "_SCRATCH": str(self.layernorm_scratch),
            "_SAVE_Q_BASE": str(self.save_q_base),
            "_SAVE_K_BASE": str(self.save_k_base),
            "_SAVE_V_BASE": str(self.save_v_base),
            "_ATTENTION_SCRATCH": str(self.attention_scratch),
            "_SCRATCH_Z": str(self.scratch_z),
            "_SCRATCH_LN1": str(self.scratch_ln1),
            "_SCRATCH_GELU": str(self.scratch_gelu),
            "_SO_SCRATCH": str(self.so_scratch),
            "_SAVE_RES_BASE": str(self.save_res_base),
            "_SAVE_OUT_BASE": str(self.save_out_base),
            "_UNIT_VEC_BASE": str(self.unit_vec_base),
        }

    def named_regions(self) -> dict[str, tuple[int, int]]:
        """Return non-overlapping packed-v2 regions for validation/manifests."""
        if self.version != PACKED_LAYOUT:
            raise ValueError("named_regions is only unambiguous for packed-v2")
        projection_end = self.projection_base + 6 * self.projection_stride
        return {
            "input": (self.input_base, self.hidden_size * self.seq_len),
            "projections": (
                self.projection_base,
                projection_end - self.projection_base,
            ),
            "layernorm_parameters": (self.ln1_gamma, 4 * self.ln_param_span),
            "q": (self.save_q_base, self.sequence_span),
            "k": (self.save_k_base, self.sequence_span),
            "v": (self.save_v_base, self.sequence_span),
            "attention_scratch": (self.attention_scratch, self.native_dim),
            "layernorm_scratch": (
                self.layernorm_scratch,
                (self.num_tiles + 1) * self.tile_stride,
            ),
            "z": (self.scratch_z, self.hidden_size),
            "ln1": (self.scratch_ln1, self.hidden_size),
            "gelu": (self.scratch_gelu, self.hidden_size),
            "self_output": (self.so_scratch, self.hidden_size),
            "residual": (self.save_res_base, self.sequence_span),
            "output": (self.save_out_base, self.sequence_span),
            "unit_vectors": (
                self.unit_vec_base,
                self.native_dim * self.native_dim,
            ),
        }


def bert_dram_layout(
    native_dim: int,
    hidden_size: int,
    seq_len: int,
    *,
    version: str = LEGACY_LAYOUT,
) -> BertDramLayout:
    """Create and validate a BERT DRAM layout for one build configuration."""
    native_dim = int(native_dim)
    hidden_size = int(hidden_size)
    seq_len = int(seq_len)
    if version not in SUPPORTED_LAYOUTS:
        raise ValueError(f"unsupported BERT layout {version!r}")
    if native_dim < 1 or hidden_size < 1 or seq_len < 1:
        raise ValueError("native_dim, hidden_size, and seq_len must be positive")
    if hidden_size % native_dim:
        raise ValueError("hidden_size must be divisible by native_dim")
    num_tiles = hidden_size // native_dim
    if num_tiles > 2:
        raise ValueError("current BERT firmware supports at most two hidden tiles")

    matrix_size = hidden_size * hidden_size
    projection_stride = matrix_size + hidden_size

    if version == LEGACY_LAYOUT:
        tile_stride = 8
        projection_base = hidden_size * seq_len + 4
        ln_base = projection_base + 6 * projection_stride
        ln_param_span = num_tiles * tile_stride
        layout = BertDramLayout(
            version=version, native_dim=native_dim, hidden_size=hidden_size,
            seq_len=seq_len, num_tiles=num_tiles, tile_stride=tile_stride,
            input_base=0, projection_base=projection_base,
            matrix_size=matrix_size, projection_stride=projection_stride,
            ln1_gamma=ln_base, ln1_beta=ln_base + ln_param_span,
            ln2_gamma=ln_base + 2 * ln_param_span,
            ln2_beta=ln_base + 3 * ln_param_span,
            ln_param_span=ln_param_span,
            save_q_base=0x200, save_k_base=0x300, save_v_base=0x400,
            attention_scratch=0x500, layernorm_scratch=0x500,
            scratch_z=0x600, scratch_ln1=0x620, scratch_gelu=0x640,
            so_scratch=0x580, save_res_base=0x700,
            save_out_base=0x800, unit_vec_base=0x900,
            end_address=0x900 + native_dim * native_dim,
        )
    else:
        if seq_len > native_dim:
            raise ValueError("packed-v2 requires seq_len <= native_dim")
        tile_stride = native_dim
        projection_base = _align(hidden_size * seq_len + 4, native_dim)
        cursor = projection_base + 6 * projection_stride
        ln_param_span = num_tiles * tile_stride
        ln_base = _align(cursor, native_dim)
        cursor = ln_base + 4 * ln_param_span

        def allocate(length: int) -> int:
            nonlocal cursor
            cursor = _align(cursor, native_dim)
            base = cursor
            cursor += length
            return base

        sequence_span = seq_len * num_tiles * tile_stride
        save_q = allocate(sequence_span)
        save_k = allocate(sequence_span)
        save_v = allocate(sequence_span)
        attention_scratch = allocate(native_dim)
        layernorm_scratch = allocate((num_tiles + 1) * tile_stride)
        scratch_z = allocate(hidden_size)
        scratch_ln1 = allocate(hidden_size)
        scratch_gelu = allocate(hidden_size)
        so_scratch = allocate(hidden_size)
        save_res = allocate(sequence_span)
        save_out = allocate(sequence_span)
        unit_vec = allocate(native_dim * native_dim)
        layout = BertDramLayout(
            version=version, native_dim=native_dim, hidden_size=hidden_size,
            seq_len=seq_len, num_tiles=num_tiles, tile_stride=tile_stride,
            input_base=0, projection_base=projection_base,
            matrix_size=matrix_size, projection_stride=projection_stride,
            ln1_gamma=ln_base, ln1_beta=ln_base + ln_param_span,
            ln2_gamma=ln_base + 2 * ln_param_span,
            ln2_beta=ln_base + 3 * ln_param_span,
            ln_param_span=ln_param_span,
            save_q_base=save_q, save_k_base=save_k, save_v_base=save_v,
            attention_scratch=attention_scratch,
            layernorm_scratch=layernorm_scratch, scratch_z=scratch_z,
            scratch_ln1=scratch_ln1, scratch_gelu=scratch_gelu,
            so_scratch=so_scratch, save_res_base=save_res,
            save_out_base=save_out, unit_vec_base=unit_vec,
            end_address=cursor,
        )
        ordered = sorted(
            (base, base + length, name)
            for name, (base, length) in layout.named_regions().items()
        )
        for left, right in zip(ordered, ordered[1:]):
            if left[1] > right[0]:
                raise ValueError(f"packed-v2 regions overlap: {left[2]} and {right[2]}")

    if layout.end_address > DRAM_ELEMENT_CAPACITY:
        raise ValueError("BERT layout exceeds emulator DRAM capacity")
    if layout.end_address > LO_ADDRESS_CAPACITY:
        raise ValueError("BERT layout exceeds the 24-bit LO address space")
    return layout
