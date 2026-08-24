"""Workload semantics for the model-independent NPU DAG exporter.

The event tracer and micro-op DAG only know ISA resources.  This module is the
small, explicit boundary that turns a DRAM address into a model tensor.  New
models should add an adapter here (or register one at runtime) instead of
adding address checks to :mod:`emulator.npu_dag_structured`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from emulator.npu_micro_op_dag import _DRAM_REGIONS, _classify_dram_addr


ReusePolicy = tuple[str, str, str]


@dataclass(frozen=True)
class TensorRegion:
    """One half-open DRAM tensor interval, measured in float elements."""

    name: str
    base: int
    size: int
    role: str
    position_stride: int | None = None
    tile_stride: int | None = None
    reuse_policy: ReusePolicy | None = None

    @property
    def end(self) -> int:
        return self.base + self.size

    def contains(self, address: int) -> bool:
        return self.base <= address < self.end


class DagWorkloadAdapter:
    """Address and cache semantics supplied by one model workload."""

    name = "generic"
    target_file: str | None = None
    axis_name = "seq_len"
    execution_scope = "NPU firmware instruction stream"
    supports_specialized_contracts = False

    def __init__(self, config: dict[str, int]):
        self.config = dict(config)

    def classify_dram(self, address: int) -> dict[str, Any]:
        return {
            "tensor": "DRAM_UNKNOWN",
            "role": "unknown",
            "position": None,
            "tile": None,
            "slice": f"DRAM_UNKNOWN[address=0x{address:x}]",
            "position_in_config": True,
        }

    def semantic_phase(self, label: str) -> str | None:
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_file": self.target_file,
            "axis_name": self.axis_name,
            "execution_scope": self.execution_scope,
            "adapter": f"{type(self).__module__}.{type(self).__name__}",
            "specialized_contracts": self.supports_specialized_contracts,
        }


class RegionDagWorkloadAdapter(DagWorkloadAdapter):
    """Declarative adapter for models with a stable flat DRAM layout."""

    regions: tuple[TensorRegion, ...] = ()

    def __init__(self, config: dict[str, int]):
        super().__init__(config)
        self.regions = tuple(self.build_regions())
        ordered = sorted(self.regions, key=lambda region: region.base)
        for left, right in zip(ordered, ordered[1:]):
            if left.end > right.base:
                raise ValueError(
                    f"overlapping DAG tensor regions: {left.name} and {right.name}"
                )

    def build_regions(self) -> list[TensorRegion]:
        return []

    def classify_dram(self, address: int) -> dict[str, Any]:
        region = next(
            (candidate for candidate in self.regions if candidate.contains(address)),
            None,
        )
        if region is None:
            return super().classify_dram(address)

        relative = address - region.base
        position = (
            relative // region.position_stride
            if region.position_stride
            else None
        )
        position_offset = (
            position * region.position_stride if position is not None else 0
        )
        tile = (
            (relative - position_offset) // region.tile_stride
            if region.tile_stride
            else None
        )
        if position is not None and tile is not None:
            tensor_slice = f"{region.name}[pos={position},tile={tile}]"
        elif position is not None:
            tensor_slice = f"{region.name}[pos={position}]"
        elif tile is not None:
            tensor_slice = f"{region.name}[tile={tile}]"
        else:
            tensor_slice = f"{region.name}[address=0x{address:x}]"

        record: dict[str, Any] = {
            "tensor": region.name,
            "role": region.role,
            "position": position,
            "tile": tile,
            "slice": tensor_slice,
            "position_in_config": (
                position is None
                or 0 <= position < int(self.config.get("seq_len", 0))
            ),
        }
        if region.reuse_policy:
            record.update(
                {
                    "reuse_family": region.reuse_policy[0],
                    "cache_level": region.reuse_policy[1],
                    "reuse_kind": region.reuse_policy[2],
                }
            )
        return record


class BertDagWorkloadAdapter(DagWorkloadAdapter):
    """Compatibility adapter preserving the original BERT annotations."""

    name = "bert"
    target_file = "firmware/bert/bert_layer.c"
    supports_specialized_contracts = True

    _position_regions = {"X", "Q", "K", "V", "RES", "OUT"}
    _projection_layout = (
        ("Q", "W_Q", "B_Q"),
        ("K", "W_K", "B_K"),
        ("V", "W_V", "B_V"),
        ("SELF_OUTPUT", "W_SELF_OUTPUT", "B_SELF_OUTPUT"),
        ("FFN_INTERMEDIATE", "W_FFN_INTERMEDIATE", "B_FFN_INTERMEDIATE"),
        ("FFN_OUTPUT", "W_FFN_OUTPUT", "B_FFN_OUTPUT"),
    )

    def _classify_low(self, address: int) -> tuple[str, int | None, int | None]:
        dim = int(self.config["dim"])
        hidden_size = int(self.config["hidden_size"])
        seq_len = int(self.config["seq_len"])
        input_end = hidden_size * seq_len
        if 0 <= address < input_end:
            return "X", address // hidden_size, (address % hidden_size) // dim

        proj_base = input_end + 4
        mat_size = hidden_size * hidden_size
        stride = mat_size + hidden_size
        for index, (_projection, weight_name, bias_name) in enumerate(
            self._projection_layout
        ):
            weight_base = proj_base + index * stride
            bias_base = weight_base + mat_size
            if weight_base <= address < bias_base:
                return weight_name, None, (address - weight_base) // (dim * dim)
            if bias_base <= address < bias_base + hidden_size:
                return bias_name, None, (address - bias_base) // dim

        num_tiles = max(hidden_size // dim, 1)
        ln_base = proj_base + len(self._projection_layout) * stride
        ln_size = num_tiles * 8
        for index, name in enumerate(
            ("LN1_GAMMA", "LN1_BETA", "LN2_GAMMA", "LN2_BETA")
        ):
            base = ln_base + index * ln_size
            if base <= address < base + ln_size:
                return name, None, (address - base) // 8
        return "UNKNOWN_LOW", None, None

    @staticmethod
    def _reuse_policy(tensor: str) -> ReusePolicy | None:
        if tensor.startswith("B_") or tensor.startswith(("LN1_", "LN2_")):
            return "loop-invariant-parameters", "L2", "loop-invariant-load"
        if tensor == "UNIT_VEC":
            return "loop-invariant-parameters", "L2", "loop-invariant-load"
        if tensor == "X":
            return "sequence-input", "L2", "sequence-input-reuse"
        if tensor.startswith("W_"):
            return "weight-stationary", "L3", "weight-stationary-load"
        return None

    def classify_dram(self, address: int) -> dict[str, Any]:
        dim = int(self.config["dim"])
        hidden_size = int(self.config["hidden_size"])
        seq_len = int(self.config["seq_len"])
        num_tiles = max(hidden_size // dim, 1)
        if address < 0x200:
            tensor, position, tile = self._classify_low(address)
        else:
            tensor, raw_position = _classify_dram_addr(address, num_tiles, dim)
            position = (
                int(raw_position)
                if tensor in self._position_regions and raw_position >= 0
                else None
            )
            tile = None
            region_bases = {name: base for base, name, _span in _DRAM_REGIONS}
            if tensor in region_bases:
                base = region_bases[tensor]
                if tensor in self._position_regions:
                    position_stride = num_tiles * 8
                    relative = address - base - (position or 0) * position_stride
                    tile_stride = 8
                elif tensor == "UNIT_VEC":
                    relative = address - base
                    tile_stride = dim
                else:
                    relative = address - base
                    tile_stride = 8
                if relative >= 0:
                    tile = relative // tile_stride

        if tensor.startswith(("W_", "B_", "LN1_", "LN2_")):
            tensor_slice = (
                f"{tensor}[tile={tile}]"
                if tile is not None
                else f"{tensor}[address=0x{address:x}]"
            )
        elif position is not None and tile is not None:
            tensor_slice = f"{tensor}[pos={position},tile={tile}]"
        elif position is not None:
            tensor_slice = f"{tensor}[pos={position}]"
        elif tile is not None:
            tensor_slice = f"{tensor}[tile={tile}]"
        else:
            tensor_slice = f"{tensor}[address=0x{address:x}]"

        role = (
            "weight" if tensor.startswith("W_")
            else "parameter" if tensor.startswith(("B_", "LN1_", "LN2_"))
            else "input" if tensor == "X"
            else "attention" if tensor in {"Q", "K", "V", "SCORE", "SOFTMAX"}
            else "intermediate"
        )
        record: dict[str, Any] = {
            "tensor": tensor,
            "role": role,
            "position": position,
            "tile": tile,
            "slice": tensor_slice,
            "position_in_config": position is None or 0 <= position < seq_len,
        }
        policy = self._reuse_policy(tensor)
        if policy:
            record.update(
                {
                    "reuse_family": policy[0],
                    "cache_level": policy[1],
                    "reuse_kind": policy[2],
                }
            )
        return record


class Adder140pDagWorkloadAdapter(RegionDagWorkloadAdapter):
    """Qwen3-style 140p adder layout and NPU-subgraph semantics."""

    name = "adder_140p"
    target_file = "adderboard/firmware/adder_140p.c"
    execution_scope = (
        "two NPU firmware phases; host embedding/norm/projection/RoPE and LM head excluded"
    )

    def build_regions(self) -> list[TensorRegion]:
        dim = int(self.config.get("hidden_size", 4))
        seq_len = int(self.config.get("seq_len", 24))
        max_seq = 35
        weight = ("weight-stationary", "L3", "weight-stationary-load")
        invariant = (
            "loop-invariant-parameters",
            "L2",
            "loop-invariant-load",
        )
        nt = (seq_len + dim - 1) // dim
        scratch_base = 0x2000
        s_x = scratch_base
        s_q = s_x + max_seq * dim
        s_k = s_q + max_seq * dim
        s_v = s_k + max_seq * dim
        s_ctx = s_v + max_seq * dim
        s_attn_out = s_ctx + max_seq * dim
        s_attn_res = s_attn_out + max_seq * dim
        s_score = s_attn_res + max_seq * dim
        s_prob = s_score + max_seq
        s_temp = s_prob + max_seq
        s_mask = s_temp + max_seq
        s_vt = s_mask + max_seq * nt * dim
        phase2 = 0x3000
        s_h2 = phase2
        s_gate = s_h2 + max_seq * dim
        s_up = s_gate + max_seq * dim
        s_ffn_res = s_up + max_seq * dim
        s_last_h = s_ffn_res + max_seq * dim
        s_temp2 = s_last_h + max_seq * dim

        regions = [
            TensorRegion("EMBEDDING", 0x000, 40, "parameter", reuse_policy=invariant),
            TensorRegion("RMSNORM1", 0x100, 4, "parameter", reuse_policy=invariant),
            TensorRegion("RMSNORM2", 0x200, 4, "parameter", reuse_policy=invariant),
            TensorRegion("RMSNORM_FINAL", 0x300, 4, "parameter", reuse_policy=invariant),
            TensorRegion("W_Q", 0x400, 16, "weight", tile_stride=dim * dim, reuse_policy=weight),
            TensorRegion("W_KV", 0x500, 16, "weight", tile_stride=dim * dim, reuse_policy=weight),
            TensorRegion("Q_NORM", 0x600, 4, "parameter", reuse_policy=invariant),
            TensorRegion("K_NORM", 0x700, 4, "parameter", reuse_policy=invariant),
            TensorRegion("W_GATE", 0x800, 16, "weight", tile_stride=dim * dim, reuse_policy=weight),
            TensorRegion("W_UP", 0x900, 16, "weight", tile_stride=dim * dim, reuse_policy=weight),
            TensorRegion("W_DOWN", 0xA00, 16, "weight", tile_stride=dim * dim, reuse_policy=weight),
            TensorRegion("ROPE_TABLE", 0xB00, 140, "parameter", position_stride=dim, reuse_policy=invariant),
            TensorRegion("EMBED_B", 0xC00, 4, "parameter", reuse_policy=invariant),
            TensorRegion("W_Q_T", 0xD00, 16, "weight", tile_stride=dim * dim, reuse_policy=weight),
            TensorRegion("PHASE_FLAG", 0x1F00, 1, "control"),
            TensorRegion("X", s_x, max_seq * dim, "input", position_stride=dim),
            TensorRegion("Q", s_q, max_seq * dim, "attention", position_stride=dim),
            TensorRegion("K", s_k, max_seq * dim, "attention", position_stride=dim),
            TensorRegion("V", s_v, max_seq * dim, "attention", position_stride=dim),
            TensorRegion("ATTENTION_CONTEXT", s_ctx, max_seq * dim, "attention", position_stride=dim),
            TensorRegion("ATTENTION_OUTPUT", s_attn_out, max_seq * dim, "attention", position_stride=dim),
            TensorRegion("ATTENTION_RESIDUAL", s_attn_res, max_seq * dim, "intermediate", position_stride=dim),
            TensorRegion("ATTENTION_SCORE", s_score, max_seq, "attention"),
            TensorRegion("ATTENTION_PROB", s_prob, max_seq, "attention"),
            TensorRegion("ATTENTION_TEMP", s_temp, max_seq, "scratch"),
            TensorRegion("CAUSAL_MASK", s_mask, max_seq * nt * dim, "mask", tile_stride=dim),
            TensorRegion("V_TRANSPOSED", s_vt, nt * dim * dim, "attention", tile_stride=dim),
            TensorRegion("FFN_NORM_INPUT", s_h2, max_seq * dim, "input", position_stride=dim),
            TensorRegion("FFN_GATE", s_gate, max_seq * dim, "ffn", position_stride=dim),
            TensorRegion("FFN_UP", s_up, max_seq * dim, "ffn", position_stride=dim),
            TensorRegion("FFN_OUTPUT", s_ffn_res, max_seq * dim, "ffn", position_stride=dim),
            TensorRegion("LAYER_OUTPUT", s_last_h, max_seq * dim, "output", position_stride=dim),
            TensorRegion("FFN_TEMP", s_temp2, max_seq * dim, "scratch", position_stride=dim),
            TensorRegion("FINAL_HIDDEN", 0x4000, dim, "output"),
        ]
        return regions


_ADAPTERS: dict[str, Callable[[dict[str, int]], DagWorkloadAdapter]] = {
    "bert": BertDagWorkloadAdapter,
    "adder": Adder140pDagWorkloadAdapter,
    "adder_140p": Adder140pDagWorkloadAdapter,
}


def register_dag_workload(
    name: str,
    factory: Callable[[dict[str, int]], DagWorkloadAdapter],
) -> None:
    """Register an out-of-tree workload adapter for the current process."""

    if not name or not callable(factory):
        raise ValueError("workload adapter registration needs a name and factory")
    _ADAPTERS[name] = factory


def get_dag_workload(
    name: str | None,
    config: dict[str, int],
) -> DagWorkloadAdapter:
    normalized = (name or "bert").strip().lower().replace("-", "_")
    try:
        factory = _ADAPTERS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"unknown DAG workload {name!r}; registered workloads: {supported}"
        ) from exc
    return factory(config)


def available_dag_workloads() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
