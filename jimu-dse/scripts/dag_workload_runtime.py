"""Runtime adapters used by DAG visualization and closed-loop probes.

Each adapter owns only workload-specific build and input preparation.  Event
collection, micro-op construction, dependency analysis, and report generation
remain shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class WorkloadConfig:
    dim: int
    hidden_size: int
    seq_len: int
    num_head: int

    def as_dict(self) -> dict[str, int]:
        return {
            "dim": self.dim,
            "hidden_size": self.hidden_size,
            "seq_len": self.seq_len,
            "num_head": self.num_head,
        }


@dataclass
class WorkloadRun:
    npu: Any
    tracer: Any
    recorder: Any
    phase_ranges: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkloadRuntime:
    name = "generic"
    make_target = ""
    supports_operator_graph = False
    supports_symbolic_graph = False

    def build(self, config: WorkloadConfig, *, build_kind: str = "graph") -> Path:
        raise NotImplementedError

    def run(self, config: WorkloadConfig, elf_path: Path) -> WorkloadRun:
        raise NotImplementedError

    def _run_make(
        self,
        config: WorkloadConfig,
        build_dir: str,
        env: dict[str, str],
    ) -> Path:
        full_env = {**os.environ, **env}
        result = subprocess.run(
            [
                "make",
                "-C",
                str(REPO_ROOT / "firmware"),
                f"BUILD_DIR={build_dir}",
                f"TARGET={self.make_target}",
                "clean",
                "all",
            ],
            capture_output=True,
            text=True,
            env=full_env,
        )
        if result.returncode != 0:
            detail = result.stderr or result.stdout
            raise RuntimeError(
                f"{self.name} firmware build failed (rc={result.returncode}):\n"
                + detail[-1000:]
            )
        elf_path = REPO_ROOT / "firmware" / build_dir / f"{self.make_target}.elf"
        if not elf_path.is_file():
            raise RuntimeError(f"firmware build did not produce {elf_path}")
        return elf_path


class BertRuntime(WorkloadRuntime):
    name = "bert"
    make_target = "bert"
    supports_operator_graph = True
    supports_symbolic_graph = True

    def build(self, config: WorkloadConfig, *, build_kind: str = "graph") -> Path:
        dim = config.dim
        hidden_size = config.hidden_size
        seq_len = config.seq_len
        num_tiles = hidden_size // dim
        proj_base = hidden_size * seq_len + 4
        mat_size = hidden_size * hidden_size
        stride = mat_size + hidden_size
        ln_base = proj_base + 6 * stride
        ln_size = num_tiles * 8
        prefix = "build_graph" if build_kind == "graph" else "build_probe"
        build_dir = f"{prefix}_dim{dim}_h{hidden_size}_seq{seq_len}"
        return self._run_make(
            config,
            build_dir,
            {
                "NATIVE_DIM": str(dim),
                "SEQ_LEN": str(seq_len),
                "_HIDDEN_SIZE": str(hidden_size),
                "_PROJ_BASE": str(proj_base),
                "_MAT_SIZE": str(mat_size),
                "_STRIDE": str(stride),
                "_NUM_TILES": str(num_tiles),
                "_LN1_GAMMA": str(ln_base),
                "_LN1_BETA": str(ln_base + ln_size),
                "_LN2_GAMMA": str(ln_base + 2 * ln_size),
                "_LN2_BETA": str(ln_base + 3 * ln_size),
                "_SCRATCH": str(0x500),
                "NUM_HEAD": str(config.num_head),
            },
        )

    @staticmethod
    def _params(config: WorkloadConfig) -> dict[str, Any]:
        from tests.gen_golden_bert import bert_encoder_layer

        _golden, params = bert_encoder_layer(
            add_mask=False,
            num_head=config.num_head,
            head_size=config.hidden_size // config.num_head,
            hidden_size=config.hidden_size,
            seq_len=config.seq_len,
            native_dim=config.dim,
            precision="emulator_float32",
            seed=42,
        )
        return params

    @staticmethod
    def _load_dram(npu: Any, config: WorkloadConfig, params: dict[str, Any]) -> None:
        from emulator.npu_device_mini import MEM_DRAM

        dim = config.dim
        hidden_size = config.hidden_size
        seq_len = config.seq_len
        num_tiles = hidden_size // dim
        input_elements = hidden_size * seq_len
        npu._vrf[MEM_DRAM][0:input_elements] = np.zeros(
            input_elements, dtype=np.float32
        )
        proj_base = input_elements + 4
        mat_size = hidden_size * hidden_size
        stride = mat_size + hidden_size

        def tiled_copy(dst: int, matrix: np.ndarray) -> None:
            blocks = []
            for tile_row in range(matrix.shape[0] // dim):
                for tile_col in range(matrix.shape[1] // dim):
                    blocks.append(
                        matrix[
                            tile_row * dim : (tile_row + 1) * dim,
                            tile_col * dim : (tile_col + 1) * dim,
                        ].flatten()
                    )
            data = np.concatenate(blocks)
            npu._vrf[MEM_DRAM][dst : dst + len(data)] = data

        def bias_copy(dst: int, bias: np.ndarray) -> None:
            data = bias.astype(np.float32).flatten()[:hidden_size]
            npu._vrf[MEM_DRAM][dst : dst + len(data)] = data

        for index, name in enumerate(("Q", "K", "V", "selfoutput")):
            offset = proj_base + index * stride
            tiled_copy(offset, params[name]["W"].astype(np.float32))
            bias_copy(offset + mat_size, params[name]["b"])
        for index, (weight, bias) in enumerate(
            (("W_intmfc", "b_intmfc"), ("W_outfc", "b_outfc")), start=4
        ):
            offset = proj_base + index * stride
            tiled_copy(offset, params[weight].astype(np.float32))
            bias_copy(offset + mat_size, params[bias])

        for lane in range(dim):
            unit = np.zeros(dim, dtype=np.float32)
            unit[lane] = 1.0
            base = 0x900 + lane * dim
            npu._vrf[MEM_DRAM][base : base + dim] = unit

        ln_base = proj_base + 6 * stride
        ln_size = num_tiles * 8
        values = (
            params["LayerNorm"]["W"][0],
            params["LayerNorm"]["b"][0],
            params["LayerNorm"]["W"][1],
            params["LayerNorm"]["b"][1],
        )
        for index, vector in enumerate(values):
            destination = ln_base + index * ln_size
            flat = vector.astype(np.float32).flatten()
            for tile_row in range(num_tiles):
                chunk = flat[tile_row * dim : (tile_row + 1) * dim]
                start = destination + tile_row * 8
                npu._vrf[MEM_DRAM][start : start + len(chunk)] = chunk

    def run(self, config: WorkloadConfig, elf_path: Path) -> WorkloadRun:
        from emulator.npu_device_mini import NpuDeviceMini
        from emulator.npu_event_trace import EventTracer
        from emulator.trace_recorder import TraceRecorder
        from iss.mini_rv64 import MiniRV64

        npu = NpuDeviceMini(native_dim=config.dim)
        npu.set_hidden_size(config.hidden_size)
        npu.set_seq_len(config.seq_len)
        self._load_dram(npu, config, self._params(config))
        tracer = EventTracer(npu)
        recorder = TraceRecorder(npu)
        cpu = MiniRV64()
        cpu.set_mmio_device(recorder)
        cpu.load_elf(str(elf_path))
        cpu.run(cycles=300_000)
        return WorkloadRun(npu=npu, tracer=tracer, recorder=recorder)


class Adder140pRuntime(WorkloadRuntime):
    name = "adder_140p"
    make_target = "adder_140p"

    def build(self, config: WorkloadConfig, *, build_kind: str = "graph") -> Path:
        prefix = "build_graph" if build_kind == "graph" else "build_probe"
        build_dir = f"{prefix}_adder140p_dim{config.dim}_seq{config.seq_len}"
        return self._run_make(
            config,
            build_dir,
            {
                "NATIVE_DIM": str(config.dim),
                "SEQ_LEN": str(config.seq_len),
            },
        )

    @staticmethod
    def _tokens(
        dram: np.ndarray,
        seq_len: int,
        *,
        synthetic_weights: bool = False,
    ) -> list[int]:
        from adderboard.golden.golden_140p import forward
        from adderboard.layout.layout_140p import encode_prompt

        tokens = encode_prompt(5, 5)[:]
        if seq_len < len(tokens):
            return tokens[:seq_len]
        while len(tokens) < seq_len:
            tokens.append(
                0
                if synthetic_weights
                else int(np.argmax(forward(dram, tokens)))
            )
        return tokens

    @staticmethod
    def _trace_dram() -> tuple[np.ndarray, bool]:
        """Load trained weights, or structural values when torch is absent.

        Synthetic values are legal only for trace/DAG generation.  The closed
        loop correctness commands still load the real checkpoint and therefore
        fail rather than silently validating against these values.
        """

        from adderboard.layout.layout_140p import ADDR_MAP, build_dram

        try:
            dram, _metadata = build_dram()
            return dram, False
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
        dram = np.zeros(524288, dtype=np.float32)
        for name in ("norm1", "norm2", "norm_final", "q_norm", "k_norm"):
            base = ADDR_MAP[name]
            dram[base : base + 4] = 1.0
        return dram, True

    def run(self, config: WorkloadConfig, elf_path: Path) -> WorkloadRun:
        if config.dim != 4 or config.hidden_size != 4 or config.num_head != 1:
            raise ValueError(
                "adder_140p requires dim=4, hidden_size=4, num_head=1"
            )
        from adderboard.golden.golden_140p import (
            _load_weights,
            apply_rope_numpy,
            rms_norm,
        )
        from adderboard.layout.layout_140p import (
            HEAD_DIM,
            MODEL_DIM,
        )
        from emulator.npu_device_mini import MEM_DRAM
        from emulator.npu_event_trace import EventTracer
        from emulator.npu_fp32 import NpuFP32
        from emulator.trace_recorder import TraceRecorder
        from iss.mini_rv64 import MiniRV64

        max_seq = 35
        s_x = 0x2000
        s_q = s_x + max_seq * MODEL_DIM
        s_k = s_q + max_seq * MODEL_DIM
        s_v = s_k + max_seq * MODEL_DIM
        s_ctx = s_v + max_seq * MODEL_DIM
        s_attn_out = s_ctx + max_seq * MODEL_DIM
        s_attn_res = s_attn_out + max_seq * MODEL_DIM
        s_score = s_attn_res + max_seq * MODEL_DIM
        s_prob = s_score + max_seq
        s_temp = s_prob + max_seq
        s_mask_table = s_temp + max_seq
        s_flags = 0x1F00
        s_base2 = 0x3000
        s_gate = s_base2 + max_seq * MODEL_DIM
        s_up = s_gate + max_seq * MODEL_DIM

        dram, synthetic_weights = self._trace_dram()
        tokens = self._tokens(
            dram,
            config.seq_len,
            synthetic_weights=synthetic_weights,
        )
        npu = NpuFP32(native_dim=config.dim)
        npu.load_dram(dram)
        npu.set_hidden_size(config.hidden_size)
        npu.set_seq_len(config.seq_len)
        weights = _load_weights(npu._vrf[MEM_DRAM])
        length = len(tokens)

        x = np.array([weights["embedding"][token] for token in tokens])
        h1 = rms_norm(x, weights["norm1"])
        q = h1 @ weights["W_q"].T
        kv = h1 @ weights["W_kv"].T
        k = kv.copy()
        v = kv.copy()
        q = rms_norm(q, weights["q_norm"])
        k = rms_norm(k, weights["k_norm"])
        positions = np.arange(length, dtype=np.int32)
        q = apply_rope_numpy(
            q.reshape(length, 1, HEAD_DIM), weights["rope_table"], positions
        ).reshape(length, HEAD_DIM)
        k = apply_rope_numpy(
            k.reshape(length, 1, HEAD_DIM), weights["rope_table"], positions
        ).reshape(length, HEAD_DIM)
        for position in range(length):
            start = position * MODEL_DIM
            npu._vrf[MEM_DRAM][s_x + start : s_x + start + MODEL_DIM] = x[position]
            npu._vrf[MEM_DRAM][s_q + start : s_q + start + MODEL_DIM] = q[position]
            npu._vrf[MEM_DRAM][s_k + start : s_k + start + MODEL_DIM] = k[position]
            npu._vrf[MEM_DRAM][s_v + start : s_v + start + MODEL_DIM] = v[position]

        tile_count = (length + MODEL_DIM - 1) // MODEL_DIM
        for query in range(length):
            for tile_col in range(tile_count):
                base = tile_col * MODEL_DIM
                valid = min(MODEL_DIM, length - base)
                mask = np.full(MODEL_DIM, -1e30, dtype=np.float32)
                for lane in range(valid):
                    if base + lane <= query:
                        mask[lane] = 0.0
                address = s_mask_table + (query * tile_count + tile_col) * MODEL_DIM
                npu._vrf[MEM_DRAM][address : address + MODEL_DIM] = mask

        transposed_v = s_mask_table + max_seq * tile_count * MODEL_DIM
        for tile_col in range(tile_count):
            base = tile_col * MODEL_DIM
            valid = min(MODEL_DIM, length - base)
            tile = np.zeros((MODEL_DIM, MODEL_DIM), dtype=np.float32)
            for lane in range(valid):
                tile[:, lane] = v[base + lane]
            address = transposed_v + tile_col * MODEL_DIM * MODEL_DIM
            npu._vrf[MEM_DRAM][address : address + MODEL_DIM * MODEL_DIM] = tile.flatten()

        tracer = EventTracer(npu)
        recorder = TraceRecorder(npu)
        npu._vrf[MEM_DRAM][s_flags] = np.frombuffer(
            np.uint32(0).tobytes(), dtype=np.float32
        )[0]
        npu._spu_srf[6] = HEAD_DIM ** -0.5
        cpu = MiniRV64()
        cpu.set_mmio_device(recorder)
        cpu.load_elf(str(elf_path))
        cpu.run(cycles=300_000)
        phase1_end = len(tracer.events)

        attn_res = np.zeros((length, MODEL_DIM), dtype=np.float32)
        for position in range(length):
            start = s_attn_res + position * MODEL_DIM
            attn_res[position] = npu._vrf[MEM_DRAM][start : start + MODEL_DIM]
        h2 = rms_norm(attn_res, weights["norm2"])
        gate = h2 @ weights["W_gate"].T
        up = h2 @ weights["W_up"].T
        for position in range(length):
            offset = position * MODEL_DIM
            npu._vrf[MEM_DRAM][s_base2 + offset : s_base2 + offset + MODEL_DIM] = h2[position]
            npu._vrf[MEM_DRAM][s_gate + offset : s_gate + offset + MODEL_DIM] = gate[position]
            npu._vrf[MEM_DRAM][s_up + offset : s_up + offset + MODEL_DIM] = up[position]

        npu._vrf[MEM_DRAM][s_flags] = np.frombuffer(
            np.uint32(1).tobytes(), dtype=np.float32
        )[0]
        cpu = MiniRV64()
        cpu.set_mmio_device(recorder)
        cpu.load_elf(str(elf_path))
        cpu.run(cycles=200_000)
        phase2_end = len(tracer.events)
        return WorkloadRun(
            npu=npu,
            tracer=tracer,
            recorder=recorder,
            phase_ranges=[
                {
                    "kind": "attention_and_residual",
                    "label": "Adder 140p firmware phase 1: attention/O/residual",
                    "event_start": 0,
                    "event_end": phase1_end,
                    "execution_domain": "npu_firmware",
                },
                {
                    "kind": "swiglu_ffn_and_residual",
                    "label": "Adder 140p firmware phase 2: SwiGLU/down/residual",
                    "event_start": phase1_end,
                    "event_end": phase2_end,
                    "execution_domain": "npu_firmware",
                },
            ],
            metadata={
                "token_count": length,
                "trace_data": (
                    "synthetic-structure-only"
                    if synthetic_weights
                    else "trained-checkpoint"
                ),
                "host_gap": "RMSNorm2 plus W_gate/W_up",
                "excluded_host_stages": [
                    "embedding",
                    "RMSNorm1",
                    "Q/KV projection",
                    "QK norm and RoPE",
                    "RMSNorm2 and gate/up projection",
                    "final RMSNorm and LM head",
                ],
            },
        )


_RUNTIMES: dict[str, Callable[[], WorkloadRuntime]] = {
    "bert": BertRuntime,
    "adder": Adder140pRuntime,
    "adder_140p": Adder140pRuntime,
}


def register_workload_runtime(
    name: str,
    factory: Callable[[], WorkloadRuntime],
) -> None:
    if not name or not callable(factory):
        raise ValueError("runtime registration needs a name and factory")
    _RUNTIMES[name] = factory


def get_workload_runtime(name: str | None) -> WorkloadRuntime:
    normalized = (name or "bert").strip().lower().replace("-", "_")
    try:
        return _RUNTIMES[normalized]()
    except KeyError as exc:
        supported = ", ".join(sorted(_RUNTIMES))
        raise ValueError(
            f"unknown DAG runtime {name!r}; registered workloads: {supported}"
        ) from exc


def write_build_metadata(
    elf_path: Path,
    config: WorkloadConfig,
    workload: str,
) -> Path:
    output = Path(str(elf_path) + ".jimu-build.json")
    payload = {
        **config.as_dict(),
        "workload": workload,
        "elf_path": str(elf_path),
        "elf_sha256": hashlib.sha256(elf_path.read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output
