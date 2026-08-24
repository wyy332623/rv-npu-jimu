from pathlib import Path
import re

import numpy as np
import pytest

from emulator.npu_device_mini import (
    MEM_ADDSUB_VRF_2,
    MEM_MULTIPLY_VRF,
    NpuDeviceMini,
    NpuMemoryAccessError,
    OP_V_RD,
    OP_V_WR,
)


def test_addsub_vrf_2_id_matches_firmware_header():
    header = Path("firmware/npu_isa.h").read_text()
    assert MEM_ADDSUB_VRF_2 == 9
    assert re.search(r"\bMEM_ADDSUB_VRF_2\s*=\s*9\b", header)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_unknown_vrf_bank_is_rejected(operation):
    npu = NpuDeviceMini(native_dim=4)
    npu._pipeline = np.ones(4, dtype=np.float32)

    with pytest.raises(NpuMemoryAccessError, match="unknown VRF bank 255"):
        if operation == "read":
            npu._v_rd(OP_V_RD, 255, 0)
        else:
            npu._v_wr(OP_V_WR, 255, 0)


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("addr", [-1, 62])
def test_partial_or_negative_vrf_access_is_rejected(operation, addr):
    npu = NpuDeviceMini(native_dim=4)
    npu._pipeline = np.ones(4, dtype=np.float32)

    with pytest.raises(NpuMemoryAccessError, match="exceeds"):
        if operation == "read":
            npu._v_rd(OP_V_RD, MEM_MULTIPLY_VRF, addr)
        else:
            npu._v_wr(OP_V_WR, MEM_MULTIPLY_VRF, addr)
