from emulator.npu_event_trace import (
    EventTracer,
    OP_INST_ISSUE,
    OP_V_RD_DRAM,
    OP_V_RD_DRAM_INC,
)


class FakeDevice:
    native_dim = 4

    def _push_instruction(self, inst):
        opcode = (inst >> 24) & 0xFF
        operand = inst & 0xFFFFFF
        if opcode == OP_V_RD_DRAM_INC:
            self._execute(OP_V_RD_DRAM, 0, 0, operand)
            self._execute(OP_V_RD_DRAM, 0, 0, operand + self.native_dim)
        else:
            self._execute(opcode, 0, 0, operand)

    def _execute(
        self, opcode, opd0, opd1, full_operand=0, pipeline=None, vpipe_a=None
    ):
        return pipeline, vpipe_a


def raw(opcode, operand=0):
    return (opcode << 24) | operand


def test_event_tracer_records_chain_inc_parent_and_memory_span():
    device = FakeDevice()
    tracer = EventTracer(device)

    device._push_instruction(raw(OP_V_RD_DRAM_INC, 100))
    device._push_instruction(raw(OP_INST_ISSUE))
    device._push_instruction(raw(OP_V_RD_DRAM, 200))

    first, second, marker, after = tracer.events
    assert first["opcode"] == OP_V_RD_DRAM
    assert first["raw_instruction_idx"] == 0
    assert first["expanded_idx"] == 0
    assert second["expanded_idx"] == 1
    assert first["inc_parent_opcode"] == OP_V_RD_DRAM_INC
    assert first["chain_id"] == marker["chain_id"] == 0
    assert after["chain_id"] == 1
    assert first["memory"] == {
        "direction": "read",
        "address": 100,
        "elements": 4,
        "end_address": 104,
    }
    assert second["memory"]["address"] == 104

    tracer.unpatch()
