"""
MiniRV64 — Minimal RISC-V RV64 ISS for NPU Firmware Simulation.

A lightweight instruction set simulator that:
  - Loads a bare-metal ELF file
  - Executes RV64IM instructions (the subset used by compiled firmware)
  - Traps MMIO load/store to a Python device handler
  - Runs until a configurable cycle limit or the firmware halts

This is a fallback when the full PySpike (Spike) build is unavailable.
The MMIO device API matches what PySpike's MMIO devices expect.
"""

import struct
from pathlib import Path
from typing import Optional, Callable


# RV64IM opcode masks
_OP_MASK  = 0x7F
_OP_SHIFT = 0
_FUNCT3_MASK = 0x7000
_FUNCT3_SHIFT = 12
_FUNCT7_MASK = 0xFE000000
_FUNCT7_SHIFT = 25


class MiniRV64:
    """Minimal RV64IM RISC-V instruction set simulator."""

    def __init__(self, mmio_base: int = 0x80000000, mmio_size: int = 0x10000):
        self.pc = 0
        self.regs = [0] * 32
        self.mem = bytearray(16 * 1024 * 1024)  # 16 MB system memory
        self.mmio_base = mmio_base
        self.mmio_size = mmio_size
        self.mmio_dev: Optional[Callable] = None  # load(addr, size) -> bytes, store(addr, data) -> None
        self.halted = False
        self.cycle_count = 0
        self.inst_count = 0

    def load_elf(self, path: str):
        """Load a RISC-V ELF file into memory."""
        from elftools.elf.elffile import ELFFile
        with open(path, 'rb') as f:
            elf = ELFFile(f)
            for segment in elf.iter_segments():
                if segment['p_type'] == 'PT_LOAD':
                    addr = segment['p_paddr']
                    data = segment.data()
                    self.write_mem(addr, data)

            # Entry point
            self.pc = elf.header.e_entry

    def set_mmio_device(self, dev):
        """Set the MMIO device handler (must have load/store methods)."""
        self.mmio_dev = dev

    # -- Memory access --
    def _mmio_hit(self, addr: int) -> bool:
        return self.mmio_base <= addr < self.mmio_base + self.mmio_size

    def read_mem(self, addr: int, size: int) -> bytes:
        if self._mmio_hit(addr):
            if self.mmio_dev:
                return self.mmio_dev.load(addr - self.mmio_base, size)
            return b'\x00' * size
        offset = addr
        if offset + size > len(self.mem):
            return b'\x00' * size
        return bytes(self.mem[offset:offset + size])

    def write_mem(self, addr: int, data: bytes):
        if self._mmio_hit(addr):
            if self.mmio_dev:
                self.mmio_dev.store(addr - self.mmio_base, data)
            return
        offset = addr
        for i, b in enumerate(data):
            if offset + i < len(self.mem):
                self.mem[offset + i] = b

    def read_u8(self, addr: int) -> int:
        return self.read_mem(addr, 1)[0]

    def read_u16(self, addr: int) -> int:
        return struct.unpack('<H', self.read_mem(addr, 2))[0]

    def read_u32(self, addr: int) -> int:
        return struct.unpack('<I', self.read_mem(addr, 4))[0]

    def read_u64(self, addr: int) -> int:
        return struct.unpack('<Q', self.read_mem(addr, 8))[0]

    def write_u64(self, addr: int, val: int):
        self.write_mem(addr, struct.pack('<Q', val & 0xFFFFFFFFFFFFFFFF))

    def write_u32(self, addr: int, val: int):
        self.write_mem(addr, struct.pack('<I', val & 0xFFFFFFFF))

    def write_u16(self, addr: int, val: int):
        self.write_mem(addr, struct.pack('<H', val & 0xFFFF))

    def write_u8(self, addr: int, val: int):
        self.write_mem(addr, bytes([val & 0xFF]))

    # -- Instruction execution --
    def _sext(self, val: int, bits: int) -> int:
        """Sign-extend value to 64 bits."""
        if val & (1 << (bits - 1)):
            return val | (~0 << bits)
        return val

    def step(self):
        """Execute one instruction."""
        if self.halted:
            return

        inst = self.read_u32(self.pc)
        opcode = inst & _OP_MASK

        rd = (inst >> 7) & 0x1F
        funct3 = (inst >> 12) & 0x7
        rs1 = (inst >> 15) & 0x1F
        rs2 = (inst >> 20) & 0x1F
        funct7 = (inst >> 25) & 0x7F

        next_pc = self.pc + 4
        self.inst_count += 1

        # ---- RV64I Base Instruction Set ----
        if opcode == 0x37:  # LUI
            self.regs[rd] = self._sext(inst & 0xFFFFF000, 32)
        elif opcode == 0x17:  # AUIPC
            self.regs[rd] = self.pc + self._sext(inst & 0xFFFFF000, 32)
        elif opcode == 0x6F:  # JAL
            imm = ((inst >> 31) & 0x1) << 20 | \
                  ((inst >> 12) & 0xFF) << 12 | \
                  ((inst >> 20) & 0x1) << 11 | \
                  ((inst >> 21) & 0x3FF) << 1
            self.regs[rd] = self.pc + 4
            next_pc = self.pc + self._sext(imm, 21)
        elif opcode == 0x67:  # JALR
            imm = self._sext((inst >> 20) & 0xFFF, 12)
            tmp = self.regs[rs1]  # read rs1 BEFORE writing rd (in case rd == rs1)
            self.regs[rd] = self.pc + 4
            next_pc = (tmp + imm) & ~1
        elif opcode == 0x63:  # BRANCH
            imm = ((inst >> 31) & 0x1) << 12 | \
                  ((inst >> 7) & 0x1) << 11 | \
                  ((inst >> 25) & 0x3F) << 5 | \
                  ((inst >> 8) & 0xF) << 1
            imm = self._sext(imm, 13)
            rs1_v = self.regs[rs1]
            rs2_v = self.regs[rs2]
            taken = False
            if funct3 == 0:   # BEQ
                taken = rs1_v == rs2_v
            elif funct3 == 1:  # BNE
                taken = rs1_v != rs2_v
            elif funct3 == 4:  # BLT
                taken = rs1_v < rs2_v
            elif funct3 == 5:  # BGE
                taken = rs1_v >= rs2_v
            elif funct3 == 6:  # BLTU
                taken = (rs1_v & 0xFFFFFFFF) < (rs2_v & 0xFFFFFFFF)
            elif funct3 == 7:  # BGEU
                taken = (rs1_v & 0xFFFFFFFF) >= (rs2_v & 0xFFFFFFFF)
            if taken:
                next_pc = self.pc + imm
        elif opcode == 0x03:  # LOAD
            imm = self._sext((inst >> 20) & 0xFFF, 12)
            addr = self.regs[rs1] + imm
            if funct3 == 0:   # LB
                self.regs[rd] = self._sext(self.read_u8(addr), 8)
            elif funct3 == 1:  # LH
                self.regs[rd] = self._sext(self.read_u16(addr), 16)
            elif funct3 == 2:  # LW
                self.regs[rd] = self._sext(self.read_u32(addr), 32)
            elif funct3 == 3:  # LD
                self.regs[rd] = self.read_u64(addr)
            elif funct3 == 4:  # LBU
                self.regs[rd] = self.read_u8(addr)
            elif funct3 == 5:  # LHU
                self.regs[rd] = self.read_u16(addr)
            elif funct3 == 6:  # LWU
                self.regs[rd] = self.read_u32(addr)
        elif opcode == 0x23:  # STORE
            imm = ((inst >> 7) & 0x1F) | ((inst >> 25) & 0x3F) << 5
            imm = self._sext(imm, 12)
            addr = self.regs[rs1] + imm
            val = self.regs[rs2]
            if funct3 == 0:   # SB
                self.write_u8(addr, val)
            elif funct3 == 1:  # SH
                self.write_u16(addr, val)
            elif funct3 == 2:  # SW
                self.write_u32(addr, val)
            elif funct3 == 3:  # SD
                self.write_u64(addr, val)
        elif opcode == 0x13:  # OP-IMM
            imm = self._sext((inst >> 20) & 0xFFF, 12)
            if funct3 == 0:    # ADDI
                self.regs[rd] = self.regs[rs1] + imm
            elif funct3 == 1:  # SLLI
                shamt = ((inst >> 20) & 0x3F)
                self.regs[rd] = self.regs[rs1] << shamt
            elif funct3 == 2:  # SLTI
                self.regs[rd] = 1 if self.regs[rs1] < imm else 0
            elif funct3 == 3:  # SLTIU
                self.regs[rd] = 1 if (self.regs[rs1] & 0xFFFFFFFF) < (imm & 0xFFFFFFFF) else 0
            elif funct3 == 4:  # XORI
                self.regs[rd] = self.regs[rs1] ^ imm
            elif funct3 == 5:  # SRLI / SRAI
                shamt = ((inst >> 20) & 0x3F)
                if funct7 & 0x20:
                    self.regs[rd] = self.regs[rs1] >> shamt  # arithmetic (sign extend handles)
                else:
                    self.regs[rd] = (self.regs[rs1] & 0xFFFFFFFF) >> shamt
            elif funct3 == 6:  # ORI
                self.regs[rd] = self.regs[rs1] | imm
            elif funct3 == 7:  # ANDI
                self.regs[rd] = self.regs[rs1] & imm
        elif opcode == 0x33:  # OP
            if funct7 == 0:
                if funct3 == 0:    # ADD
                    self.regs[rd] = self.regs[rs1] + self.regs[rs2]
                elif funct3 == 1:  # SLL
                    self.regs[rd] = self.regs[rs1] << (self.regs[rs2] & 0x3F)
                elif funct3 == 2:  # SLT
                    self.regs[rd] = 1 if self.regs[rs1] < self.regs[rs2] else 0
                elif funct3 == 3:  # SLTU
                    self.regs[rd] = 1 if (self.regs[rs1] & 0xFFFFFFFF) < (self.regs[rs2] & 0xFFFFFFFF) else 0
                elif funct3 == 4:  # XOR
                    self.regs[rd] = self.regs[rs1] ^ self.regs[rs2]
                elif funct3 == 5:  # SRL
                    self.regs[rd] = (self.regs[rs1] & 0xFFFFFFFF) >> (self.regs[rs2] & 0x3F)
                elif funct3 == 6:  # OR
                    self.regs[rd] = self.regs[rs1] | self.regs[rs2]
                elif funct3 == 7:  # AND
                    self.regs[rd] = self.regs[rs1] & self.regs[rs2]
            elif funct7 == 0x20:
                if funct3 == 0:    # SUB
                    self.regs[rd] = self.regs[rs1] - self.regs[rs2]
                elif funct3 == 5:  # SRA
                    self.regs[rd] = self.regs[rs1] >> (self.regs[rs2] & 0x3F)
            elif funct7 == 0x01:  # RV64M: MUL
                if funct3 == 0:   # MUL
                    lo = (self.regs[rs1] & 0xFFFFFFFF) * (self.regs[rs2] & 0xFFFFFFFF)
                    self.regs[rd] = lo & 0xFFFFFFFF
                    if rd:
                        self.regs[rd] = self._sext(self.regs[rd] & 0xFFFFFFFF, 32)
                        # sign extend to 64
                    else:
                        self.regs[rd] = lo & 0xFFFFFFFF
                elif funct3 == 1:  # MULH
                    self.regs[rd] = self._sext((self.regs[rs1] * self.regs[rs2]) >> 64, 64)
                elif funct3 == 4:  # DIV
                    if self.regs[rs2] != 0:
                        self.regs[rd] = self.regs[rs1] / self.regs[rs2]
                elif funct3 == 5:  # DIVU
                    if self.regs[rs2] != 0:
                        self.regs[rd] = (self.regs[rs1] & 0xFFFFFFFF) / (self.regs[rs2] & 0xFFFFFFFF)
        elif opcode == 0x1B:  # OP-IMM-32 (ADDIW, SLLIW, SRLIW, SRAIW)
            imm = self._sext((inst >> 20) & 0xFFF, 12)
            if funct3 == 0:    # ADDIW
                self.regs[rd] = self._sext((self.regs[rs1] + imm) & 0xFFFFFFFF, 32)
            elif funct3 == 1:  # SLLIW
                shamt = ((inst >> 20) & 0x1F)
                self.regs[rd] = self._sext((self.regs[rs1] << shamt) & 0xFFFFFFFF, 32)
            elif funct3 == 5:  # SRLIW / SRAIW
                shamt = ((inst >> 20) & 0x1F)
                if funct7 & 0x20:  # SRAIW
                    self.regs[rd] = self._sext(self.regs[rs1] >> shamt, 32)
                else:  # SRLIW
                    self.regs[rd] = self._sext((self.regs[rs1] & 0xFFFFFFFF) >> shamt, 32)
        elif opcode == 0x3B:  # OP-32 (ADDW, SUBW, SLLW, SRLW, SRAW, MULW, DIVUW, REMUW)
            if funct7 == 0:
                if funct3 == 0:    # ADDW
                    self.regs[rd] = self._sext((self.regs[rs1] + self.regs[rs2]) & 0xFFFFFFFF, 32)
                elif funct3 == 1:  # SLLW
                    self.regs[rd] = self._sext((self.regs[rs1] << (self.regs[rs2] & 0x1F)) & 0xFFFFFFFF, 32)
                elif funct3 == 5:  # SRLW
                    self.regs[rd] = self._sext((self.regs[rs1] & 0xFFFFFFFF) >> (self.regs[rs2] & 0x1F), 32)
            elif funct7 == 0x20:
                if funct3 == 0:    # SUBW
                    self.regs[rd] = self._sext((self.regs[rs1] - self.regs[rs2]) & 0xFFFFFFFF, 32)
                elif funct3 == 5:  # SRAW
                    self.regs[rd] = self._sext(self.regs[rs1] >> (self.regs[rs2] & 0x1F), 32)
            elif funct7 == 1:  # RV64M word-size ops
                if funct3 == 0:    # MULW
                    lo = (self.regs[rs1] & 0xFFFFFFFF) * (self.regs[rs2] & 0xFFFFFFFF)
                    self.regs[rd] = self._sext(lo & 0xFFFFFFFF, 32)
                elif funct3 == 5:  # DIVUW
                    num = self.regs[rs1] & 0xFFFFFFFF
                    den = self.regs[rs2] & 0xFFFFFFFF
                    if den != 0:
                        self.regs[rd] = num // den
                    else:
                        self.regs[rd] = 0xFFFFFFFF
                elif funct3 == 6:  # REMUW
                    num = self.regs[rs1] & 0xFFFFFFFF
                    den = self.regs[rs2] & 0xFFFFFFFF
                    if den != 0:
                        self.regs[rd] = num % den
                    else:
                        self.regs[rd] = num
                    self.regs[rd] = self._sext((self.regs[rs1] & 0xFFFFFFFF) >> (self.regs[rs2] & 0x1F), 32)
        elif opcode == 0x73:  # SYSTEM (CSR, ECALL, EBREAK)
            if funct3 == 0:
                if rd == 0 and rs1 == 0 and funct7 == 0:  # ECALL
                    self.halted = True
                elif rd == 0 and rs1 == 1 and funct7 == 0:  # EBREAK
                    self.halted = True
            else:
                # CSR read/write — ignore for now (treat as nop)
                pass
        # else: unknown opcode — treat as nop or halt
        else:
            if opcode not in (0x37, 0x17, 0x6F, 0x67, 0x63, 0x03, 0x23, 0x13, 0x33, 0x1B, 0x3B, 0x73):
                self.halted = True  # unknown opcode

        # x0 is always zero
        self.regs[0] = 0
        self.pc = next_pc
        self.cycle_count += 1

    def run(self, cycles: int = 1000):
        """Run for up to `cycles` instructions or until halted."""
        for _ in range(cycles):
            if self.halted:
                break
            self.step()

    def read_register(self, idx: int) -> int:
        return self.regs[idx]

    def dump_state(self):
        print(f"PC={self.pc:#x} cycle={self.cycle_count} inst={self.inst_count} halted={self.halted}")
        for i in range(0, 32, 4):
            print(f"  x{i:02d}={self.regs[i]:#018x} x{i+1:02d}={self.regs[i+1]:#018x} "
                  f"x{i+2:02d}={self.regs[i+2]:#018x} x{i+3:02d}={self.regs[i+3]:#018x}")
