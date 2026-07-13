// NPU — Register Map
//
// Shared between firmware (C), HDL (Amaranth), and tests (Python).
// All NPU registers are memory-mapped at NPU_MMIO_BASE.

#ifndef NPU_REGS_H
#define NPU_REGS_H

#define NPU_MMIO_BASE          0x80000000UL

// --- Control registers ---
#define NPU_INST_FIFO          0x00    // [WO] Push 32-bit instruction word
#define NPU_STATUS             0x04    // [RO] NPU status
#define NPU_RESET              0x08    // [WO] Write 1 to reset
#define NPU_CHAIN_STATUS       0x0C    // [RO] Per-unit busy: bit0=VMM, bit1=MMM, bit2=MVU

// --- Status bits (read from NPU_STATUS) ---
//   bit 0: BUSY  (1 = busy)
//   bit 1: DONE  (1 = idle, last operation completed)
//   bit 2: FULL  (1 = instruction FIFO full, backpressure)
#define NPU_STATUS_BUSY        0x01
#define NPU_STATUS_DONE        0x02
#define NPU_STATUS_FULL        0x04    // Instruction FIFO full — backpressure signal
#define NPU_STATUS_ERROR       0xFF

// --- Scalar register addresses (used via S_WR/S_RD) ---
#define REG_TILE_ROWS          1
#define REG_TILE_COLS          2
#define REG_ITERATIONS         3
#define REG_READ_VECTOR_MASK   15
#define REG_WRITE_VECTOR_MASK  16
#define REG_READ_MATRIX_MASK   17
#define REG_PRECISION_MODE     20

// --- Data registers ---
#define NPU_DATA_IN_ADDR       0x10    // [WO] Input tensor address
#define NPU_DATA_OUT_ADDR      0x14    // [WO] Output tensor address
#define NPU_DATA_IN_SIZE       0x18    // [WO] Input size in bytes
#define NPU_DATA_OUT_SIZE      0x1C    // [RO] Output size in bytes

// --- Data exchange registers ---
#define NPU_REG_HIDDEN_SIZE    0x20    // [RW] hidden_size set by test harness
#define NPU_REG_SEQ_LEN        0x24    // [RW] sequence length set by test harness

// --- DRAM window (CPU ↔ NPU DRAM) ---
// CPU reads/writes NPU DRAM at 32-bit float granularity.
// Address: NPU_DRAM_BASE + float_offset*4
#define NPU_DRAM_BASE          0x40    // [RW] Base of DRAM MMIO window
#define NPU_DRAM_SIZE          0x7FC0  //   32704 bytes = 8176 floats (scratch up to ~0x7FC0)

// --- SPU SRF window (CPU ↔ NPU scalar register file) ---
// CPU reads/writes NPU SPU SRF at 32-bit float granularity.
// Address: NPU_SRF_BASE + srf_index*4
#define NPU_SRF_BASE           0x8000  // [RW] Base of SPU SRF MMIO window
#define NPU_SRF_SIZE           0x100   //   256 bytes = 64 registers (SRF[0..63])

#define NPU_SKU_ID             0xF0    // [RO] SKU identifier
#define NPU_VERSION            0xF4    // [RO] Version

// --- NPU internal register addresses (written via s_wr / read via s_rd) ---
#define REG_TILE_ROWS          1       // Number of tile rows
#define REG_TILE_COLS          2       // Number of tile columns
#define REG_ITERATIONS         3       // Number of MVU iterations
#define REG_VECTOR_LENGTH      10      // Vector length
#define REG_MVM_VECTOR_START   11      // MVM vector start index
#define REG_MVM_VECTOR_LENGTH  12      // MVM vector length

#endif // NPU_REGS_H
