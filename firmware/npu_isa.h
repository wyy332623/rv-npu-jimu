// NPU — Instruction Set Architecture
//
// Instruction encoding matching the reference NPU ISA.
// Hand-written C header (no codegen required).

#ifndef NPU_ISA_H
#define NPU_ISA_H

#include <stdint.h>

// -----------------------------------------------------------------------
// Instruction formats
// -----------------------------------------------------------------------
// Standard Instruction (SI):  32 bits
//   Bit:  31..24   23..16   15..0
//        ┌────────┬────────┬────────┐
//        │ OpCode │  Opd0  │  Opd1  │
//        └────────┴────────┴────────┘
//
// Long operand (LO):  32 bits
//   Bit:  31..24   23..0
//        ┌────────┬──────────────────┐
//        │ OpCode │      Opd0        │
//        └────────┴──────────────────┘

#define SI(op, opd0, opd1) \
    (((uint32_t)(op)  << 24) | \
     ((uint32_t)(opd0) << 16) | \
     ((uint32_t)(opd1)))

#define LO(op, opd0) \
    (((uint32_t)(op) << 24) | \
     ((uint32_t)(opd0)))

// -----------------------------------------------------------------------
// Opcodes (matching reference NPU ISA)
// -----------------------------------------------------------------------
enum OpCode {
    // Scalar operations
    OP_S_WR            = 0,
    OP_S_RD            = 1,

    // Vector operations
    OP_V_RD            = 2,
    OP_M_RD            = 3,
    OP_V_WR            = 5,
    OP_M_WR            = 6,

    // Matrix-vector
    OP_MV_MUL          = 7,

    // Vector-vector arithmetic
    OP_VV_ADD          = 8,
    OP_VV_A_SUB_B      = 9,
    OP_VV_B_SUB_A      = 10,
    OP_VV_MUL          = 11,

    // Activation functions
    OP_V_SIGM          = 12,
    OP_V_TANH          = 13,
    OP_V_RELU          = 14,
    OP_VV_MAX          = 15,

    // Increment variants
    OP_V_RD_INC        = 16,
    OP_V_WR_INC        = 17,
    OP_VV_ADD_INC      = 18,
    OP_VV_MAX_INC      = 19,

    // DRAM operations
    OP_V_RD_DRAM       = 20,
    OP_V_WR_DRAM       = 21,
    OP_V_RD_DRAM_INC   = 22,
    OP_V_WR_DRAM_INC   = 23,
    OP_M_RD_DRAM       = 24,
    OP_M_WR_DRAM       = 25,
    OP_V_RD_3D         = 26,
    OP_MV_MUL_INC      = 27,

    // More vector ops
    OP_VV_MIN          = 30,
    OP_VV_MUL_INC      = 31,
    OP_V_EXP           = 37,
    OP_S_SQRT          = 38,
    OP_S_RECIP         = 35,
    OP_V_GELU          = 42,

    // Extended ops
    OP_V_FUNC          = 43,    // sub: 0=softmax, 1=layernorm
    OP_SS_ADD          = 44,
    OP_INST_ISSUE      = 45,
};

// -----------------------------------------------------------------------
// Sub-opcodes (for OP_V_FUNC)
// -----------------------------------------------------------------------
enum SubOpCode {
    SUB_SOFTMAX        = 0,
    SUB_LAYERNORM      = 1,
};

// -----------------------------------------------------------------------
// Memory targets (used as opd0 in v_rd, v_wr, m_rd, m_wr)
// -----------------------------------------------------------------------
enum MemTarget {
    MEM_DRAM            = 0,
    MEM_MULTIPLY_VRF    = 1,
    MEM_NET_OUTPUT_Q    = 2,
    MEM_NET_INPUT_Q     = 3,
    MEM_MATRIX_RF       = 4,
    MEM_MVM_INITIAL_VRF = 5,
    MEM_MFU_INITIAL_VRF = 6,
    MEM_ADDSUB_VRF_0    = 7,
    MEM_ADDSUB_VRF_1    = 8,
    MEM_ADDSUB_VRF_2    = 9,
    MEM_FILL            = 12,
    MEM_MVM_ACC_VRF     = 13,
    MEM_SPU_ADD_REDUCE  = 14,
    MEM_SPU_MAX_REDUCE  = 15,
    MEM_SPU_ABSMAX_REDUCE = 16,
    MEM_SPU_BROADCAST   = 17,
    MEM_VEC_TO_MAT_ROW  = 18,
};

// -----------------------------------------------------------------------
// Native dimension (from SKU)
// -----------------------------------------------------------------------
#ifndef NATIVE_DIM
#define NATIVE_DIM      128
#endif

// Register addresses (for OP_S_WR)

#endif // NPU_ISA_H
