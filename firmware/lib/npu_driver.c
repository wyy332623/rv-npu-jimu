/* NPU — Firmware MMIO Driver
 *
 * Provides the interface between RISC-V firmware and the NPU hardware.
 * The NPU is memory-mapped at 0x80000000.
 */

#include <stdint.h>
#include "npu_regs.h"

/* Helper: MMIO base address as pointer */
#define NPU_MMIO_PTR ((volatile uint32_t*)(uintptr_t)NPU_MMIO_BASE)

/* Send one NPU instruction word (blocks if FIFO full) */
void npu_send_inst(uint32_t inst)
{
    volatile uint32_t* base = NPU_MMIO_PTR;
    /* Wait until FIFO is not full */
    while (base[NPU_STATUS / 4] & NPU_STATUS_FULL);
    /* Write instruction */
    base[NPU_INST_FIFO / 4] = inst;
}

/* Read NPU register at MMIO offset */
uint32_t npu_read_reg(uint32_t offset)
{
    volatile uint32_t* base = NPU_MMIO_PTR;
    return base[offset / 4];
}

/* Write NPU register at MMIO offset */
void npu_write_reg(uint32_t offset, uint32_t val)
{
    volatile uint32_t* base = NPU_MMIO_PTR;
    base[offset / 4] = val;
}

/* Wait for NPU to complete current operation */
void npu_wait_done(void)
{
    volatile uint32_t* base = NPU_MMIO_PTR;
    while (!(base[NPU_STATUS / 4] & NPU_STATUS_DONE));
}

/* Send NPU status as done */
void npu_set_done(void)
{
    npu_write_reg(NPU_STATUS, NPU_STATUS_DONE);
}
