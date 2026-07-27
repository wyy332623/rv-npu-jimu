/* NPU — Firmware MMIO Driver
 *
 * Provides the interface between RISC-V firmware and the NPU hardware.
 * The NPU is memory-mapped at 0x80000000.
 *
 * Chain-aware dispatch:
 *   - npu_send_inst() pushes instructions into the FIFO without polling
 *     for FULL (the FIFO is deep enough for one chain group).
 *   - npu_issue_chain() commits the current chain group via INST_ISSUE.
 *   - npu_wait_chain() polls CHAIN_STATUS until all functional units
 *     (VMM, MMM, MVU, pipe) are idle.
 */

#include <stdint.h>
#include "npu_regs.h"
#include "npu_isa.h"

/* Helper: MMIO base address as pointer */
#define NPU_MMIO_PTR ((volatile uint32_t*)(uintptr_t)NPU_MMIO_BASE)

/* Send one NPU instruction word — no FIFO-full stall.
 * Instructions flow freely into the FIFO.  The caller must
 * issue npu_issue_chain() + npu_wait_chain() at chain boundaries
 * to ensure completion before the next chain depends on results.
 */
void npu_send_inst(uint32_t inst)
{
    volatile uint32_t* base = NPU_MMIO_PTR;
    base[NPU_INST_FIFO / 4] = inst;
}

/* Issue the current chain group: send INST_ISSUE to commit all
 * preceding instructions as one parallel-dispatch group.
 */
void npu_issue_chain(void)
{
    npu_send_inst(SI(OP_INST_ISSUE, 0, 0));
}

/* Poll CHAIN_STATUS until all functional units are idle.
 * Bit layout: bit0=VMM (vector math), bit1=MMM (matrix memory), bit2=MVU (matrix-vector).
 * Returns when all bits are 0 (all units idle).
 */
void npu_wait_chain(void)
{
    volatile uint32_t* base = NPU_MMIO_PTR;
    while (base[NPU_CHAIN_STATUS / 4] != 0);
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

/* Wait for NPU to complete current operation (legacy — polls STATUS_DONE).
 * Prefer npu_issue_chain() + npu_wait_chain() for chain-aware code.
 */
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
