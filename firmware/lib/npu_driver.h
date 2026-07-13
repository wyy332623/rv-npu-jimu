/* NPU — Firmware MMIO Driver API */
#ifndef NPU_DRIVER_H
#define NPU_DRIVER_H

#include <stdint.h>

void npu_send_inst(uint32_t inst);
uint32_t npu_read_reg(uint32_t offset);
void npu_write_reg(uint32_t offset, uint32_t val);
void npu_wait_done(void);
void npu_set_done(void);

#endif /* NPU_DRIVER_H */
