# NPU — RISC-V Firmware Build
#
CC = riscv64-unknown-elf-gcc

NATIVE_DIM ?= 128
SEQ_LEN ?= 1
# DRAM layout macros are computed by the caller (_build_firmware in test)
# and passed via environment variables. Defaults for NATIVE_DIM=128, SEQ_LEN=1.
_HIDDEN_SIZE ?= 128
_PROJ_BASE ?= 132
_MAT_SIZE ?= 16384
_STRIDE ?= 16512
_NUM_TILES ?= 1
# LN param DRAM offsets (computed by test harness)
_LN1_GAMMA ?= 0
_LN1_BETA ?= 0
_LN2_GAMMA ?= 0
_LN2_BETA ?= 0
_SCRATCH ?= 0
NUM_HEAD ?= 2

CFLAGS = -Os -g -DNATIVE_DIM=$(NATIVE_DIM) \
         -D_PROJ_BASE=$(_PROJ_BASE) -D_MAT_SIZE=$(_MAT_SIZE) \
         -D_STRIDE=$(_STRIDE) -D_NUM_TILES=$(_NUM_TILES) \
         -D_HIDDEN_SIZE=$(_HIDDEN_SIZE) -D_SEQ_LEN=$(SEQ_LEN) \
         -D_LN1_GAMMA=$(_LN1_GAMMA) -D_LN1_BETA=$(_LN1_BETA) \
         -D_LN2_GAMMA=$(_LN2_GAMMA) -D_LN2_BETA=$(_LN2_BETA) \
         -D_SCRATCH=$(_SCRATCH) -DNUM_HEAD=$(NUM_HEAD) \
         -ffreestanding -nostdlib -nostartfiles -mno-relax \
         -march=rv64im -mabi=lp64 \
         -I. -Ilib
LDFLAGS = -T firmware.ld -nostartfiles -nostdlib -Wl,-e,_start -Wl,--no-dynamic-linker

BUILD_DIR ?= build
TARGET ?= bert
DIAG ?= 0

ifeq ($(TARGET),bert)
  ifeq ($(DIAG),1)
    SRCS = startup.S lib/npu_driver.c bert/batch_diag.c
  else
    SRCS = startup.S lib/npu_driver.c bert/bert_layer.c
  endif
  ELF = $(BUILD_DIR)/bert.elf
else
  $(error Unknown TARGET=$(TARGET), expected 'bert')
endif

# Map local sources to build-dir objects.
_local_objs = $(SRCS:%.S=$(BUILD_DIR)/%.o)
_local_objs := $(_local_objs:%.c=$(BUILD_DIR)/%.o)
OBJS = $(_local_objs)

.PHONY: all clean dirs

all: $(ELF)

$(BUILD_DIR)/%.o: %.S firmware.ld | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c | dirs
	$(CC) $(CFLAGS) -c $< -o $@

dirs:
	@mkdir -p $(BUILD_DIR) $(BUILD_DIR)/lib $(BUILD_DIR)/bert

$(BUILD_DIR)/bert.elf: $(OBJS) firmware.ld
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $@
	@echo "=== Firmware built: $@ ==="
	@file $@

clean:
	rm -rf $(BUILD_DIR)
