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
_TILE_STRIDE ?= 8
_SAVE_Q_BASE ?= 0x200
_SAVE_K_BASE ?= 0x300
_SAVE_V_BASE ?= 0x400
_ATTENTION_SCRATCH ?= 0x500
_SCRATCH_Z ?= 0x600
_SCRATCH_LN1 ?= 0x620
_SCRATCH_GELU ?= 0x640
_SO_SCRATCH ?= 0x580
_SAVE_RES_BASE ?= 0x700
_SAVE_OUT_BASE ?= 0x800
_UNIT_VEC_BASE ?= 0x900
NUM_HEAD ?= 2

CFLAGS = -Os -g -DNATIVE_DIM=$(NATIVE_DIM) \
         -D_PROJ_BASE=$(_PROJ_BASE) -D_MAT_SIZE=$(_MAT_SIZE) \
         -D_STRIDE=$(_STRIDE) -D_NUM_TILES=$(_NUM_TILES) \
         -D_HIDDEN_SIZE=$(_HIDDEN_SIZE) -D_SEQ_LEN=$(SEQ_LEN) \
         -D_LN1_GAMMA=$(_LN1_GAMMA) -D_LN1_BETA=$(_LN1_BETA) \
         -D_LN2_GAMMA=$(_LN2_GAMMA) -D_LN2_BETA=$(_LN2_BETA) \
         -D_SCRATCH=$(_SCRATCH) -D_TILE_STRIDE=$(_TILE_STRIDE) \
         -D_SAVE_Q_BASE=$(_SAVE_Q_BASE) -D_SAVE_K_BASE=$(_SAVE_K_BASE) \
         -D_SAVE_V_BASE=$(_SAVE_V_BASE) \
         -D_ATTENTION_SCRATCH=$(_ATTENTION_SCRATCH) \
         -D_SCRATCH_Z=$(_SCRATCH_Z) -D_SCRATCH_LN1=$(_SCRATCH_LN1) \
         -D_SCRATCH_GELU=$(_SCRATCH_GELU) -D_SO_SCRATCH=$(_SO_SCRATCH) \
         -D_SAVE_RES_BASE=$(_SAVE_RES_BASE) \
         -D_SAVE_OUT_BASE=$(_SAVE_OUT_BASE) \
         -D_UNIT_VEC_BASE=$(_UNIT_VEC_BASE) -DNUM_HEAD=$(NUM_HEAD) \
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
