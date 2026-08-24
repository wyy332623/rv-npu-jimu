# NPU — Firmware Optimization
#
# Top-level Makefile for common development tasks.

.PHONY: all kernels firmware opencode test clean clean-run

BUILD_DIR ?= _build

all: kernels firmware

# -----------------------------------------------------------------------
# C Kernel Library (libnpukernels.so)
# -----------------------------------------------------------------------
kernels:
	cmake -B $(BUILD_DIR)/kernels -S kernels -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR)/kernels
	@ln -sf $(BUILD_DIR)/kernels/libnpukernels.so .

# -----------------------------------------------------------------------
# RISC-V Firmware ELFs
# -----------------------------------------------------------------------
firmware:
	$(MAKE) -C firmware BUILD_DIR=$(abspath $(BUILD_DIR))/firmware

# -----------------------------------------------------------------------
# OpenCode agent configuration (auto-generated from skill sources)
# -----------------------------------------------------------------------
opencode:
	python3 jimu-dse/scripts/skillctl.py sync
	python3 jimu-dse/scripts/skillctl.py verify
	@echo "✅ OpenCode agent configured (skills installed, permissions from global config)"

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
test:
	python3 -m pytest tests/ -v

# -----------------------------------------------------------------------
# Clean — preserves virtual environments, run history, and generated skills
# -----------------------------------------------------------------------
clean:
	bash jimu-dse/scripts/clean_workspace.sh --apply

clean-run:
	@test -n "$(RUN_DIR)" || { echo "RUN_DIR is required" >&2; exit 2; }
	bash jimu-dse/scripts/clean_workspace.sh --apply --run-dir "$(RUN_DIR)"

# -----------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  all       Build kernels + firmware"
	@echo "  kernels   Build C kernel library (libnpukernels.so)"
	@echo "  firmware  Build RISC-V firmware ELFs"
	@echo "  opencode  Configure OpenCode agent (skills + permissions)"
	@echo "  test      Run all tests"
	@echo "  clean     Remove reproducible build/cache/backup files"
	@echo "  clean-run Remove one exact run directory (requires RUN_DIR=...)"
