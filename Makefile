# NPU — Firmware Optimization
#
# Top-level Makefile for common development tasks.

.PHONY: all kernels firmware opencode test clean

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
.opencode/skills/%/SKILL.md: jimu-dse/docs/skills/isa/%.md
	@mkdir -p $(dir $@)
	cp $< $@

opencode: .opencode/skills/dag-analyze/SKILL.md .opencode/skills/vrf-cache/SKILL.md .opencode/skills/dim-optimize/SKILL.md
	@echo "✅ OpenCode agent configured (skills installed, permissions from global config)"

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
test:
	python3 -m pytest tests/ -v

# -----------------------------------------------------------------------
# Clean — removes build artifacts and run results, keeps source files
# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR) libnpukernels.so _out jimu-dse/results .opencode
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf firmware/build_dim*/

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
	@echo "  clean     Remove build artifacts, run results, agent config"
