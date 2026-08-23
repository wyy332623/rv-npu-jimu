# NPU — Firmware Optimization
#
# Top-level Makefile for common development tasks.

.PHONY: all kernels firmware opencode timing-deps rtl-lint rtl-test test validate-goals list-goals clean clean-results help

BUILD_DIR ?= _build
VENV_PYTHON := $(firstword $(wildcard .venv/bin/python .venv/Scripts/python.exe))
PYTHON ?= $(if $(VENV_PYTHON),$(VENV_PYTHON),python3)

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

opencode: .opencode/skills/dag-analyze/SKILL.md .opencode/skills/vrf-cache/SKILL.md .opencode/skills/dim-optimize/SKILL.md .opencode/skills/cycle-latency/SKILL.md .opencode/skills/rtl-dataflow/SKILL.md .opencode/skills/dataflow-optimize/SKILL.md .opencode/skills/self-verify/SKILL.md
	@echo "✅ OpenCode agent configured (skills installed, permissions from global config)"

timing-deps:
	$(PYTHON) -m pip install -r requirements-timing.txt

rtl-lint:
	verilator --lint-only --Wall -Wno-fatal -Wno-WIDTHEXPAND rtl/jimu_npu_timing_core.sv

rtl-test:
	$(PYTHON) -m pytest tests/unit/test_npu_rtl_sim.py tests/unit/test_npu_rtl_optimization_space.py -v

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests/ -v

validate-goals:
	$(PYTHON) jimu-dse/scripts/closed_loop.py validate-config --goal dram-optimization
	$(PYTHON) jimu-dse/scripts/closed_loop.py validate-config --goal compute-optimization
	$(PYTHON) jimu-dse/scripts/closed_loop.py validate-config --goal combined
	$(PYTHON) jimu-dse/scripts/closed_loop.py validate-config --goal cycle-latency-optimization
	$(PYTHON) jimu-dse/scripts/closed_loop.py validate-config --goal rtl-cycle-optimization
	$(PYTHON) jimu-dse/scripts/closed_loop.py validate-config --goal rtl-dram-optimization
	$(PYTHON) jimu-dse/scripts/closed_loop.py validate-config --goal rtl-dram-exploration
	$(PYTHON) jimu-dse/scripts/closed_loop.py validate-config --goal rtl-cycle-optimization-large

list-goals:
	$(PYTHON) jimu-dse/scripts/closed_loop.py list-goals

# -----------------------------------------------------------------------
# Clean — removes rebuildable artifacts, preserving closed-loop run results
# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR) libnpukernels.so _out .opencode
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf firmware/build_dim*/

clean-results:
	rm -rf jimu-dse/results

# -----------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  all       Build kernels + firmware"
	@echo "  kernels   Build C kernel library (libnpukernels.so)"
	@echo "  firmware  Build RISC-V firmware ELFs"
	@echo "  opencode  Configure OpenCode agent (skills + permissions)"
	@echo "  timing-deps  Install pinned SCALE-Sim timing backend"
	@echo "  rtl-lint     Lint the synthesizable RTL timing core with Verilator"
	@echo "  rtl-test     Run RTL trace-replay and overlap/contention tests"
	@echo "  test      Run all tests"
	@echo "  validate-goals  Validate every built-in optimization goal"
	@echo "  list-goals      List configurable optimization goals"
	@echo "  clean          Remove rebuildable artifacts; preserve run results"
	@echo "  clean-results  Remove all closed-loop run results"
