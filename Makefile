# NPU — Firmware Optimization
#
# Top-level Makefile for common development tasks.

.PHONY: all kernels firmware opencode timing-deps test validate-goals list-goals clean clean-results

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

opencode: .opencode/skills/dag-analyze/SKILL.md .opencode/skills/vrf-cache/SKILL.md .opencode/skills/dim-optimize/SKILL.md .opencode/skills/weighted-latency/SKILL.md .opencode/skills/cycle-latency/SKILL.md .opencode/skills/dataflow-optimize/SKILL.md .opencode/skills/self-verify/SKILL.md
	@echo "✅ OpenCode agent configured (skills installed, permissions from global config)"

timing-deps:
	python3 -m pip install -r requirements-timing.txt

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
test:
	python3 -m pytest tests/ -v

validate-goals:
	python3 jimu-dse/scripts/closed_loop.py validate-config --goal dram-optimization
	python3 jimu-dse/scripts/closed_loop.py validate-config --goal compute-optimization
	python3 jimu-dse/scripts/closed_loop.py validate-config --goal combined
	python3 jimu-dse/scripts/closed_loop.py validate-config --goal weighted-latency-optimization
	python3 jimu-dse/scripts/closed_loop.py validate-config --goal cycle-latency-optimization

list-goals:
	python3 jimu-dse/scripts/closed_loop.py list-goals

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
	@echo "  test      Run all tests"
	@echo "  validate-goals  Validate every built-in optimization goal"
	@echo "  list-goals      List configurable optimization goals"
	@echo "  clean          Remove rebuildable artifacts; preserve run results"
	@echo "  clean-results  Remove all closed-loop run results"
