.PHONY: native-build native-clean test help

# Configurable build directory (can be overridden for per-job isolation)
# Usage: make native-build BUILD_DIR=native/build-$SLURM_JOB_ID
BUILD_DIR ?= native/build

# Configurable CMake options
ENABLE_SCALAPACK ?= ON
ENABLE_PYBIND11 ?= ON

# Build native C++ executables and PyBind11 module
native-build:
	@echo "Building native C++ code..."
	@echo "  Build directory: $(BUILD_DIR)"
	@echo "  ScaLAPACK: $(ENABLE_SCALAPACK)"
	@echo "  PyBind11: $(ENABLE_PYBIND11)"
	@echo ""
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
	cmake ../.. \
		-DPOETRY_ENABLE_SCALAPACK=$(ENABLE_SCALAPACK) \
		-DPOETRY_ENABLE_PYBIND11=$(ENABLE_PYBIND11) \
		-DCMAKE_BUILD_TYPE=Release && \
	make -j$$(nproc)
	@echo ""
	@echo "Native build complete!"
	@echo "  Build directory: $(BUILD_DIR)"
	@echo "  Executables: $(BUILD_DIR)/scalapack_gp_fit, $(BUILD_DIR)/scalapack_daemon"
	@if [ "$(ENABLE_PYBIND11)" = "ON" ]; then \
		echo "  PyBind11 module: $(BUILD_DIR)/poetry_gp_native*.so"; \
		echo ""; \
		if ls $(BUILD_DIR)/poetry_gp_native*.so 1> /dev/null 2>&1; then \
			echo "Installing PyBind11 module to Python path..."; \
			cp $(BUILD_DIR)/poetry_gp_native*.so .; \
			echo "✓ Module installed: poetry_gp_native.so"; \
		else \
			echo "⚠ PyBind11 module not found. Check CMake output above."; \
		fi; \
	fi

# Clean native build artifacts
native-clean:
	@echo "Cleaning native build artifacts..."
	@if [ "$(BUILD_DIR)" != "native/build" ]; then \
		echo "  Build directory: $(BUILD_DIR)"; \
		rm -rf $(BUILD_DIR); \
	else \
		rm -rf native/build; \
	fi
	rm -f poetry_gp_native*.so
	@echo "✓ Native build cleaned"

# Run all tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Show help
help:
	@echo "Poetry GP - Makefile targets:"
	@echo ""
	@echo "  native-build  - Build C++ executables and PyBind11 module"
	@echo "  native-clean  - Clean native build artifacts"
	@echo "  test          - Run Python tests"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Environment setup:"
	@echo "  ./scripts/bootstrap_env.sh        - Setup CPU-only environment"
	@echo "  ./scripts/bootstrap_env.sh --gpu  - Setup GPU-enabled environment"
	@echo ""
	@echo "Native backends:"
	@echo "  fit_backend='python'           - Scipy (baseline)"
	@echo "  fit_backend='native_lapack'    - PyBind11 LAPACK (m < 5000, zero overhead)"
	@echo "  fit_backend='native_reference' - ScaLAPACK MPI (m >= 5000, distributed)"
