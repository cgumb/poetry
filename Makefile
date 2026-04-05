.PHONY: native-build native-clean test help

# Build native C++ executables and PyBind11 module
native-build:
	@echo "Building native C++ code..."
	mkdir -p native/build
	cd native/build && \
	cmake .. \
		-DPOETRY_ENABLE_SCALAPACK=ON \
		-DPOETRY_ENABLE_PYBIND11=ON \
		-DCMAKE_BUILD_TYPE=Release && \
	make -j$$(nproc)
	@echo ""
	@echo "Native build complete!"
	@echo "  Executables: native/build/scalapack_gp_fit, native/build/scalapack_daemon"
	@echo "  PyBind11 module: native/build/poetry_gp_native*.so"
	@echo ""
	@if [ -f native/build/poetry_gp_native*.so ]; then \
		echo "Installing PyBind11 module to Python path..."; \
		cp native/build/poetry_gp_native*.so .; \
		echo "✓ Module installed: poetry_gp_native.so"; \
	else \
		echo "⚠ PyBind11 module not found. Check CMake output above."; \
	fi

# Clean native build artifacts
native-clean:
	@echo "Cleaning native build artifacts..."
	rm -rf native/build
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
