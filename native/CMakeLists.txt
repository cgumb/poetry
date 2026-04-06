cmake_minimum_required(VERSION 3.18)
project(poetry_scalapack_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(POETRY_ENABLE_SCALAPACK "Enable discovery and linkage of a ScaLAPACK library if available" ON)
option(POETRY_ENABLE_PYBIND11 "Enable PyBind11 for in-memory Python bridge" ON)
set(POETRY_SCALAPACK_LIBRARY "" CACHE STRING "Optional explicit path to a ScaLAPACK library")

find_package(MPI REQUIRED)

# Configurable BLAS vendor preference with automatic fallback
set(POETRY_BLAS_VENDOR "OpenBLAS" CACHE STRING "Preferred BLAS vendor (OpenBLAS, Generic, FlexiBLAS, etc.)")

# Try preferred BLAS vendor first
if(POETRY_BLAS_VENDOR AND NOT POETRY_BLAS_VENDOR STREQUAL "Generic")
  set(BLA_VENDOR ${POETRY_BLAS_VENDOR})
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    message(STATUS "BLAS found using vendor: ${POETRY_BLAS_VENDOR}")
  else()
    message(STATUS "BLAS vendor ${POETRY_BLAS_VENDOR} not found, trying generic BLAS...")
    unset(BLA_VENDOR)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
      message(STATUS "BLAS found (generic)")
    endif()
  endif()
else()
  # User explicitly requested Generic or unset
  unset(BLA_VENDOR)
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    message(STATUS "BLAS found (generic)")
  endif()
endif()

# Final check - only fail if no BLAS found at all
if(NOT BLAS_FOUND)
  message(FATAL_ERROR "Could not find BLAS library. Try setting -DPOETRY_BLAS_VENDOR=Generic or install a BLAS library (e.g., openblas-devel, blas-devel).")
endif()

set(POETRY_SCALAPACK_FOUND FALSE)
set(POETRY_SCALAPACK_LIBRARIES "")

if(POETRY_ENABLE_SCALAPACK)
  if(POETRY_SCALAPACK_LIBRARY)
    if(EXISTS "${POETRY_SCALAPACK_LIBRARY}")
      set(POETRY_SCALAPACK_FOUND TRUE)
      set(POETRY_SCALAPACK_LIBRARIES "${POETRY_SCALAPACK_LIBRARY}")
    else()
      message(FATAL_ERROR "POETRY_SCALAPACK_LIBRARY does not exist: ${POETRY_SCALAPACK_LIBRARY}")
    endif()
  else()
    find_library(
      POETRY_SCALAPACK_AUTO_LIBRARY
      NAMES scalapack scalapack-openmpi scalapack-mpich scalapack-mpi scalapack64 scalapack64-openmpi
    )
    if(POETRY_SCALAPACK_AUTO_LIBRARY)
      set(POETRY_SCALAPACK_FOUND TRUE)
      set(POETRY_SCALAPACK_LIBRARIES "${POETRY_SCALAPACK_AUTO_LIBRARY}")
    endif()
  endif()
endif()

# Build main executable
add_executable(scalapack_gp_fit scalapack_gp_fit_entry.cpp)
target_link_libraries(scalapack_gp_fit PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES})

# Link BLAS for kernel assembly (dgemm)
# ScaLAPACK will bring its own BLAS, but we need it even without ScaLAPACK
if(POETRY_SCALAPACK_FOUND)
  target_link_libraries(scalapack_gp_fit PRIVATE ${POETRY_SCALAPACK_LIBRARIES})
  target_compile_definitions(scalapack_gp_fit PRIVATE HAVE_SCALAPACK=1)
  message(STATUS "ScaLAPACK support enabled: ${POETRY_SCALAPACK_LIBRARIES}")
else()
  message(STATUS "ScaLAPACK support not found; building MPI reference backend only")
endif()

# Build persistent daemon (Milestone 1C)
# Note: daemon needs access to run_scalapack_distributed but not main()
# Solution: Build scalapack_gp_fit.cpp with DAEMON_BUILD flag to exclude main
add_executable(scalapack_daemon scalapack_daemon.cpp scalapack_gp_fit.cpp)
target_compile_definitions(scalapack_daemon PRIVATE DAEMON_BUILD=1)
target_link_libraries(scalapack_daemon PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES})
if(POETRY_SCALAPACK_FOUND)
  target_link_libraries(scalapack_daemon PRIVATE ${POETRY_SCALAPACK_LIBRARIES})
  target_compile_definitions(scalapack_daemon PRIVATE HAVE_SCALAPACK=1)
endif()

# PyBind11 module for in-memory Python bridge (Phase 1: LAPACK only)
if(POETRY_ENABLE_PYBIND11)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(poetry_gp_native pybind11_lapack.cpp)
  target_link_libraries(poetry_gp_native PRIVATE ${BLAS_LIBRARIES})

  # Enable OpenMP for multi-threaded BLAS (if available)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(poetry_gp_native PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "PyBind11 module built with OpenMP support")
  else()
    message(STATUS "PyBind11 module built without OpenMP (single-threaded BLAS)")
  endif()

  message(STATUS "PyBind11 module enabled: poetry_gp_native")
endif()
