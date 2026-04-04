cmake_minimum_required(VERSION 3.18)
project(poetry_scalapack_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(POETRY_ENABLE_SCALAPACK "Enable discovery and linkage of a ScaLAPACK library if available" ON)
set(POETRY_SCALAPACK_LIBRARY "" CACHE STRING "Optional explicit path to a ScaLAPACK library")

find_package(MPI REQUIRED)

# Prefer OpenBLAS over FlexiBLAS to avoid dual-library conflicts
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

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
