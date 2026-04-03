cmake_minimum_required(VERSION 3.18)
project(poetry_scalapack_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(POETRY_ENABLE_SCALAPACK "Enable discovery and linkage of a ScaLAPACK library if available" ON)
set(POETRY_SCALAPACK_LIBRARY "" CACHE STRING "Optional explicit path to a ScaLAPACK library")

find_package(MPI REQUIRED)

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

add_executable(scalapack_gp_fit scalapack_gp_fit_entry.cpp)
target_link_libraries(scalapack_gp_fit PRIVATE MPI::MPI_CXX)

# Intentionally do not add a second generic BLAS/LAPACK stack here.
# The discovered/shared ScaLAPACK library should bring along its own BLAS
# dependency chain from the same managed environment.
if(POETRY_SCALAPACK_FOUND)
  target_link_libraries(scalapack_gp_fit PRIVATE ${POETRY_SCALAPACK_LIBRARIES})
  target_compile_definitions(scalapack_gp_fit PRIVATE HAVE_SCALAPACK=1)
  message(STATUS "ScaLAPACK support enabled: ${POETRY_SCALAPACK_LIBRARIES}")
else()
  message(STATUS "ScaLAPACK support not found; building MPI reference backend only")
endif()
