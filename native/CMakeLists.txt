cmake_minimum_required(VERSION 3.18)
project(poetry_scalapack_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(scalapack_gp_fit scalapack_gp_fit.cpp)
target_link_libraries(scalapack_gp_fit PRIVATE MPI::MPI_CXX)

# This scaffold currently builds only the file-interface executable.
# A future follow-up should add BLACS/ScaLAPACK discovery and linkage here.
