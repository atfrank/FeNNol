cmake_minimum_required(VERSION 3.18)

# Use GCC 11 for all compilation to avoid LTO version mismatch
set(CMAKE_C_COMPILER gcc-11)
set(CMAKE_CXX_COMPILER g++-11)
set(CMAKE_CUDA_HOST_COMPILER g++-11)

project(fennol_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Disable LTO to avoid version mismatch between CUDA host compiler and system compiler
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-lto")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

# Source files
set(CUDA_SOURCES
    src/integrate.cu
    src/restraints.cu
    src/physics.cu
    src/thermostats.cu
    src/multi_gpu.cu
    src/gb_born_radii.cu
    src/gb_energy_forces.cu
    src/gb_nonpolar.cu
    src/gnn_neighborlist.cu
    src/gnn_message_passing.cu
)

set(CPP_SOURCES
    src/bindings.cpp
)

# Create shared library
pybind11_add_module(fennol_cuda
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Include directories
target_include_directories(fennol_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries
target_link_libraries(fennol_cuda PRIVATE
    CUDA::cudart
)

# Add library search path for pthread on newer Linux systems
if(UNIX AND NOT APPLE)
    # Use -pthread flag instead of explicit library linking
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-pthread")
endif()

# Set CUDA compilation flags
set_target_properties(fennol_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION OFF
)

# Compiler flags
target_compile_options(fennol_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler=-fPIC
        -Xcompiler=-fno-lto
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -fno-lto
    >
)

# Install target
install(TARGETS fennol_cuda LIBRARY DESTINATION fennol/cuda)
