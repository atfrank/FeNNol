cmake_minimum_required(VERSION 3.18)
project(fennol_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

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

# Set CUDA compilation flags
set_target_properties(fennol_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler flags
target_compile_options(fennol_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler=-fPIC
    >
)

# Install target
install(TARGETS fennol_cuda LIBRARY DESTINATION fennol/cuda)
