#!/bin/bash
#
# Robust CUDA rebuild script for FeNNol
# This script does a complete clean rebuild of the CUDA extension
#

set -e  # Exit on error

echo "=========================================="
echo "FeNNol CUDA Complete Rebuild Script"
echo "=========================================="
echo ""

# Store the project root
PROJECT_ROOT="/home/aaron/ATX/software/ATF-FeNNol/v4/FeNNol"
CUDA_DIR="$PROJECT_ROOT/src/fennol/cuda"

# Environment variables
export FENNOL_BUILD_CUDA=1
export LIBRARY_PATH=/lib/x86_64-linux-gnu:$LIBRARY_PATH
export CMAKE_PREFIX_PATH=/home/aaron/miniforge3/envs/fennol-claude-dev-cuda/lib/python3.13/site-packages/pybind11/share/cmake/pybind11
export CMAKE_MAKE_PROGRAM=/usr/bin/make
export CUDACXX=/usr/local/cuda-12/bin/nvcc
export PATH=/usr/local/cuda-12/bin:/usr/bin:$PATH
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FENNOL=0.1.dev
export PYTHON=/home/aaron/miniforge3/envs/fennol-claude-dev-cuda/bin/python

echo "Step 1: Uninstall existing FeNNol package"
$PYTHON -m pip uninstall -y fennol 2>/dev/null || echo "  (no existing installation)"
echo ""

echo "Step 2: Clean all build artifacts"
cd "$PROJECT_ROOT"
rm -rf build dist *.egg-info src/fennol.egg-info
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  - Cleaned Python caches"

cd "$CUDA_DIR"
rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile *.so build
echo "  - Cleaned CUDA build files"
echo ""

echo "Step 3: Build CUDA extension with CMake"
cd "$CUDA_DIR"
/usr/bin/cmake .
echo ""

echo "Step 4: Compile CUDA kernels"
/usr/bin/make
echo ""

echo "Step 5: Copy .so file to site-packages"
cp "$CUDA_DIR/fennol_cuda.cpython-313-x86_64-linux-gnu.so" \
   /home/aaron/miniforge3/envs/fennol-claude-dev-cuda/lib/python3.13/site-packages/ 2>/dev/null || true
echo "  - Copied CUDA extension"
echo ""

echo "Step 6: Install FeNNol package (editable mode, skip CUDA build)"
cd "$PROJECT_ROOT"
FENNOL_BUILD_CUDA=0 $PYTHON -m pip install -e . --no-deps
echo ""

echo "=========================================="
echo "✓ Rebuild complete!"
echo "=========================================="
echo ""
echo "To test, run:"
echo "  python test_3atom_forces.py"
echo "  python test_md_tiny_dt.py"
