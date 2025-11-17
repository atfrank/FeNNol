# Installation Troubleshooting Guide

This document describes common installation issues and their solutions, particularly for CUDA-enabled installations on modern Linux systems with Python 3.13+.

## Python 3.13+ Installation Issues

### Issue: UnboundLocalError during CUDA build

**Symptom:**
```
UnboundLocalError: cannot access local variable 'subprocess' where it is not associated with a value
```

**Cause:** Python 3.13 introduced stricter scoping rules that prevent shadowing of module-level imports with local imports inside function scopes.

**Fix:** Removed duplicate `import subprocess` statement at line 72 in `setup.py`. The module is already imported at the top of the file (line 7).

**Commit:** Fixed in setup.py by removing duplicate subprocess import

---

## CUDA Build Issues

### Issue 1: Missing C++ Standard Library Headers

**Symptom:**
```
error: namespace "std" has no member "runtime_error"
error: namespace "std" has no member "string"
```

**Cause:** CUDA header files using `std::runtime_error` and `std::string` without proper includes.

**Fix:** Added required includes to `src/fennol/cuda/include/common.cuh`:
```cpp
#include <stdexcept>
#include <string>
```

---

### Issue 2: Device Code in Host Compilation

**Symptom:**
```
error: '__longlong_as_double' was not declared in this scope
error: 'atomicCAS' was not declared in this scope
error: 'threadIdx' was not declared in this scope
```

**Cause:** Device-only functions being compiled by the C++ host compiler when included in `bindings.cpp`.

**Fix:** Added `#ifdef __CUDACC__` guards around device-only code in `common.cuh`:
- `atomicAddDouble()` function
- `warpReduceSum()` template
- `blockReduceSum()` template

---

### Issue 3: Missing pthread and libc Libraries

**Symptom:**
```
/usr/bin/ld: cannot find /lib64/libpthread.so.0: No such file or directory
/usr/bin/ld: cannot find /usr/lib64/libc.so.6: No such file or directory
```

**Cause:** On modern Ubuntu/Debian systems, libraries are in `/lib/x86_64-linux-gnu/` but the linker looks for them in `/lib64/` and `/usr/lib64/`.

**Fix:** Create symlinks (requires sudo):
```bash
sudo mkdir -p /lib64 /usr/lib64
sudo ln -sf /lib/x86_64-linux-gnu/libpthread.so.0 /lib64/libpthread.so.0
sudo ln -sf /lib/x86_64-linux-gnu/libc.so.6 /lib64/libc.so.6
sudo ln -sf /lib/x86_64-linux-gnu/libc_nonshared.a /usr/lib64/libc_nonshared.a
```

---

### Issue 4: LTO Version Mismatch

**Symptom:**
```
lto1: fatal error: bytecode stream in file 'cmake_device_link.o' generated with LTO version 13.1 instead of the expected 11.3
```

**Cause:** CUDA compiler using a different GCC version than the system's default C++ compiler, causing incompatible LTO bytecode versions.

**Fix:** Modified `CMakeLists.txt` to use consistent compiler versions:
```cmake
# Use GCC 11 for all compilation to avoid LTO version mismatch
set(CMAKE_C_COMPILER gcc-11)
set(CMAKE_CXX_COMPILER g++-11)
set(CMAKE_CUDA_HOST_COMPILER g++-11)

# Disable LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-lto")
```

---

### Issue 5: CMake Cannot Find pybind11

**Symptom:**
```
CMake Error: Could not find a package configuration file provided by "pybind11"
```

**Cause:** CMake doesn't know where to find pybind11's CMake configuration files when installed via pip.

**Fix:** Set `CMAKE_PREFIX_PATH` environment variable before installation:
```bash
CMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir)
```

---

## Complete Installation Instructions for CUDA-Enabled Build

### Prerequisites

1. Install pybind11 and build dependencies:
```bash
pip install pybind11 setuptools setuptools-scm
```

2. Create necessary symlinks (requires sudo, one-time setup):
```bash
sudo mkdir -p /lib64 /usr/lib64
sudo ln -sf /lib/x86_64-linux-gnu/libpthread.so.0 /lib64/libpthread.so.0
sudo ln -sf /lib/x86_64-linux-gnu/libc.so.6 /lib64/libc.so.6
sudo ln -sf /lib/x86_64-linux-gnu/libc_nonshared.a /usr/lib64/libc_nonshared.a
```

### Installation Command

From the FeNNol repository root:

```bash
CMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir) \
FENNOL_BUILD_CUDA=1 \
pip install -e . --no-build-isolation
```

### Verification

Verify the installation:
```bash
python -c "import fennol; import fennol.cuda.fennol_cuda; print('CUDA extension loaded successfully')"
```

---

## System Requirements

- Python 3.8+
- CUDA Toolkit 12.x
- GCC 11 or compatible version
- CMake 3.18+
- pybind11

---

## Platform-Specific Notes

### Ubuntu 22.04 / Debian

The fixes documented above were tested on Ubuntu 22.04. The library path issues and LTO mismatch are common on this platform.

### Other Linux Distributions

For other distributions, library paths may differ. Adjust symlink paths accordingly based on your system's library locations.

---

## Troubleshooting Tips

1. **Check GCC version compatibility:**
   ```bash
   gcc --version
   nvcc --version
   ```

2. **Verify CUDA installation:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

3. **Check pybind11 location:**
   ```bash
   python -m pybind11 --cmakedir
   ```

4. **Clean build if needed:**
   ```bash
   pip uninstall fennol
   rm -rf build/ dist/ *.egg-info
   ```

---

## Contributing

If you encounter additional installation issues not covered here, please open an issue on the GitHub repository with:
- Your operating system and version
- Python version
- CUDA version
- Complete error messages
- Steps to reproduce
