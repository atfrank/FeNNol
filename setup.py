"""
Setup script for FeNNol with CUDA extension support.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension that uses CMake for building."""

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command that uses CMake."""

    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the CUDA extension. "
                "Install with: pip install cmake"
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]

        build_args = ['--config', 'Release']

        # Check if CUDA is available
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                print("CUDA compiler found. Building CUDA extension...")
            else:
                print("CUDA compiler not found. Skipping CUDA extension.")
                return
        except FileNotFoundError:
            print("CUDA compiler not found. Skipping CUDA extension.")
            return

        # Detect CUDA architecture
        if 'CMAKE_CUDA_ARCHITECTURES' not in os.environ:
            # Try to detect GPU architecture
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    # Get first GPU architecture
                    arch = result.stdout.strip().split('\n')[0].replace('.', '')
                    cmake_args.append(f'-DCMAKE_CUDA_ARCHITECTURES={arch}')
                    print(f"Detected CUDA architecture: {arch}")
            except Exception as e:
                print(f"Could not detect CUDA architecture: {e}")
                print("Using default architectures: 60;70;75;80;86")

        # Configure
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )

        # Build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )


def should_build_cuda_extension():
    """
    Determine if we should build the CUDA extension.

    Returns True if:
    - FENNOL_BUILD_CUDA environment variable is set to 1
    - OR nvcc is available and user didn't explicitly disable it
    """
    # Check environment variable
    build_cuda = os.environ.get('FENNOL_BUILD_CUDA', None)
    if build_cuda == '1':
        return True
    elif build_cuda == '0':
        return False

    # Auto-detect CUDA availability
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# Configure extensions
ext_modules = []
cmdclass = {}

if should_build_cuda_extension():
    print("=" * 60)
    print("CUDA Extension Build")
    print("=" * 60)
    print("Building CUDA-accelerated kernels for FeNNol")
    print("This may take several minutes...")
    print("=" * 60)

    ext_modules.append(
        CMakeExtension('fennol.cuda.fennol_cuda', sourcedir='src/fennol/cuda')
    )
    cmdclass['build_ext'] = CMakeBuild
else:
    print("=" * 60)
    print("Skipping CUDA Extension")
    print("=" * 60)
    print("CUDA extension will not be built.")
    print("To enable CUDA acceleration:")
    print("  1. Install CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)")
    print("  2. Set FENNOL_BUILD_CUDA=1 environment variable")
    print("  3. Run: pip install -e .")
    print("=" * 60)


# Use setuptools setup with optional CUDA extension
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
