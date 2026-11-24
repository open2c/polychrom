import os
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# The native CPU engine benefits from OpenMP; it still builds without it (single-threaded).
# Extra flags (e.g. "-march=znver3") can be passed via POLYCHROM_NATIVE_CFLAGS.
native_compile_args = ["-O3", "-std=c++17", "-fno-math-errno"]
native_link_args = []
if sys.platform != "darwin" and os.environ.get("POLYCHROM_NO_OPENMP") is None:
    native_compile_args.append("-fopenmp")
    native_link_args.append("-fopenmp")
native_compile_args += os.environ.get("POLYCHROM_NATIVE_CFLAGS", "").split()

# Define Cython extensions
ext_modules = [
    Extension(
        "polychrom._polymer_math",
        ["polychrom/_polymer_math.pyx", "polychrom/__polymer_math.cpp"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "polychrom._native_engine",
        ["polychrom/_native_engine.pyx"],
        include_dirs=[numpy.get_include(), "polychrom"],
        depends=["polychrom/_native/engine.hpp"],
        language="c++",
        extra_compile_args=native_compile_args,
        extra_link_args=native_link_args,
    ),
]

setup(
    ext_modules=cythonize(ext_modules, language_level="3"),
)
