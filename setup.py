import os
import platform
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# The native CPU engine benefits from OpenMP; it still builds without it (single-threaded).
# Extra flags (e.g. "-march=znver3") can be passed via POLYCHROM_NATIVE_CFLAGS.
# No FMA contraction, so that the baseline and the x86-64-v3 build give the same results.
native_compile_args = ["-O3", "-std=c++17", "-fno-math-errno", "-ffp-contract=off"]
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

# A second build of the engine for x86-64-v3 CPUs (AVX2, FMA, BMI2: Intel Haswell and AMD
# Zen or newer), selected at run time by polychrom.native.
if platform.machine().lower() in ("x86_64", "amd64") and sys.platform != "win32":
    ext_modules.append(
        Extension(
            "polychrom._native_engine_x86_64_v3",
            ["polychrom/_native_engine_x86_64_v3.pyx"],
            include_dirs=[numpy.get_include(), "polychrom"],
            depends=["polychrom/_native/engine.hpp", "polychrom/_native_engine.pyx"],
            language="c++",
            extra_compile_args=native_compile_args[:3] + ["-mavx2", "-mfma", "-mbmi2"] + native_compile_args[3:],
            extra_link_args=native_link_args,
        )
    )

setup(
    ext_modules=cythonize(ext_modules, language_level="3"),
)
