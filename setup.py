import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# Define Cython extensions
ext_modules = [
    Extension(
        "polychrom._polymer_math",
        ["polychrom/_polymer_math.pyx", "polychrom/__polymer_math.cpp"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    ext_modules=cythonize(ext_modules, language_level="3"),
)
