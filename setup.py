from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

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
