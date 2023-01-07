import io
import os
import re
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages

cmdclass = {}


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read("polychrom", "__init__.py"),
        re.MULTILINE,
    ).group(1)
    return version


def get_long_description():
    return _read("README.md")


def get_requirements(path):
    content = _read(path)
    return [req for req in content.split("\n") if req != "" and not req.startswith("#")]


cmdclass.update({"build_ext": build_ext})

ext_modules = cythonize(
    [
        Extension(
            "polychrom._polymer_math",
            ["polychrom/_polymer_math.pyx", "polychrom/__polymer_math.cpp"],
        )
    ]
)


setup(
    name="polychrom",
    author="Mirny Lab",
    author_email="espresso@mit.edu",
    version=get_version(),
    url="http://github.com/mirnylab/polychrom",
    description=("A library for polymer simulations and their analyses."),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords=["genomics", "polymer", "Hi-C", "molecular dynamics", "chromosomes"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "polychrom = polychrom.cli:cli",
        ]
    },
)
