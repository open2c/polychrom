import os 

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy 
cmdclass = {}


cmdclass.update({'build_ext': build_ext} )

ext_modules = cythonize([Extension(
    'polychrom._polymer_math',
    ["polychrom/_polymer_math.pyx", 'polychrom/__polymer_math.cpp'],
    )])
                
setup(
    name='polychrom',
    url='http://github.com/mirnylab/polychrom',
    description=('A library for polymer simulations.'),
    ext_modules=ext_modules,
    cmdclass = cmdclass,
    packages=find_packages(),

)
