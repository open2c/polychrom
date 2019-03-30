from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy 
cmdclass = {}


cmdclass.update({'build_ext': build_ext} )

setup(
    name='openmmlib',
    url='http://mirnylab.bitbucket.org/hiclib/index.html',
    description=('Hi-C data analysis library.'),
      ext_modules=[],
      cmdclass = cmdclass,
       packages=['openmmlib'],

)
