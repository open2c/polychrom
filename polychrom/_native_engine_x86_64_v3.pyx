# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
"""The native engine compiled for x86-64-v3 (AVX2, FMA, BMI2).

polychrom.native imports it instead of polychrom._native_engine when the CPU
supports these instructions.
"""

include "_native_engine.pyx"
