# distutils: language=c++ 

cimport cython 

import numpy as np

cdef extern from "__polymer_math.h":
    long int _getLinkingNumberCpp(int M, double *olddata, int N)


def getLinkingNumber(data1, data2, randomOffset=True):
    if len(data1) == 3:
        data1 = np.array(data1.T)
    if len(data2) == 3:
        data2 = np.array(data2.T)
    if len(data1[0]) != 3:
        raise ValueError
    if len(data2[0]) != 3:
        raise ValueError
    data1 = np.asarray(data1, dtype=np.double, order="C")
    data2 = np.asarray(data2, dtype=np.double, order="C")
    if randomOffset:
        data1 += np.random.random(data1.shape) * 0.0000001
        data2 += np.random.random(data2.shape) * 0.0000001
    cdef double[:,:] olddata = np.concatenate([data1, data2], axis=0)
    olddata = np.array(olddata, dtype=float, order="C")
    M = len(data1)
    N = len(olddata)

    L = _getLinkingNumberCpp(M, &olddata[0,0], N)
    return L

