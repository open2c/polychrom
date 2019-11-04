# distutils: language=c++ 

cimport cython 

import numpy as np

cdef extern from "__polymer_math.h":
    long int _getLinkingNumberCpp(int M, double *olddata, int N)
    void _mutualSimplifyCpp ( 
        double *datax1, double *datay1, double *dataz1, int N1,
        double *datax2, double *datay2, double *dataz2, int N2,
        long *ret
    )
    int _simplifyCpp (double *datax, double *datay, double *dataz, int N) 




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


def mutualSimplify(data1, data2):
    """a weave.inline wrapper for polymer simplification code
    Calculates a simplified topologically equivalent polymer ring"""

    if len(data1) != 3:
        data1 = np.transpose(data1)
    if len(data1) != 3:
        raise ValueError("Wrong dimensions of data")
    if len(data2) != 3:
        data2 = np.transpose(data2)
    if len(data2) != 3:
        raise ValueError("Wrong dimensions of data")

    cdef double[:] datax1 = np.array(data1[0], float, order="C")
    cdef double[:] datay1 = np.array(data1[1], float, order="C")
    cdef double[:] dataz1 = np.array(data1[2], float, order="C")

    cdef double[:] datax2 = np.array(data2[0], float, order="C")
    cdef double[:] datay2 = np.array(data2[1], float, order="C")
    cdef double[:] dataz2 = np.array(data2[2], float, order="C")

    cdef int N1 = len(datax1)
    cdef int N2 = len(datax2)

    cdef long[:] ret = np.array([1, 1])

    _mutualSimplifyCpp(
        &datax1[0], &datay1[0], &dataz1[0], N1,
        &datax2[0], &datay2[0], &dataz2[0], N2,
        &ret[0])

    data1 = np.array([datax1, datay1, dataz1]).T
    data2 = np.array([datax2, datay2, dataz2]).T

    return data1[:ret[0]], data2[:ret[1]]


def simplifyPolymer(data):
    """a weave.inline wrapper for polymer simplification code
    Calculates a simplified topologically equivalent polymer ring"""

    if len(data) != 3:
        data = np.transpose(data)
    if len(data) != 3:
        raise ValueError("Wrong dimensions of data")
    cdef double[:] datax = np.array(data[0], float, order="C")
    cdef double[:] datay = np.array(data[1], float, order="C")
    cdef double[:] dataz = np.array(data[2], float, order="C")
    
    cdef int N = len(datax)

    new_N = _simplifyCpp(&datax[0], &datay[0], &dataz[0], N)

    data = np.array([datax, datay, dataz]).T

    return data[:new_N]