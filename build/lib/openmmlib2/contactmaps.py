# Code written by: Maksim Imakaev (imakaev@mit.edu)
"""
This file contains a bunch of method to work on contact maps of a Hi-C data.
It uses a lot of methods from mirnylib repository.





Find contacts of a conformation
-------------------------------

To be filled in later


Find average contact maps
-------------------------

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import traceback

import numpy as np
from mirnylib.h5dict import h5dict
from mirnylib.numutils import sumByArray

from math import sqrt
from mirnylib.systemutils import fmapred, fmap, deprecate, setExceptionHook
import sys
import mirnylib.numutils
from .polymerutils import load
import warnings
from . import polymerutils
import time

#import matplotlib
#import matplotlib.pyplot as plt

from scipy.spatial import ckdtree


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass 


class TimeoutException(Exception): pass


a = np.random.random((100, 3)) * 3

#matplotlib.rcParams.update({'font.size': 8})


def giveContactsCKDTree(X, cutoff=1.7):
    tree = ckdtree.cKDTree(X)

    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    return pairs

def condensed_to_pair_indices(n, k):
    x = n - (4. * n ** 2 - 4 * n - 8 * k + 1) ** .5 / 2 - .5
    i = x.astype(int)
    j = k + i * (i + 3 - 2 * n) / 2 + 1
    return np.array([i, j]).T



def rad2(data):
    """
    Returns Rg(N^(2/3)
    This is an important point, as for a dense equilibrium globule at N^{2/3}
    both P(s) and Rg(s) plots reach a plateau.
    """

    if len(data) != 3:
        data = np.transpose(data)
    if len(data) != 3:
        raise ValueError("Wrong dimensions of data")

    def give_radius_scaling(data):
        N = len(data[0])
        target = int(N ** (2 / 3.))
        coms = np.cumsum(data, 1)
        coms2 = np.cumsum(data ** 2, 1)

        def radius_gyration(len2):
            coms2d = (-coms2[:, :-len2] + coms2[:, len2:]) / len2
            comsd = ((coms[:, :-len2] - coms[:, len2:]) / len2) ** 2
            diffs = coms2d - comsd
            sums = np.sqrt(np.sum(diffs, 0))
            return np.mean(sums)

        return radius_gyration(target)

    return give_radius_scaling(data)


def giveIntContacts(data):
    """give all contacts of a polymer on a cubic lattice
    Intersections are not counted as contacts. Sorry :(

    Parameters
    ----------
    data : Nx3 or 3xN array of ints
    """

    data = np.asarray(data, dtype=int)

    if len(data.shape) != 2:
        raise ValueError("Wrong dimensions of data")
    if 3 not in data.shape:
        raise ValueError("Wrong size of data: %s,%s" % data.shape)
    if data.shape[0] == 3:
        data = data.T

    data -= np.min(data, axis=0)[None, :]

    M = np.max(data) + 1
    if M > 1500:
        raise ValueError("Polymer is to big, can't create bounding box!")

    N = len(data)
    tocheck = np.zeros(M * M * M, dtype=np.int32) - 1
    tocheck[data[:, 0] + data[:, 1] * M + data[:, 2] * M *
            M] = np.arange(N, dtype=np.int32)
    tocheck.shape = (M, M, M)
    contacts1 = np.concatenate([tocheck[1:, :, :].ravel(), tocheck[
                                                           :, 1:, :].ravel(), tocheck[:, :, 1:].ravel()])
    contacts2 = np.concatenate([tocheck[:-1, :, :].ravel(), tocheck[
                                                            :, :-1, :].ravel(), tocheck[:, :, :-1].ravel()])
    mask = (contacts1 != -1) * (contacts2 != -1)
    contacts1 = contacts1[mask]
    contacts2 = contacts2[mask]
    contacts3 = np.minimum(contacts1, contacts2)
    contacts4 = np.maximum(contacts1, contacts2)
    return np.concatenate([contacts3[:, None], contacts4[:, None]], 1)


def giveContacts(data, cutoff=1.7, method="auto"):
    """Returns contacts of a single polymer with a given cutoff

    .. warning:: Use this only to find contacts of a single polymer chain
    with distance between monomers of 1.
    Multiple chains will lead to silent bugs.

    Parameters
    ----------
    data : Nx3 or 3xN array
        Polymer configuration. One chaon only.
    cutoff : float , optional
        Cutoff distance that defines contact

    Returns
    -------

    k by 2 array of contacts. Each row corresponds to a contact.
    """
    if data.shape[1] != 3:
        data = data.T
    data = np.ascontiguousarray(data, dtype=np.double)
    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    if max(data.shape) < 1000:
        return contactsCython(data, cutoff)
    else:
        return giveContactsCKDTree(data, cutoff)


methods = {"cython": contactsCython, "ckdtree": giveContactsCKDTree}



def rescalePoints(points, bins):
    "converts array of contacts to the reduced resolution contact map"
    a = np.histogram2d(points[:, 0], points[:, 1], bins)[0]
    a = a + np.transpose(a)

    return a


def rescaledMap(data, bins, cutoff=1.7, contactMap=None):
    # print data.sum(), bins.sum(), cutoff
    """calculates a rescaled contact map of a structure
    Parameters
    ----------
    data : Nx3 or 3xN array
        polymer conformation

    bins : Lx1 array
        bin starts

    cutoff : float, optional
        cutoff for contacts

    Returns
    -------
        resXres array with the contact map
    """

    t = giveContacts(data, cutoff)
    x = np.searchsorted(bins, t[:, 0]) - 1
    y = np.searchsorted(bins, t[:, 1]) - 1
    assert x.min() >= 0
    assert y.min() >= 0
    assert x.max() < len(bins) - 1
    assert y.max() < len(bins) - 1
    matrixSize = len(bins) - 1
    index = matrixSize * x + y
    
    unique, inds = np.unique(index, return_counts=True)    
    uniquex = unique // matrixSize
    uniquey = unique % matrixSize

    if contactMap is None:
        contactMap = np.zeros((matrixSize, matrixSize), dtype=int)
    contactMap[uniquex, uniquey] += inds

    return contactMap


def pureMap(data, cutoff=1.7, contactMap=None):
    """calculates an all-by-all contact map of a single polymer chain.
    Doesn't work for multi-chain polymers!!!
    If contact map is supplied, it just updates it

    Parameters
    ----------
    data : Nx3 or 3xN array
        polymer conformation
    cutoff : float
        cutoff for contacts
    contactMap : NxN array, optional
        contact map to update, if averaging is used
    """
    data = np.asarray(data)
    if len(data.shape) != 2:
        raise ValueError("Wrong dimensions of data")
    if 3 not in data.shape:
        raise ValueError("Wrong size of data: %s,%s" % data.shape)
    if data.shape[0] == 3:
        data = data.T
    data = np.asarray(data, float, order="C")

    t = giveContacts(data, cutoff)
    N = data.shape[0]
    if contactMap is None:
        contactMap = np.zeros((N, N), "int32")
    contactMap[t[:, 0], t[:, 1]] += 1
    contactMap[t[:, 1], t[:, 0]] += 1
    return contactMap


observedOverExpected = deprecate(mirnylib.numutils.observedOverExpected)




def averageBinnedContactMap(filenames, chains=None, binSize=None, cutoff=1.7,
                            n=4,  # Num threads
                            loadFunction=load,
                            exceptionsToIgnore=None, printProbability=1):
    """
    Returns an average contact map of a set of conformations.
    Non-existing files are ignored if exceptionsToIgnore is set to IOError.
    example:\n

    An example:

    .. code-block:: python
        >>> filenames = ["myfolder/blockd%d.dat" % i for i in xrange(1000)]
        >>> cmap = averageBinnedContactMap(filenames) + 1  #getting cmap
        #either showing a log of a map (+1 for zeros)
        >>> plt.imshow(numpy.log(cmap +1))
        #or truncating a map
        >>> vmax = np.percentile(cmap, 99.9)
        >>> plt.imshow(cmap, vmax=vmax)
        >>> plt.show()

    Parameters
    ----------
    filenames : list of strings
        Filenames to average map over
    chains : list of tuples or Nx2 array
        (start,end+1) of each chain
    binSize : int
        size of each bin in monomers
    cutoff : float, optional
        Cutoff to calculate contacts
    n : int, optional
        Number of threads to use.
        By default 4 to minimize RAM consumption.
    exceptionsToIgnore : list of Exceptions
        List of exceptions to ignore when finding the contact map.
        Put IOError there if you want it to ignore missing files.

    Returns
    -------
    tuple of two values:
    (i) MxM numpy array with the conntact map binned to binSize resolution.
    (ii) chromosomeStarts a list of start sites for binned map.

    """
    n = min(n, len(filenames))
    subvalues = [filenames[i::n] for i in range(n)]

    getResolution = 0
    fileInd = 0
    while getResolution == 0:
        try:
            data = loadFunction(filenames[fileInd])  # load filename
            getResolution = 1
        except:
            fileInd = fileInd + 1
        if fileInd >= len(filenames):
            print("no valid files found in filenames")
            raise ValueError("no valid files found in filenames")

    if chains is None:
        chains = [[0, len(data)]]
    if binSize is None:
        binSize = int(np.floor(len(data) / 500))

    bins = []
    chains = np.asarray(chains)
    chainBinNums = (
        np.ceil((chains[:, 1] - chains[:, 0]) / (0.0 + binSize)))
    for i in range(len(chainBinNums)):
        bins.append(binSize * (np.arange(int(chainBinNums[i])))
                    + chains[i, 0])
    bins.append(np.array([chains[-1, 1] + 1]))
    bins = np.concatenate(bins)
    bins = bins - .5
    Nbase = len(bins) - 1

    if Nbase > 10000:
        warnings.warn(UserWarning('very large contact map'
                                  ' may be difficult to visualize'))

    chromosomeStarts = np.cumsum(chainBinNums)
    chromosomeStarts = np.hstack((0, chromosomeStarts))

    def myaction(values):  # our worker receives some filenames
        mysum = None  # future contact map.
        for i in values:
            try:
                data = loadFunction(i)
                if np.random.random() < printProbability:
                    print(i)
            except tuple(exceptionsToIgnore):
                print("file not found", i)
                continue

            if data.shape[0] == 3:
                data = data.T
            if mysum is None:  # if it's the first filename,


                mysum = rescaledMap(data, bins, cutoff)  # create a map

            else:  # if not
                rescaledMap(data, bins, cutoff, mysum)
                # use existing map and fill in contacts

        return mysum

    blocks = fmap(myaction, subvalues)
    blocks = [i for i in blocks if i is not None]
    a = blocks[0]
    for i in blocks[1:]:
        a = a + i
    a = a + a.T

    return a, chromosomeStarts


def averagePureContactMap(filenames,
                          cutoff=1.7,
                          n=4,  # Num threads
                          loadFunction=load,
                          exceptionsToIgnore=[],
                          printProbability=0.005):
    """
        Parameters
    ----------
    filenames : list of strings
        Filenames to average map over
    cutoff : float, optional
        Cutoff to calculate contacts
    n : int, optional
        Number of threads to use.
        By default 4 to minimize RAM consumption with pure maps.
    exceptionsToIgnore : list of Exceptions
        List of exceptions to ignore when finding the contact map.
        Put IOError there if you want it to ignore missing files.

    Returns
    -------

    An NxN (for pure map) numpy array with the contact map.
    """

    """
    Now we actually need to modify our contact map by adding
    contacts from each new file to the contact map.
    We do it this way because our contact map is huge (maybe a gigabyte!),
    so we can't just add many gigabyte-sized arrays together.
    Instead of this each worker creates an empty "average contact map",
    and then loads files one by one and adds contacts from each file to a contact map.
    Maps from different workers are then added together manually.
    """

    n = min(n, len(filenames))
    subvalues = [filenames[i::n] for i in range(n)]

    def myaction(values):  # our worker receives some filenames
        mysum = None  # future contact map.
        for i in values:
            try:
                data = loadFunction(i)
                if np.random.random() < printProbability:
                    print(i)
            except tuple(exceptionsToIgnore):
                print("file not found", i)
                continue
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print("File is: ", i)
                return -1

            if data.shape[0] == 3:
                data = data.T
            if mysum is None:  # if it's the first filename,

                if len(data) > 6000:
                    warnings.warn(UserWarning('very large contact map'
                                              ' may cause errors. these may be fixed with n=1 threads.'))
                if len(data) > 20000:
                    warnings.warn(UserWarning('very large contact map'
                                              ' may be difficult to visualize.'))

                mysum = pureMap(data, cutoff)  # create a map

            else:  # if not
                pureMap(data, cutoff, mysum)
                # use existing map and fill in contacts

        return mysum

    blocks = fmap(myaction, subvalues)
    blocks = [i for i in blocks if i is not None]
    a = blocks[0]
    for i in blocks[1:]:
        a = a + i
    return a
